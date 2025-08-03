import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import flax
from model import MlpMixer

class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict = None

def cross_entropy_loss(logits, labels, smoothing=0.1):
    num_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, num_classes)
    soft_labels = one_hot * (1 - smoothing) + smoothing / num_classes
    return optax.softmax_cross_entropy(logits, soft_labels).mean()

def l2_regularization(params):
    return sum([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)])

def create_train_state(rng, model, learning_rate, num_epochs=None, steps_per_epoch=None, warmup_steps=0, weight_decay=1e-4):
    variables = model.init(rng, jnp.ones([1, 32, 32, 3]), train=True)
    params = variables['params']
    batch_stats = variables.get('batch_stats')
    schedule = None

    if num_epochs and steps_per_epoch:
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = min(warmup_steps, total_steps - 1)
        decay_steps = max(1, total_steps - warmup_steps)
        if warmup_steps > 0:
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=0.0
            )
        else:
            schedule = optax.cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=total_steps
            )
        tx = optax.adamw(schedule, weight_decay=weight_decay)
    else:
        tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    )
    return state, schedule

@jax.jit
def train_step(state, batch, weight_decay=1e-4):
    imgs, labels = batch

    def loss_fn(params):
        dropout_rng = jax.random.fold_in(jax.random.PRNGKey(0), state.step)
        variables = {'params': params, 'batch_stats': state.batch_stats}
        outputs, new_model_state = state.apply_fn(
            variables, imgs, train=True, rngs={'dropout': dropout_rng}, mutable=['batch_stats']
        )
        logits = outputs
        ce_loss = cross_entropy_loss(logits, labels)
        l2_loss = l2_regularization(params)
        loss = ce_loss + weight_decay * l2_loss
        return loss, (logits, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state['batch_stats'])
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, {'loss': loss, 'accuracy': accuracy}

def eval_step(params, batch, apply_fn, batch_stats):
    imgs, labels = batch
    variables = {'params': params, 'batch_stats': batch_stats}
    logits = apply_fn(variables, imgs, train=False, mutable=False)
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {'loss': loss, 'accuracy': accuracy}

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):  # patience 放寬，min_delta 容忍
        self.best_loss = float('inf')
        self.counter = 0
        self.patience = patience
        self.min_delta = min_delta

    def should_stop(self, val_loss):
        # 只有 val_loss 比 best_loss 少 min_delta 才算進步
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def split_dataset(data, split_ratio=0.9):
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]