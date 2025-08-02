import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from dataset import load_dataset
from model import MlpMixer

# 🔸 Cross entropy loss（加入 label smoothing）
def cross_entropy_loss(logits, labels, smoothing=0.1):
    num_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, num_classes)
    soft_labels = one_hot * (1 - smoothing) + smoothing / num_classes
    return optax.softmax_cross_entropy(logits, soft_labels).mean()

# 🔸 建立訓練狀態（包含學習率排程）
def create_train_state(rng, model, learning_rate, num_epochs=None, steps_per_epoch=None, warmup_steps=0, weight_decay=1e-2):
    params = model.init(rng, jnp.ones([1, 32, 32, 3]), train=True)
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
        tx = optax.adamw(schedule)

        print("📊 Learning rate schedule preview:")
        for i in range(0, min(20, total_steps), max(1, total_steps // 20)):
            print(f"Step {i}: lr = {float(schedule(i)):.6f}")
    else:
        tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    return state, schedule

# 🔸 單步訓練
@jax.jit
def train_step(state, batch):
    imgs, labels = batch

    def loss_fn(params):
        dropout_rng = jax.random.fold_in(jax.random.PRNGKey(0), state.step)
        logits = state.apply_fn(params, imgs, train=True, rngs={'dropout': dropout_rng})
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, {'loss': loss, 'accuracy': accuracy}

# 🔸 驗證步驟（不使用 JIT，避免 apply_fn 錯誤）
def eval_step(params, batch, apply_fn):
    imgs, labels = batch
    logits = apply_fn(params, imgs, train=False)  # ✅ 完整修正
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {'loss': loss, 'accuracy': accuracy}

# 🔸 EarlyStopping 機制
class EarlyStopping:
    def __init__(self, patience=5):
        self.best_loss = float('inf')
        self.counter = 0
        self.patience = patience

    def should_stop(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# 🔸 Train / Val 分割
def split_dataset(data, split_ratio=0.9):
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]