import optax
import jax
import jax.numpy as jnp
from flax.training import train_state

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    return optax.softmax_cross_entropy(logits, one_hot).mean()

# 新增warmup_cosine_decay_schedule
def create_train_state(rng, model, learning_rate, num_epochs=None, steps_per_epoch=None, warmup_steps=0):
    params = model.init(rng, jnp.ones([1, 32, 32, 3]))
    if num_epochs is not None and steps_per_epoch is not None:
        total_steps = num_epochs * steps_per_epoch
        # 防呆：warmup_steps 不可超過 total_steps-1
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
            schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=total_steps)
        tx = optax.adamw(schedule)
        # debug: 印出 schedule 前幾步
        print("Learning rate schedule preview:")
        for i in range(0, min(20, total_steps), max(1, total_steps // 20)):
            print(f"Step {i}: lr = {float(schedule(i)):.6f}")
    else:
        tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    imgs, labels = batch

    def loss_fn(params):
        logits = state.apply_fn(params, imgs)
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, {'loss': loss, 'accuracy': accuracy}