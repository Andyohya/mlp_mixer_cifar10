import jax.numpy as jnp

def evaluate(model, params, test_data):
    acc_total, count = 0, 0
    for imgs, labels in test_data:
        logits = model.apply(params, imgs)
        preds = jnp.argmax(logits, axis=-1)
        acc_total += jnp.sum(preds == labels)
        count += imgs.shape[0]
    return acc_total / count
