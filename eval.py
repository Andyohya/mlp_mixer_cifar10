import jax.numpy as jnp

def evaluate(model, params, test_data):
    accs = []
    for imgs, labels in test_data:
        logits = model.apply(params, imgs, train=False)  # ✅ 正確結構
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        accs.append(acc)
    return jnp.mean(jnp.array(accs))