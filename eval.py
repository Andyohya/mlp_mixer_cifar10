import jax.numpy as jnp

def evaluate(model, params, batch_stats, test_data):
    accs = []
    for imgs, labels in test_data:
        variables = {'params': params, 'batch_stats': batch_stats}
        logits = model.apply(variables, imgs, train=False, mutable=False)
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        accs.append(acc)
    return jnp.mean(jnp.array(accs))