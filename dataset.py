import tensorflow as tf
import jax.numpy as jnp

def preprocess(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.squeeze(label)

def load_dataset(batch_size=16, train=True):
    (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    data = (x, y) if train else (x_test, y_test)
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return [(jnp.array(imgs), jnp.array(labels)) for imgs, labels in ds]