import tensorflow as tf
import jax.numpy as jnp

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.247, 0.243, 0.261]

def preprocess(image, label, train=True):
    if train:
        image = tf.image.resize_with_crop_or_pad(image, 40, 40)
        image = tf.image.random_crop(image, [32, 32, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
    else:
        image = tf.image.resize(image, [32, 32])
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - CIFAR10_MEAN) / CIFAR10_STD
    return image, tf.squeeze(label)

def load_dataset(batch_size=16, train=True):
    (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    data = (x, y) if train else (x_test, y_test)
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.map(lambda img, lbl: preprocess(img, lbl, train))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return [(jnp.array(imgs), jnp.array(labels)) for imgs, labels in ds]