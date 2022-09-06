import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import tensorflow_datasets as tfds

train_ds, validation_ds = tfds.load(
    "tf_flowers", split=["train[:85%]", "train[85%:]"], as_supervised=True
)

IMAGE_SIZE = 224
NUM_IMAGES = 1000

images = []
labels = []

for (image, label) in train_ds.take(NUM_IMAGES):
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    images.append(image.numpy())
    labels.append(label.numpy())

images = np.array(images)
labels = np.array(labels)

'''
!wget -q https://git.io/JuMq0 -O flower_model_bit_0.96875.zip
!unzip -qq flower_model_bit_0.96875.zip
'''
#load pretrained model
bit_model = tf.keras.models.load_model("flower_model_bit_0.96875")
print(bit_model.count_params())

#create embedding model for getting feature representation using the pretrained model
embedding_model = tf.keras.Sequential(
    [
        tf.keras.layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.Rescaling(scale=1.0 / 255),
        bit_model.layers[1],
        tf.keras.layers.Normalization(mean=0, variance=1),
    ],
    name="embedding_model",
)

embedding_model.summary()
