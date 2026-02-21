from tensorflow.keras import layers, Model
from src.config import LATENT_DIM, IMAGE_SIZE, CHANNELS, NUM_CLASSES


def build_generator():

    noise = layers.Input(shape=(LATENT_DIM,))
    label = layers.Input(shape=(1,), dtype="int32")

    label_embedding = layers.Embedding(NUM_CLASSES, LATENT_DIM)(label)
    label_embedding = layers.Flatten()(label_embedding)

    combined = layers.multiply([noise, label_embedding])

    x = layers.Dense(8 * 8 * 256)(combined)

    x = layers.Reshape((8, 8, 256))(x)

    for filters in [256, 128, 64, 32]:

        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    output = layers.Conv2D(CHANNELS, 3, activation="tanh", padding="same")(x)

    return Model([noise, label], output)



def build_discriminator():

    image = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    label = layers.Input(shape=(1,), dtype="int32")

    label_embedding = layers.Embedding(NUM_CLASSES, IMAGE_SIZE * IMAGE_SIZE)(label)
    label_embedding = layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 1))(label_embedding)

    combined = layers.Concatenate()([image, label_embedding])

    x = combined

    for filters in [64, 128, 256, 512]:

        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)

    output = layers.Dense(1, activation="sigmoid")(x)

    return Model([image, label], output)