import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from src.config import *


def generate_sample():

    model = load_model(CHECKPOINT_DIR + "/generator_cgan.h5")

    noise = np.random.normal(0, 1, (1, LATENT_DIM))

    label = np.array([0])   # generate Crack image

    img = model.predict([noise, label])[0]

    img = (img + 1) / 2

    plt.imshow(img)

    plt.savefig(OUTPUT_DIR + "/generated_image.png")

    plt.show()