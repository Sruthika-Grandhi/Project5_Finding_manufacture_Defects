import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from src.config import *


def deploy():
    model = load_model(CHECKPOINT_DIR + "/generator_cgan.h5")

    noise = np.random.normal(0, 1, (1, LATENT_DIM))

    image = model.predict(noise)[0]

    image = (image + 1) / 2

    plt.imshow(image)

    plt.title("Deployment Output")

    plt.show()