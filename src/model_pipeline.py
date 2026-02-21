import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from src.config import IMAGE_SIZE, DATA_DIR


def load_images():

    images = []
    labels = []

    class_names = sorted(os.listdir(DATA_DIR))

    for label, class_name in enumerate(class_names):

        class_path = os.path.join(DATA_DIR, class_name)

        for file in os.listdir(class_path):

            img = load_img(
                os.path.join(class_path, file),
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
            )

            img = img_to_array(img)

            img = (img / 127.5) - 1

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)