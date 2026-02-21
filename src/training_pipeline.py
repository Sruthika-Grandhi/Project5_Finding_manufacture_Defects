import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.config import *
from src.gan_architecture import build_generator, build_discriminator
from src.model_pipeline import load_images
from src.monitoring import save_loss_graph


def train():

    print("\nLoading Dataset...")
    images, labels = load_images()

    print("Dataset Loaded:", images.shape)

    generator = build_generator()
    discriminator = build_discriminator()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=BETA_1
    )

    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    noise = tf.keras.Input(shape=(LATENT_DIM,))
    label = tf.keras.Input(shape=(1,))

    fake_img = generator([noise, label])

    discriminator.trainable = False

    validity = discriminator([fake_img, label])

    gan = tf.keras.Model([noise, label], validity)

    gan.compile(
        loss="binary_crossentropy",
        optimizer=optimizer
    )

    real = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))

    g_losses = []
    d_losses = []

    print("\nStarting Training...\n")

    for epoch in range(EPOCHS):

        idx = np.random.randint(0, images.shape[0], BATCH_SIZE)

        real_imgs = images[idx]
        labels_batch = labels[idx]

        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

        fake_imgs = generator.predict(
            [noise, labels_batch],
            verbose=0
        )

        d_loss_real = discriminator.train_on_batch(
            [real_imgs, labels_batch], real
        )

        d_loss_fake = discriminator.train_on_batch(
            [fake_imgs, labels_batch], fake
        )

        d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

        g_loss = gan.train_on_batch(
            [noise, labels_batch], real
        )

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"| D Loss: {d_loss:.4f} "
            f"| G Loss: {g_loss:.4f}"
        )

        g_losses.append(g_loss)
        d_losses.append(d_loss)

        # SAVE IMAGE EVERY 10 EPOCHS
        if epoch % 10 == 0:

            sample_noise = np.random.normal(
                0, 1, (1, LATENT_DIM)
            )

            sample_label = np.array([labels_batch[0]])

            gen_img = generator.predict(
                [sample_noise, sample_label],
                verbose=0
            )[0]

            gen_img = (gen_img + 1) / 2

            plt.imshow(gen_img)
            plt.axis("off")

            plt.savefig(
                OUTPUT_DIR + f"/epoch_{epoch}.png"
            )

            plt.close()

    print("\nTraining Completed")

    print("\nSaving Model...")

    generator.save(
        CHECKPOINT_DIR + "/generator_cgan.h5"
    )

    print("Model Saved")

    save_loss_graph(
        g_losses,
        d_losses
    )

    print("Loss Graph Saved")