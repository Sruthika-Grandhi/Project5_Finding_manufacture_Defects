import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR


def save_loss_graph(g_loss, d_loss):

    plt.plot(g_loss, label="Generator")

    plt.plot(d_loss, label="Discriminator")

    plt.legend()

    plt.title("GAN Loss")

    plt.savefig(OUTPUT_DIR + "/loss.png")

    plt.show()