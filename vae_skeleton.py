"""
Visualize latent space rep --> make it square!
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers
import matplotlib.pyplot as plt
import sys ;
import math

latent_dim = 25
EPOCHS = 100 ;


"""
## Create a sampling layer
"""


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

"""
## Define the VAE as a `Model` with a custom `train_step`
"""
class VAE(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ## Build the decoder
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        # ---
        #decoder_outputs_mean = layers.Reshape((784,))(layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x))
        #decoder_outputs_logvar = layers.Reshape((784,))(layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x))
        #sampled = Sampling()([decoder_outputs_mean, decoder_outputs_logvar]) ;
        #decoder_outputs = layers.Reshape((28,28,1))(sampled) ;
        # ---
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        # ---
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        self.decoder.build(input_shape = (latent_dim,)) ; 
        self.decoder.summary()

        ## Build the encoder
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder.build(input_shape = (28,28,1)) ; 
        self.encoder.summary() ;

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            #reconstruction_loss = ops.mean(
            #    ops.sum(
            #        keras.losses.binary_crossentropy(data, reconstruction),
            #        axis=(1, 2),
            #    )
            #)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum((data-reconstruction)**2, axis=(1,2,3))) ;
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)
        return {
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }


def pca_2d(x: np.ndarray) -> np.ndarray:
  centered = x - np.mean(x, axis=0, keepdims=True)
  _, _, vh = np.linalg.svd(centered, full_matrices=False)
  components = vh[:2].T
  return centered @ components


if __name__ == "__main__":
  if len(sys.argv) < 2:
    raise ValueError("Usage: python vae_skeleton.py train|test")

  (x_train, t_train), (x_test, t_test) = keras.datasets.mnist.load_data() ;
  mnist_digits_train = np.reshape(x_train, (-1, 28,28,1)).astype("float32") / 255.
  mnist_digits_test = np.reshape(x_test, (-1, 28,28,1)).astype("float32") / 255.
  mnist_digits_train_0 = mnist_digits_train[t_train == 0] ;
  mnist_digits_test_0 = mnist_digits_test[t_test == 0] ;
  mnist_digits_train_1 = mnist_digits_train[t_train == 1] ;
  mnist_digits_test_1 = mnist_digits_test[t_test == 1] ;

  vae = VAE()
  vae.compile(optimizer=keras.optimizers.Adam())
  os.makedirs("outputs", exist_ok=True)
  if sys.argv[1] == "train":
    vae.fit(mnist_digits_train_0, epochs=EPOCHS, batch_size=128)
    vae.encoder.save_weights("vae_enc.weights.h5") ;
    vae.decoder.save_weights("vae_dec.weights.h5") ;
  else:
    vae.decoder.load_weights("vae_dec.weights.h5") ;
    vae.encoder.load_weights("vae_enc.weights.h5") ;

  # ----- visualize latent vectors for 100 class 0/1 test samples -----
  n_per_class = 50
  vis_samples = np.concatenate([mnist_digits_test_0[:n_per_class], mnist_digits_test_1[:n_per_class]], axis=0)
  vis_labels = np.array([0] * n_per_class + [1] * n_per_class)
  vis_z_mean, _, vis_z = vae.encoder.predict(vis_samples, verbose=0)

  side = int(math.sqrt(latent_dim))
  if side * side != latent_dim:
    raise ValueError("latent_dim must be a perfect square for direct latent visualization.")

  rows, cols = 10, 10
  fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
  for idx, ax in enumerate(axes.flat):
    z_img = vis_z[idx].reshape(side, side)
    ax.imshow(z_img, cmap="viridis")
    ax.set_title(f"y={vis_labels[idx]}", fontsize=8)
    ax.axis("off")
  plt.tight_layout()
  plt.savefig("outputs/vae_latent_vectors_0_1.png", dpi=150)
  plt.close(fig)
  print("Saved: outputs/vae_latent_vectors_0_1.png")

  vis_z_2d = pca_2d(vis_z_mean)
  plt.figure(figsize=(7, 6))
  plt.scatter(vis_z_2d[vis_labels == 0, 0], vis_z_2d[vis_labels == 0, 1], s=20, alpha=0.8, label="Class 0")
  plt.scatter(vis_z_2d[vis_labels == 1, 0], vis_z_2d[vis_labels == 1, 1], s=20, alpha=0.8, label="Class 1")
  plt.xlabel("PC1")
  plt.ylabel("PC2")
  plt.title("Latent Space Scatter (PCA of z_mean)")
  plt.legend()
  plt.tight_layout()
  plt.savefig("outputs/vae_latent_scatter_0_1.png", dpi=150)
  plt.close()
  print("Saved: outputs/vae_latent_scatter_0_1.png")

  # ----- reconstruction MSE histogram + best threshold (1000 samples) -----
  eval_per_class = 500
  eval_samples = np.concatenate([mnist_digits_test_0[:eval_per_class], mnist_digits_test_1[:eval_per_class]], axis=0)
  eval_labels = np.array([0] * eval_per_class + [1] * eval_per_class)
  _, _, eval_z = vae.encoder.predict(eval_samples, verbose=0)
  eval_recon = vae.decoder.predict(eval_z, verbose=0)
  mse = np.mean((eval_samples - eval_recon) ** 2, axis=(1, 2, 3))

  plt.figure(figsize=(8, 5))
  plt.hist(mse[eval_labels == 0], bins=40, alpha=0.6, label="Class 0")
  plt.hist(mse[eval_labels == 1], bins=40, alpha=0.6, label="Class 1")
  plt.xlabel("Reconstruction MSE")
  plt.ylabel("Count")
  plt.title("VAE Reconstruction Error Distribution by Class")
  plt.legend()
  plt.tight_layout()
  plt.savefig("outputs/vae_mse_hist_0_1.png", dpi=150)
  plt.close()
  print("Saved: outputs/vae_mse_hist_0_1.png")

  plt.figure(figsize=(7, 5))
  plt.boxplot([mse[eval_labels == 0], mse[eval_labels == 1]], labels=["Class 0", "Class 1"], showfliers=True)
  plt.ylabel("Reconstruction MSE")
  plt.title("Class-wise Reconstruction Error Boxplot")
  plt.tight_layout()
  plt.savefig("outputs/vae_error_boxplot_0_1.png", dpi=150)
  plt.close()
  print("Saved: outputs/vae_error_boxplot_0_1.png")

  recon_n = 8
  recon_samples = np.concatenate([mnist_digits_test_0[:recon_n], mnist_digits_test_1[:recon_n]], axis=0)
  recon_labels = np.array([0] * recon_n + [1] * recon_n)
  _, _, recon_z = vae.encoder.predict(recon_samples, verbose=0)
  recon_out = vae.decoder.predict(recon_z, verbose=0)
  recon_err = np.abs(recon_samples - recon_out)

  rows, cols = 3, 2 * recon_n
  fig, axes = plt.subplots(rows, cols, figsize=(2.0 * recon_n, 5))
  for i in range(2 * recon_n):
    axes[0, i].imshow(recon_samples[i, :, :, 0], cmap="gray")
    axes[0, i].set_title(f"y={recon_labels[i]}", fontsize=8)
    axes[0, i].axis("off")
    axes[1, i].imshow(recon_out[i, :, :, 0], cmap="gray")
    axes[1, i].axis("off")
    axes[2, i].imshow(recon_err[i, :, :, 0], cmap="hot")
    axes[2, i].axis("off")
  axes[0, 0].set_ylabel("Original", fontsize=10)
  axes[1, 0].set_ylabel("Reconst", fontsize=10)
  axes[2, 0].set_ylabel("|Error|", fontsize=10)
  plt.tight_layout()
  plt.savefig("outputs/vae_reconstruction_grid_0_1.png", dpi=150)
  plt.close(fig)
  print("Saved: outputs/vae_reconstruction_grid_0_1.png")

  thresholds = np.unique(mse)
  best_acc = -1.0
  best_t = thresholds[0]
  best_stats = (0, 0, 0, 0)  # TP, TN, FP, FN
  for t in thresholds:
    pred = (mse >= t).astype(np.int32)  # high MSE -> class 1
    tp = int(np.sum((pred == 1) & (eval_labels == 1)))
    tn = int(np.sum((pred == 0) & (eval_labels == 0)))
    fp = int(np.sum((pred == 1) & (eval_labels == 0)))
    fn = int(np.sum((pred == 0) & (eval_labels == 1)))
    acc = (tp + tn) / len(eval_labels)
    if acc > best_acc:
      best_acc = acc
      best_t = float(t)
      best_stats = (tp, tn, fp, fn)

  tp, tn, fp, fn = best_stats
  print(f"Best threshold: {best_t:.8f}")
  print(f"Best accuracy: {best_acc:.4f}")
  print(f"Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
