import tensorflow as tf ;
import numpy as np ;
import sys ;
import matplotlib.pyplot as plt ;
import math ;
import os ;

LATENT_DIM = 25 ;
EPOCHS=10 ;

# subclassing keras functional model
class AE(tf.keras.Model):
  def __init__(self, **kwargs):
	# mandatory
    tf.keras.Model.__init__(self, **kwargs) ;

    # define encoder
    inp = tf.keras.Input(shape=(784,)) ;
    x = tf.keras.layers.Dense(50)(inp) ;
    out = tf.keras.layers.Dense(LATENT_DIM)(x) ;
    self.encoder = tf.keras.Model(inp, out) ;
    self.encoder.build(input_shape=(784,)) ;

    # define decoder
    inp = tf.keras.Input(shape=(LATENT_DIM,)) ;
    x = tf.keras.layers.Dense(50)(inp) ;
    out = tf.keras.layers.Dense(784)(x) ;
    self.decoder = tf.keras.Model(inp, out) ;
    self.decoder.build(input_shape=(784,)) ;

    self.reconstruction_loss_tracker = tf.keras.metrics.Mean() ;

  @property
  def metrics(self):
        return [
            self.reconstruction_loss_tracker 
        ]

  def train_step(self, d, **kwargs):
    with tf.GradientTape() as g:
      latent = self.encoder(d) ;
      reconstruction = self.decoder(latent) ;
      recloss = tf.reduce_mean((d-reconstruction)**2) ;
    # self.trainable_variables gets defined by compile()
    grads = g.gradient(recloss, self.trainable_weights) ;
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights) ) ;
    self.reconstruction_loss_tracker.update_state(recloss) ;
    # train_step should return a dictionary of metrics, displayed during fit()
    return {"rec_loss":self.reconstruction_loss_tracker.result() } ;

if __name__=="__main__":
  if len(sys.argv) < 2:
    raise ValueError("Usage: python ae_skeleton.py train|test")

  (x_train, t_train), (x_test, t_test) = tf.keras.datasets.mnist.load_data()
  scalar_labels_train = t_train ;
  scalar_labels_test = t_test ;
  train_digits = np.reshape(x_train, (-1,784)).astype("float32") / 255. ;
  test_digits = np.reshape(x_test, (-1,784)).astype("float32") / 255. ;
  train_digits_0 = train_digits[scalar_labels_train==0] ;
  print(train_digits_0.shape) ;
  test_digits0 = test_digits[scalar_labels_test==0] ;
  test_digits1 = test_digits[scalar_labels_test==1] ;

  ae = AE() ;
  ae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01), run_eagerly = False) ;
  os.makedirs("outputs", exist_ok=True) ;
  if sys.argv[1] == "train":
    ae.fit(train_digits_0, epochs = EPOCHS, batch_size = 100) ;
    ae.encoder.save_weights("ae1.weights.h5") ;
    ae.decoder.save_weights("ae2.weights.h5") ;
  else:
    ae.encoder.load_weights("ae1.weights.h5") ;
    ae.decoder.load_weights("ae2.weights.h5") ;
  
  # ----- latent visualization for 100 class 0/1 samples -------------
  vis_per_class = 50
  vis_samples = np.concatenate([test_digits0[:vis_per_class], test_digits1[:vis_per_class]], axis=0)
  vis_labels = np.array([0] * vis_per_class + [1] * vis_per_class)
  vis_latent = ae.encoder.predict(vis_samples, verbose=0)

  side = int(math.sqrt(LATENT_DIM))
  if side * side != LATENT_DIM:
    raise ValueError("LATENT_DIM must be a perfect square for latent grid visualization.")

  fig, axes = plt.subplots(10, 10, figsize=(12, 12))
  for i, ax in enumerate(axes.flat):
    z_img = vis_latent[i].reshape(side, side)
    ax.imshow(z_img, cmap="viridis")
    ax.set_title(f"y={vis_labels[i]}", fontsize=8)
    ax.axis("off")
  plt.tight_layout()
  plt.savefig("outputs/ae_skeleton_latent_vectors_0_1.png", dpi=150)
  plt.close(fig)
  print("Saved: outputs/ae_skeleton_latent_vectors_0_1.png")

  # ----- reconstruction error + histogram for 1000 class 0/1 samples -----
  eval_per_class = 500
  eval_samples = np.concatenate([test_digits0[:eval_per_class], test_digits1[:eval_per_class]], axis=0)
  eval_labels = np.array([0] * eval_per_class + [1] * eval_per_class)
  eval_latent = ae.encoder.predict(eval_samples, verbose=0)
  eval_recon = ae.decoder.predict(eval_latent, verbose=0)
  mse = np.mean((eval_samples - eval_recon) ** 2, axis=1)

  plt.figure(figsize=(8, 5))
  plt.hist(mse[eval_labels == 0], bins=40, alpha=0.6, label="Class 0")
  plt.hist(mse[eval_labels == 1], bins=40, alpha=0.6, label="Class 1")
  plt.xlabel("Reconstruction MSE")
  plt.ylabel("Count")
  plt.title("AE Reconstruction Error Distribution by Class")
  plt.legend()
  plt.tight_layout()
  plt.savefig("outputs/ae_skeleton_mse_hist_0_1.png", dpi=150)
  plt.close()
  print("Saved: outputs/ae_skeleton_mse_hist_0_1.png")

  # ----- original / reconstruction / error panel -----
  recon_n = 8
  recon_samples = np.concatenate([test_digits0[:recon_n], test_digits1[:recon_n]], axis=0)
  recon_labels = np.array([0] * recon_n + [1] * recon_n)
  recon_latent = ae.encoder.predict(recon_samples, verbose=0)
  recon_out = ae.decoder.predict(recon_latent, verbose=0)
  recon_err = np.abs(recon_samples - recon_out)

  fig, axes = plt.subplots(3, 2 * recon_n, figsize=(2.0 * recon_n, 5))
  for i in range(2 * recon_n):
    axes[0, i].imshow(recon_samples[i].reshape(28, 28), cmap="gray")
    axes[0, i].set_title(f"y={recon_labels[i]}", fontsize=8)
    axes[0, i].axis("off")
    axes[1, i].imshow(recon_out[i].reshape(28, 28), cmap="gray")
    axes[1, i].axis("off")
    axes[2, i].imshow(recon_err[i].reshape(28, 28), cmap="hot")
    axes[2, i].axis("off")
  axes[0, 0].set_ylabel("Original", fontsize=10)
  axes[1, 0].set_ylabel("Reconst", fontsize=10)
  axes[2, 0].set_ylabel("|Error|", fontsize=10)
  plt.tight_layout()
  plt.savefig("outputs/ae_skeleton_reconstruction_grid_0_1.png", dpi=150)
  plt.close(fig)
  print("Saved: outputs/ae_skeleton_reconstruction_grid_0_1.png")

  # ----- best threshold summary -----
  thresholds = np.unique(mse)
  best_acc = -1.0
  best_t = thresholds[0]
  best_stats = (0, 0, 0, 0)  # TP, TN, FP, FN
  for t in thresholds:
    pred = (mse >= t).astype(np.int32)
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
