#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This "project_2_sVAE_1" is the codes for saving the model with parameters of best results 


# In[1]:


import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth set.")
    except RuntimeError as e:
        print(e)


# In[2]:


import os
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.activations import swish
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import swish
from tensorflow.image import ssim

print("finished preparing")


# In[ ]:


from project_config import DATASET_NPZ, GRID_RESULTS
#DATASET_NPZ   = f"{BASE_DIR_e}/all_mri_psyc_dataset_final.npz"
#GRID_RESULTS  = f"{BASE_DIR_e}/grid_results_npz"


# In[3]:


loaded = np.load(DATASET_NPZ, allow_pickle=True)
data = loaded['mri']
psyc_scores = loaded['scores']

print("List of keys     :", loaded.files)
print("")
print("data.shape       :", data.shape)
print("Max value:  {:.5f}".format(data.max()))
print("Min value: {:.5f}".format(data.min()))
print("Mean value: {:.5f}".format(data.mean()))
print("Std deviation: {:.5f}".format(data.std()))
print("")
print("psyc_scores.shape :", psyc_scores.shape)
print("Max value: {:.5f}".format(psyc_scores.max()))
print("Min value: {:.5f}".format(psyc_scores.min()))
print("Mean value: {:.5f}".format(psyc_scores.mean()))
print("Std deviation: {:.5f}".format(psyc_scores.std()))

print("Min:", tf.reduce_min(data).numpy())
print("Max:", tf.reduce_max(data).numpy())
print("Has NaN:", tf.math.reduce_any(tf.math.is_nan(data)).numpy())


# In[4]:


# Define encoder, decoder and regressor

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import Layer

# --- GroupNormalization ---
class GroupNormalization(Layer):
    def __init__(self, groups=32, axis=-1, epsilon=1e-5, **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError("Axis dimension must be defined")
        if dim % self.groups != 0:
            raise ValueError(f"Number of channels ({dim}) must be divisible by groups ({self.groups})")

        self.gamma = self.add_weight(shape=(dim,), initializer="ones", trainable=True, name="gamma")
        self.beta = self.add_weight(shape=(dim,), initializer="zeros", trainable=True, name="beta")

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        channels = inputs.shape[self.axis]
        group_size = channels // self.groups

        # reshape: [B, D, H, W, C] → [B, D, H, W, groups, group_size]
        reshaped = tf.reshape(inputs, [batch_size, -1, self.groups, group_size])
        mean, var = tf.nn.moments(reshaped, axes=[-1], keepdims=True)
        normalized = (reshaped - mean) / tf.sqrt(var + self.epsilon)

        # reshape back
        normalized = tf.reshape(normalized, input_shape)
        return self.gamma * normalized + self.beta

def auto_group_norm(x, max_groups=32, min_channels_per_group=8):
    channels = x.shape[-1]
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0 and channels // g >= min_channels_per_group:
            return GroupNormalization(groups=g)(x)
    return GroupNormalization(groups=1)(x)

swish = tf.keras.activations.swish

# --- Sampling Layer ---
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# --- Encoder ---
def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv3D(32, 3, activation=None, strides=2, padding='same',  kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = auto_group_norm(x)
    x = layers.Activation(swish)(x)
    x = layers.Conv3D(64, 3, activation=None, strides=2, padding='same',  kernel_regularizer=regularizers.l2(1e-4))(x)
    x = auto_group_norm(x)
    x = layers.Activation(swish)(x)
    x = layers.Conv3D(128, 3, activation=None, strides=2, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = auto_group_norm(x)
    x = layers.Activation(swish)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation=None, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = auto_group_norm(x)
    x = layers.Activation(swish)(x)
    z_mean = layers.Dense(latent_dim, kernel_regularizer=regularizers.l2(1e-4))(x)
    z_log_var = layers.Dense(latent_dim, kernel_regularizer=regularizers.l2(1e-4))(x)
    z = Sampling()([z_mean, z_log_var])
    return models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

# --- Decoder ---
def build_decoder(latent_dim):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(12*14*12*128, activation=None, kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=regularizers.l2(1e-4))(latent_inputs)
    x = auto_group_norm(x)
    x = layers.Activation(swish)(x)
    x = layers.Reshape((12,14,12,128))(x)

    x = layers.Conv3DTranspose(128, 3, activation=None, strides=2, padding='same', kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = auto_group_norm(x)
    x = layers.Activation(swish)(x)

    x = layers.Conv3DTranspose(64, 3, activation=None, strides=2, padding='same',  kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = auto_group_norm(x)
    x = layers.Activation(swish)(x)

    x = layers.Conv3DTranspose(32, 3, activation=None, strides=2, padding='same',  kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = auto_group_norm(x)
    x = layers.Activation(swish)(x)

    x = layers.Cropping3D(cropping=((2,3),(1,2),(2,3)))(x)
    x = layers.Conv3DTranspose(1, 3, activation='linear', padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(x)

    return models.Model(latent_inputs, x, name='decoder')

# --- Regression ---
# --- Elastic Net ---
def build_regression_head(z, output_dim=3, l1_reg=1e-5, l2_reg=1e-4, dropout_rate=0.3):
    reg = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)

    x = layers.Dense(64, activation=None, kernel_regularizer=reg)(z)
    x = auto_group_norm(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(32, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)

    output = layers.Dense(output_dim, dtype='float32')(x)
    return output


# In[5]:


# --- Define Supervised VAE ---

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

import tensorflow as tf

def correlation_loss(z, y_true):
    z_centered = z - tf.reduce_mean(z, axis=0, keepdims=True)
    y_centered = y_true - tf.reduce_mean(y_true, axis=0, keepdims=True)
    cov = tf.matmul(tf.transpose(z_centered), y_centered) / tf.cast(tf.shape(z_centered)[0], tf.float32)
    std_z = tf.sqrt(tf.reduce_mean(tf.square(z_centered), axis=0, keepdims=True) + 1e-8)
    std_y = tf.sqrt(tf.reduce_mean(tf.square(y_centered), axis=0, keepdims=True) + 1e-8)
    corr_matrix = cov / (tf.transpose(std_z) * std_y)
    mean_corr = tf.reduce_mean(tf.abs(corr_matrix))
    return -mean_corr

def covariance_regularization(z):
    z_centered = z - tf.reduce_mean(z, axis=0)
    cov = tf.matmul(z_centered, z_centered, transpose_a=True) / tf.cast(tf.shape(z_centered)[0], tf.float32)
    off_diag = cov - tf.linalg.diag(tf.linalg.diag_part(cov))
    return tf.reduce_mean(tf.square(off_diag))

class SupervisedVAE(keras.Model):
    def __init__(self, encoder, decoder, regressor,
                 regression_weight=1.0,
                 accumulation_steps=1,
                 beta_start=0.0, beta_end=1.0, beta_anneal_epochs=10):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.regressor = regressor
        self.regression_weight = regression_weight

        self.accumulation_steps = accumulation_steps
        self._gradient_accumulation_buffer = None
        self._gradient_accumulation_counter = 0

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_epochs = beta_anneal_epochs
        self.current_epoch = 0

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.regression_loss_tracker = keras.metrics.Mean(name="regression_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.regression_loss_tracker
        ]

    def compute_beta(self):
        if self.current_epoch < self.beta_anneal_epochs:
            return self.beta_start + (self.beta_end - self.beta_start) * (
                self.current_epoch / self.beta_anneal_epochs
            )
        else:
            return self.beta_end

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        y_pred = self.regressor(z)
        return reconstruction, y_pred, z_mean, z_log_var, z

    def train_step(self, data):

        x, y_true = data
        mask = tf.cast(x != 0, tf.float32)
        self.beta = self.compute_beta()

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            y_pred = self.regressor(z)

            reconstruction_loss = tf.reduce_sum(tf.square(x - reconstruction) * mask) / tf.reduce_sum(mask)
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            regression_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            corr_loss = correlation_loss(z, y_true)
            cov_loss = covariance_regularization(z)

            total_loss = reconstruction_loss + self.beta * kl_loss + self.regression_weight * regression_loss + 0.1 * corr_loss + 0.01 * cov_loss
            scaled_loss = total_loss / self.accumulation_steps

        grads = tape.gradient(scaled_loss, self.trainable_weights)
        grads = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in grads]

        if self._gradient_accumulation_buffer is None:
            self._gradient_accumulation_buffer = [tf.identity(g) if g is not None else None for g in grads]
        else:
            new_buffer = []
            for gb, g in zip(self._gradient_accumulation_buffer, grads):
                if gb is not None and g is not None:
                    new_buffer.append(gb + g)
                elif gb is not None:
                    new_buffer.append(gb)
                elif g is not None:
                    new_buffer.append(tf.identity(g))
                else:
                    new_buffer.append(None)
            self._gradient_accumulation_buffer = new_buffer

        self._gradient_accumulation_counter += 1

        if self._gradient_accumulation_counter >= self.accumulation_steps:
            grads_to_apply = []
            weights_to_apply = []
            for g, w in zip(self._gradient_accumulation_buffer, self.trainable_weights):
                if g is not None:
                    grads_to_apply.append(tf.clip_by_norm(g / self.accumulation_steps, 5.0))
                    weights_to_apply.append(w)

            if grads_to_apply:
                self.optimizer.apply_gradients(zip(grads_to_apply, weights_to_apply))

            self._gradient_accumulation_buffer = None
            self._gradient_accumulation_counter = 0

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.regression_loss_tracker.update_state(regression_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "regression_loss": self.regression_loss_tracker.result()
        }

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch

    def test_step(self, data):
        self.total_loss_tracker.reset_state()
        self.reconstruction_loss_tracker.reset_state()
        self.kl_loss_tracker.reset_state()
        self.regression_loss_tracker.reset_state()

        x, y_true = data
        mask = tf.cast(x != 0, tf.float32)

        z_mean, z_log_var, z = self.encoder(x, training=False)
        x_recon = self.decoder(z, training=False)
        y_pred = self.regressor(z, training=False)

        reconstruction_loss = tf.reduce_sum(tf.square(x - x_recon) * mask) / tf.reduce_sum(mask)
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        regression_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        corr_loss = correlation_loss(z, y_true)
        cov_loss = covariance_regularization(z)

        total_loss = reconstruction_loss + self.beta * kl_loss + self.regression_weight * regression_loss + 0.1 * corr_loss + 0.01 * cov_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.regression_loss_tracker.update_state(regression_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "regression_loss": self.regression_loss_tracker.result()
        }

# --- Beta Annealing Callback ---
class BetaAnnealingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.current_epoch = epoch + 1
        if hasattr(self.model, 'compute_beta'):
            self.model.beta = self.model.compute_beta()

# === Regression_weigth Annealing Callback ===
class RegressionAnnealingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if hasattr(self.model, "regression_weight_start") and hasattr(self.model, "regression_weight_end"):
            anneal_epochs = self.model.beta_anneal_epochs  # βと同じ期間で線形変化
            start = self.model.regression_weight_start
            end = self.model.regression_weight_end
            if epoch < anneal_epochs:
                self.model.regression_weight = start + (end - start) * (epoch / anneal_epochs)
            else:
                self.model.regression_weight = end


# In[6]:


from tensorflow.keras.callbacks import Callback
import numpy as np
import os

class LatentAndPredictionNPZLogger(Callback):
    def __init__(self, encoder, regressor, x_val, y_val, save_path):
        super().__init__()
        self.encoder = encoder
        self.regressor = regressor
        self.x_val = x_val
        self.y_val = y_val
        self.save_path = save_path

        # list for save
        self.z_mean_list = []
        self.z_log_var_list = []
        self.z_sample_list = []
        self.y_pred_list = []
        self.y_true_list = []
        self.recon_loss_list = []
        self.val_recon_loss_list = []
        self.reg_loss_list = []
        self.val_reg_loss_list = []

    def on_epoch_end(self, epoch, logs=None):
        z_mean, z_log_var, z_sample = self.encoder.predict(self.x_val, batch_size=1, verbose=0)

        y_pred = self.regressor.predict(z_mean, batch_size=1, verbose=0)

        self.z_mean_list.append(z_mean)
        self.z_log_var_list.append(z_log_var)
        self.z_sample_list.append(z_sample)
        self.y_pred_list.append(y_pred)
        self.y_true_list.append(self.y_val)
        self.recon_loss_list.append(logs.get("reconstruction_loss"))
        self.val_recon_loss_list.append(logs.get("val_reconstruction_loss"))
        self.reg_loss_list.append(logs.get("regression_loss"))
        self.val_reg_loss_list.append(logs.get("val_regression_loss"))

    def on_train_end(self, logs=None):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        np.savez(
            self.save_path,
            z_mean=np.array(self.z_mean_list),
            z_log_var=np.array(self.z_log_var_list),
            z_sample=np.array(self.z_sample_list),
            y_pred=np.array(self.y_pred_list),
            y_true=np.array(self.y_true_list),
            reconstruction_loss=np.array(self.recon_loss_list),
            val_reconstruction_loss=np.array(self.val_recon_loss_list),
            regression_loss=np.array(self.reg_loss_list),
            val_regression_loss=np.array(self.val_reg_loss_list)
        )
        print(f"All epochs saved to {self.save_path}")


# In[7]:


# StratifiedKFold

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

n_bins = 7
mmse_bins = pd.qcut(psyc_scores[:, 0], q=n_bins, labels=False, duplicates='drop')
fab_bins  = pd.qcut(psyc_scores[:, 1], q=n_bins, labels=False, duplicates='drop')
rcpm_bins = pd.qcut(psyc_scores[:, 2], q=n_bins, labels=False, duplicates='drop')

y_binned = mmse_bins.astype(str) + "_" + fab_bins.astype(str) + "_" + rcpm_bins.astype(str)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_binned)), y_binned), 1):
    print(f"Fold {fold}")
    print(f"  Train MMSE mean: {psyc_scores[train_idx,0].mean():.3f}")
    print(f"  Train FAB  mean: {psyc_scores[train_idx,1].mean():.3f}")
    print(f"  Train RCPM mean: {psyc_scores[train_idx,2].mean():.3f}")
    print(f"  Val   MMSE mean: {psyc_scores[val_idx,0].mean():.3f}")
    print(f"  Val   FAB  mean: {psyc_scores[val_idx,1].mean():.3f}")
    print(f"  Val   RCPM mean: {psyc_scores[val_idx,2].mean():.3f}")

score_names = ["MMSE", "FAB", "RCPM"]

for i, name in enumerate(score_names):
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_binned)), y_binned), 1):
        plt.figure(figsize=(6,4))
        plt.hist(psyc_scores[train_idx, i], bins=15, alpha=0.6, label='Train', density=True)
        plt.hist(psyc_scores[val_idx, i], bins=15, alpha=0.6, label='Validation', density=True)
        plt.title(f"{name} - Fold {fold}")
        plt.xlabel(f"{name} Score")
        plt.ylabel("Density")
        plt.legend()
        plt.show()


# In[8]:


# Train and save
# From previous training, the best parameters and epoch have been found.

import os
import gc
import numpy as np
import tensorflow as tf
from itertools import product
from sklearn.model_selection import StratifiedKFold

beta_end_list = [1.0]
regression_weight_start_list = [0.1]
regression_weight_end_list = [1.0]

#save_dir = "/mnt/e/T1_2024/grid_results_npz"
save_dir = GRID_RESULTS
os.makedirs(save_dir, exist_ok=True)

for beta_end, reg_start, reg_end in product(beta_end_list, regression_weight_start_list, regression_weight_end_list):
    print(f"\nStart Grid Search: β={beta_end}, λ_start={reg_start}, λ_end={reg_end}")

    target_folds = [1]  # only Fold 1

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, y_binned)):
        if (fold + 1) not in target_folds:
            continue

        print(f"\nFold {fold+1}/5 ----------------------------")

        gc.collect()
        tf.keras.backend.clear_session()

        x_train, x_val = data[train_idx], data[val_idx]
        y_train, y_val = psyc_scores[train_idx], psyc_scores[val_idx]
        x_train = x_train[..., np.newaxis]
        x_val = x_val[..., np.newaxis]

        latent_dim = 64
        encoder = build_encoder((91, 109, 91, 1), latent_dim)
        decoder = build_decoder(latent_dim)
        z_mean_input = tf.keras.layers.Input(shape=(latent_dim,))
        regression_output = build_regression_head(z_mean_input, output_dim=3)
        regressor = tf.keras.models.Model(z_mean_input, regression_output, name="regressor")

        svae = SupervisedVAE(
            encoder, decoder, regressor,
            beta_start=1e-4, beta_end=beta_end, beta_anneal_epochs=50,
            accumulation_steps=16
        )

        # --- Regression weight Annealing ---
        svae.regression_weight_start = reg_start
        svae.regression_weight_end = reg_end
        svae.regression_weight = reg_start

        svae.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
            run_eagerly=True,
            loss={'regression_output': tf.keras.losses.Huber(delta=1.0)}
        )

        # --- Callbacks ---
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_regression_loss', patience=100, restore_best_weights=True, mode='min'
        )
        beta_annealing_cb = BetaAnnealingCallback()
        regression_anneal_cb = RegressionAnnealingCallback()

        save_path = os.path.join(save_dir, f"latent_progress_beta{beta_end}_lambda{reg_start}-{reg_end}_fold{fold+1}.npz")
        latent_logger = LatentAndPredictionNPZLogger(encoder, regressor, x_val, y_val, save_path=save_path)

        callbacks = [latent_logger, early_stopping_cb, beta_annealing_cb, regression_anneal_cb]

        svae.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=66,
            batch_size=1,
            callbacks=callbacks,
            verbose=1
        )

        _ = svae(x_train[:1], training=False)  # build

        model_save_path = os.path.join(save_dir, f"model_epoch66_beta{beta_end}_lambda{reg_start}-{reg_end}_fold{fold+1}.weights.h5")
        encoder_path = os.path.join(save_dir, f"encoder_epoch66_beta{beta_end}_lambda{reg_start}-{reg_end}_fold{fold+1}.weights.h5")
        decoder_path = os.path.join(save_dir, f"decoder_epoch66_beta{beta_end}_lambda{reg_start}-{reg_end}_fold{fold+1}.weights.h5")
        regressor_path = os.path.join(save_dir, f"regressor_epoch66_beta{beta_end}_lambda{reg_start}-{reg_end}_fold{fold+1}.weights.h5")

        svae.save_weights(model_save_path)
        encoder.save_weights(encoder_path)
        decoder.save_weights(decoder_path)
        regressor.save_weights(regressor_path)

        print(f" finished saving weigths → {model_save_path}")
        print(f" finished saving of each path → {encoder_path}, {decoder_path}, {regressor_path}")
        print(f"\n Fold {fold+1}/5 finished")

        del encoder, decoder, regressor, svae, latent_logger
        gc.collect()
        tf.keras.backend.clear_session()


# In[ ]:


# Save z_matrix

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# -------------------------------
# 1. parameters and path
# -------------------------------
latent_dim = 64
input_shape = (91, 109, 91, 1)
# encoder_weights_path = "/mnt/e/T1_2024/grid_results_npz/encoder_epoch66_beta1.0_lambda0.1-1.0_fold1.weights.h5"
# save_path = "/mnt/e/T1_2024/grid_results_npz/z_matrix_fold1_epoch66.npz"
encoder_weights_path = f"{GRID_RESULTS}/encoder_epoch66_beta1.0_lambda0.1-1.0_fold1.weights.h5"
save_path = f"{GRID_RESULTS}/z_matrix_fold1_epoch66.npz"

# -------------------------------
# 2. construct the encoder model and read the weights
# -------------------------------
encoder = build_encoder(input_shape, latent_dim)
encoder.load_weights(encoder_weights_path)

# -------------------------------
# 3. nitialization of z_matrix
# -------------------------------
# data: shape (483, 91, 109, 91)
z_matrix = np.zeros((len(data), latent_dim), dtype=np.float32)

for i in tqdm(range(len(data)), desc="Computing z0–z63"):
    input_image = data[i][np.newaxis, ..., np.newaxis]  # shape: (1, D, H, W, 1)
    z_mean = encoder(input_image)[0]  # encoderの出力が [z_mean, z_logvar] の場合
    z_matrix[i] = z_mean.numpy()[0]

# -------------------------------
# 4. save (npz）
# -------------------------------
np.savez_compressed(save_path, z_matrix=z_matrix)
print("Saved z_matrix:", z_matrix.shape)


# In[ ]:




