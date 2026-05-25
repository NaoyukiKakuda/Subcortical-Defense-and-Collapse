#!/usr/bin/env python
# coding: utf-8

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
from datetime import datetime

print("finished preparing")


# In[ ]:


from project_config import BASE_DIR, JACOBIAN_DIR, FIGURE_DIR, AAL_LABEL_TXT, AAL_LABEL_NII


# In[27]:


# Next step is the decorder


# In[28]:


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


# In[29]:


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

# --- Supervised VAE ---
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


# In[30]:


latent_dim = 64 
decoder = build_decoder(latent_dim=latent_dim)
decoder.trainable = False
decoder.build((None, latent_dim))

decoder_path = "/mnt/e/T1_2024/grid_results_npz/decoder_epoch66_beta1.0_lambda0.1-1.0_fold1.weights.h5"
decoder.load_weights(decoder_path)

print("decoder loaded successfully")


# In[31]:


decoder.input_shape
decoder.summary()


# In[ ]:





# In[32]:


# Decode the latent bariables (483, 64) to reconstructed images (483, 91, 109, 91) 
# Calculate the square of first derivative along PHATE axis, smoothed by savgol_filter 

import os
import numpy as np
import gc
from scipy.signal import savgol_filter

def calculate_and_save_jacobian(z_std, z_phate, decoder, save_dir, batch_size=1):
    # 1. sort
    phate1 = z_phate[:, 0]
    sort_idx = np.argsort(phate1)
    sorted_z = z_std[sort_idx]
    sorted_phate1 = phate1[sort_idx]

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "sorted_phate1.npy"), sorted_phate1)

    # 2. dedode (float32)
    print(f"Decoding {len(sorted_z)} points...")
    recon_list = []
    for i in range(0, len(sorted_z), batch_size):
        batch_z = sorted_z[i : i + batch_size]
        batch_recon = decoder.predict(batch_z, batch_size=batch_size, verbose=0)
        recon_list.append(batch_recon.reshape(len(batch_z), -1).astype(np.float32))

    flattened_recon = np.concatenate(recon_list, axis=0)
    del recon_list
    gc.collect()

    # --- 0次 ---
    print("Smoothing & Saving Jacobian 0 (Activity)...")
    # 0次微分（deriv=0）としてSavGolをかけ、ノイズを飛ばします
    jac0 = savgol_filter(flattened_recon, window_length=31, polyorder=2, deriv=0, axis=0)
    np.save(os.path.join(save_dir, "jacobian0.npy"), jac0)
    del jac0
    gc.collect()

    # --- standazation for differntial (Z-score) ---
    print("Normalizing voxels for derivatives...")
    mean_vals = flattened_recon.mean(axis=0)
    std_vals = flattened_recon.std(axis=0) + 1e-10
    normalized_recon = (flattened_recon - mean_vals) / std_vals

    del flattened_recon
    gc.collect()

    # --- first derivative ---
    print("Calculating & Saving Jacobian 1 (Velocity)...")
    jac1 = savgol_filter(normalized_recon, window_length=21, polyorder=2, deriv=1, axis=0)
    np.save(os.path.join(save_dir, "jacobian1.npy"), jac1)
    del jac1
    gc.collect()

    # --- second derivative ---
    print("Calculating & Saving Jacobian 2 (Acceleration)...")
    jac2 = savgol_filter(normalized_recon, window_length=91, polyorder=3, deriv=2, axis=0)
    np.save(os.path.join(save_dir, "jacobian2.npy"), jac2)
    del jac2

    del normalized_recon
    gc.collect()
    print("✅ All processes completed and saved.")

# save_dir = "/mnt/e/T1_2024/jacobian"
save_dir = JACOBIAN_DIR
calculate_and_save_jacobian(z_std, z_phate, decoder, save_dir)


# In[ ]:





# In[ ]:




