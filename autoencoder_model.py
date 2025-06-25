#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Genomic Autoencoder Model v6.1
Fully compatible version with all required function names
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense, Reshape,
    BatchNormalization, Dropout, LayerNormalization, MultiHeadAttention,
    SpatialDropout1D, GaussianNoise, Activation, Lambda, Add
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import MaxNorm
import logging
import time
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _adjust_length(x, target_length):
    """Lambda layer for sequence length adjustment."""
    current_length = tf.shape(x)[1]
    pad_amount = target_length - current_length
    return tf.cond(
        pad_amount > 0,
        lambda: tf.pad(x, [[0, 0], [0, pad_amount], [0, 0]]),
        lambda: x[:, :target_length, :]
    )

def create_autoencoder(input_shape: Tuple[int, int], 
                      compression_factor: int = 18,
                      dropout_rate: float = 0.5,
                      learning_rate: float = 8e-5,
                      l1_reg: float = 1e-6,
                      l2_reg: float = 3e-4) -> Tuple[Model, Model, Model]:
    """Create the autoencoder model with specified compression factor."""
    sequence_length, n_bases = input_shape
    latent_dim = (sequence_length * n_bases) // compression_factor
    
    # Encoder
    input_seq = Input(shape=input_shape, name='encoder_input')
    x = GaussianNoise(0.15)(input_seq)
    
    x = Conv1D(96, 9, activation='relu', padding='same',
              kernel_regularizer=l1_l2(l1_reg, l2_reg),
              kernel_constraint=MaxNorm(4))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(dropout_rate)(x)
    x = MaxPooling1D(4)(x)
    
    x = Conv1D(192, 7, activation='relu', padding='same',
              kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(dropout_rate)(x)
    x = MaxPooling1D(3)(x)
    
    x = LayerNormalization(epsilon=1e-6)(x)
    attn = MultiHeadAttention(
        num_heads=8, 
        key_dim=192//8,
        dropout=0.4,
        kernel_regularizer=l1_l2(l1_reg, l2_reg)
    )(x, x)
    x = Add()([x, attn])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    x = Conv1D(384, 5, activation='relu', padding='same',
              kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    encoded = Dense(latent_dim, activation='relu',
                   kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    
    encoder = Model(input_seq, encoded, name='encoder')

    # Decoder
    latent_input = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense((sequence_length//12) * 384, activation='relu')(latent_input)
    x = Reshape(((sequence_length//12), 384))(x)
    
    x = Conv1D(192, 7, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(3)(x)
    
    x = Conv1D(96, 9, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(4)(x)
    
    x = Lambda(_adjust_length, arguments={'target_length': sequence_length})(x)
    decoded = Conv1D(n_bases, 7, activation='softmax', padding='same')(x)
    decoder = Model(latent_input, decoded, name='decoder')

    # Autoencoder
    autoencoder = Model(input_seq, decoder(encoder(input_seq)), name='autoencoder')
    
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-7,
        clipnorm=1.0
    )
    
    autoencoder.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Created autoencoder with latent dim {latent_dim}")
    return autoencoder, encoder, decoder

def train_autoencoder(autoencoder: Model, 
                     train_data: np.ndarray, 
                     val_data: np.ndarray, 
                     batch_size: int = 64,
                     epochs: int = 50, 
                     callbacks: Optional[list] = None) -> tf.keras.callbacks.History:
    """Train the autoencoder model."""
    if callbacks is None:
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=4,
                min_delta=0.005,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                save_format='keras'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.25,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

    # Data augmentation - reverse complement
    aug_train_data = np.array([x[::-1, ::-1] if np.random.rand() > 0.5 else x 
                             for x in train_data])
    
    history = autoencoder.fit(
        aug_train_data, train_data,
        validation_data=(val_data, val_data),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    best_epoch = np.argmax(history.history['val_accuracy'])
    logger.info(f"Best validation accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
    return history

def save_models(autoencoder: Model, 
               encoder: Model, 
               decoder: Model, 
               save_dir: str) -> None:
    """Save all model components."""
    os.makedirs(save_dir, exist_ok=True)
    autoencoder.save(os.path.join(save_dir, "autoencoder.keras"))
    encoder.save(os.path.join(save_dir, "encoder.keras"))
    decoder.save(os.path.join(save_dir, "decoder.keras"))
    logger.info(f"Models saved to {save_dir}")

def load_trained_models(save_dir: str) -> Tuple[Model, Model, Model]:
    """Load trained model components."""
    custom_objects = {
        'MultiHeadAttention': MultiHeadAttention,
        '_adjust_length': _adjust_length
    }
    autoencoder = load_model(
        os.path.join(save_dir, "autoencoder.keras"),
        custom_objects=custom_objects
    )
    encoder = load_model(
        os.path.join(save_dir, "encoder.keras"),
        custom_objects=custom_objects
    )
    decoder = load_model(
        os.path.join(save_dir, "decoder.keras"),
        custom_objects=custom_objects
    )
    logger.info(f"Models loaded from {save_dir}")
    return autoencoder, encoder, decoder

def compress_sequences(encoder: Model, 
                     sequences: np.ndarray, 
                     batch_size: int = 256) -> np.ndarray:
    """Compress input sequences."""
    logger.info(f"Compressing {sequences.shape[0]} sequences")
    compressed = encoder.predict(sequences, batch_size=batch_size, verbose=1)
    actual_ratio = sequences.nbytes / compressed.nbytes
    logger.info(f"Actual compression ratio: {actual_ratio:.1f}x")
    return compressed

def reconstruct_sequences(decoder: Model, 
                        compressed_sequences: np.ndarray, 
                        batch_size: int = 256) -> np.ndarray:
    """Reconstruct sequences from compressed form."""
    logger.info(f"Reconstructing {compressed_sequences.shape[0]} sequences")
    return decoder.predict(compressed_sequences, batch_size=batch_size, verbose=1)