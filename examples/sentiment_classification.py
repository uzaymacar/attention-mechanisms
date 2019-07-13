# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 11:43:05 2019

@author: ongunuzaymacar

Script containing an example and guideline for training binary sentiment classification models
using the implemented @SelfAttention() layer.

Usage:
  # Train a simple multi-layer perceptron model:
  python sentiment_classification.py
  # Train simple multi-layer perceptron model with non-penalized self-attention:
  python sentiment_classification.py --config=1
  # Check CONFIG OPTIONS comment block for more options
"""

import argparse
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense

import sys
sys.path.append('..')  # add parent directory to Python path for layers.py access
from layers import SelfAttention

# Argument specification
parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default=0,
                    help="Integer value representing a model configuration")
# CONFIG OPTIONS:
# 0: Simple Multi-Layer Perceptron Model
# 1: Simple Multi-Layer Perceptron Model w/ Self-Attention (Non-Penalized)
# 2: Simple Multi-Layer Perceptron Model w/ Self-Attention (Penalized)
args = parser.parse_args()

# Set seeds for reproducibility
tf.random.set_seed(500)

# Set global constants
vocabulary_size = 10000  # choose 10k most-used words for truncated vocabulary
sequence_length = 500    # choose 500-word sequences, either pad or truncate sequences to this
embedding_dims = 50      # number of dimensions to represent each word in vector space
batch_size = 100         # feed in the neural network in 100-example training batches
num_epochs = 10          # number of times the neural network goes over EACH training example
config = int(args.config)  # model configuration

# Load the IMDB dataset for sentiment classification
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=vocabulary_size)

# Pad & truncate sequences to fixed sequence length
X_train = pad_sequences(sequences=X_train, maxlen=sequence_length)
X_test = pad_sequences(sequences=X_test, maxlen=sequence_length)

# Create word-level binary sentiment classification model
# Input Layer
X = Input(shape=(sequence_length,), batch_size=batch_size)

# Word-Embedding Layer
embedded = Embedding(input_dim=vocabulary_size, output_dim=embedding_dims)(X)

# Optional Self-Attention Mechanisms
if config == 1:
    embedded, attention_weights = SelfAttention(size=50,
                                                num_hops=6,
                                                use_penalization=False)(embedded)
elif config == 2:
    embedded, attention_weights = SelfAttention(size=50,
                                                num_hops=6,
                                                use_penalization=True,
                                                penalty_coefficient=0.1)(embedded)

# Multi-Layer Perceptron
embedded_flattened = Flatten()(embedded)
fully_connected = Dense(units=250, activation='relu')(embedded_flattened)

# Prediction Layer
Y = Dense(units=1, activation='sigmoid')(fully_connected)

# Compile model
model = Model(inputs=X, outputs=Y)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train binary-classification model
model.fit(x=X_train, y=Y_train,
          validation_data=(X_test, Y_test),
          epochs=num_epochs, batch_size=batch_size)
