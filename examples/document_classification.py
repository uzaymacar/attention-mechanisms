# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 03:43:11 2019

@author: uzaymacar

Script containing an example and guideline for training multi-class document classification models
using the implemented @HierarchicalAttention() layer.

Usage:
  # Train a LSTM model:
  python document_classification.py
  # Train a LSTM model w/ Self-Attention:
  python document_classification.py --config=1
  Check CONFIG OPTIONS comment block for more options
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, Flatten, Dropout, Dense
from tensorflow.compat.v1.keras.layers import CuDNNLSTM   # CuDNNLSTM not yet released for TF 2.0

import sys
sys.path.append('..')  # add parent directory to Python path for layers.py access
from layers import Attention, SelfAttention

# Argument specification
parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default=0,
                    help="Integer value representing a model configuration")
# CONFIG OPTIONS:
# 0: LSTM Model
# 1: LSTM Model w/ Self-Attention
# 2: LSTM Model w/ Global Attention
# 3: LSTM Model w/ Local-p* Attention
args = parser.parse_args()

# Set seeds for reproducibility
tf.random.set_seed(500)

# Set global constants
vocabulary_size = 10000  # choose 20k most-used words for truncated vocabulary
sequence_length = 1000   # choose 1000-word sequences
embedding_dims = 50      # number of dimensions to represent each word in vector space
batch_size = 32          # feed in the neural network in 100-example training batches
num_epochs = 30          # number of times the neural network goes over EACH training example
config = int(args.config)  # model configuration

# Setup np.load() with allow_pickle=True before loading data as described in:
# https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# Load the Reuters news dataset for document classification
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=vocabulary_size)

# Set more global constants
num_categories = max(Y_train) + 1

# Restore np.load() for future normal usage
np.load = np_load_old

# Pad sequences to maximum found sequence length
X_train = pad_sequences(sequences=X_train, maxlen=sequence_length)
X_test = pad_sequences(sequences=X_test, maxlen=sequence_length)

# Compute batch size and cutoff training & validation examples to fit
training_cutoff, test_cutoff = len(X_train) % batch_size, len(X_test) % batch_size
X_train, Y_train = X_train[:-training_cutoff], Y_train[:-training_cutoff]
X_test, Y_test = X_test[:-test_cutoff], Y_test[:-test_cutoff]

# Create word-level multi-class document classification model
# Input Layer
X = Input(shape=(sequence_length,), batch_size=batch_size)

# Word-Embedding Layer
embedded = Embedding(input_dim=vocabulary_size, output_dim=embedding_dims)(X)

# Recurrent Layers
if config != 0:
    encoder_output, hidden_state, cell_state = CuDNNLSTM(units=128,
                                                         return_sequences=True,
                                                         return_state=True)(embedded)
    attention_input = [encoder_output, hidden_state]
else:
    encoder_output = CuDNNLSTM(units=128)(embedded)

# Optional Attention Mechanisms
if config == 1:
    encoder_output, attention_weights = SelfAttention(size=128,
                                                      num_hops=10,
                                                      use_penalization=False)(encoder_output)
elif config == 2:
    encoder_output, attention_weights = Attention(context='many-to-one',
                                                  alignment_type='global')(attention_input)
    encoder_output = Flatten()(encoder_output)
elif config == 3:
    encoder_output, attention_weights = Attention(context='many-to-one',
                                                  alignment_type='local-p*',
                                                  window_width=100,
                                                  score_function='scaled_dot')(attention_input)
    encoder_output = Flatten()(encoder_output)

# Prediction Layer
Y = Dense(units=num_categories, activation='softmax')(encoder_output)

# Compile model
model = Model(inputs=X, outputs=Y)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train multi-class classification model
model.fit(x=X_train, y=Y_train,
          validation_data=(X_test, Y_test),
          epochs=num_epochs, batch_size=batch_size)
