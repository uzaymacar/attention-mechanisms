# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 17:47:47 2019

@author: ongunuzaymacar

Script containing an example and guideline for training multi-class language models used for text
generation, using the implemented @Attention and @SelfAttention() layers.

Usage:
  # Train a LSTM model:
  python text_generation.py
  # Train LSTM model with non-penalized self-attention:
  python text_generation.py --config=1
  # Check CONFIG OPTIONS comment block for more options
"""

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import get_file, to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Lambda, Input, Embedding, Bidirectional, Flatten, Dense
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
# 1: LSTM Model w/ Self-Attention (Non-Penalized)
# 2: LSTM Model w/ Global Attention
# 3: LSTM Model w/ Local-p* Attention
args = parser.parse_args()

# Set seeds for reproducibility
np.random.seed(500)
tf.random.set_seed(500)

# Set global constants
sequence_length = 120   # choose 120-character sequences
embedding_dims = 50     # number of dimensions to represent each word in vector space
batch_size = 100        # feed in the neural network in 100-example training batches
num_epochs = 15         # number of times the neural network goes over EACH training example
config = int(args.config)  # model-configuration

# Load Shakespeare corpus file (.txt)
file = get_file(
    fname='shakespeare.txt',
    origin='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)

# Extract data (string format) from corpus file
corpus = open(file=file, mode='rb').read().decode(encoding='utf-8')
# Optionally trim data for memory storage and performance purposes
# corpus = corpus[:len(corpus) // 2]

# Set corpus-specific attributes for language modelling
# corpus = corpus.split()  # uncomment to convert to word-level
vocabulary = sorted(set(corpus))  # word-level vocabulary
vocabulary_size = len(vocabulary)
char_to_index = {char: index for index, char in enumerate(vocabulary)}

# Encode each sequence using word to index mapping
sequences = []
for i in range(0, len(corpus) - sequence_length, 1):
    sequences.append([char_to_index[word] for word in corpus[i: i+sequence_length]])

# Set training inputs and labels
m = len(sequences)  # number of training examples
X = np.array(sequences, dtype=np.int)
Y = np.zeros(shape=(m, vocabulary_size), dtype=np.int32)

# Fill labels
for j in range(m):
    Y[j] = to_categorical(y=[char_to_index[corpus[j+sequence_length]]], num_classes=vocabulary_size)

del corpus  # save memory

# Shuffle inputs (X) and labels (Y) set in unison
assert len(X) == len(Y)
random_order = np.random.permutation(m)
X, Y = X[random_order], Y[random_order]

# Split data to training and validation sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=500)

# Compute batch size and cutoff training & validation examples to fit
training_cutoff, test_cutoff = len(X_train) % batch_size, len(X_test) % batch_size
X_train, Y_train = X_train[:-training_cutoff], Y_train[:-training_cutoff]
X_test, Y_test = X_test[:-test_cutoff], Y_test[:-test_cutoff]


# Define custom training utilities that are widely used for language modelling
def loss(y_true, y_pred):
    """Calculates categorical crossentropy as loss"""
    return categorical_crossentropy(y_true=y_true, y_pred=y_pred)


def perplexity(labels, logits):
    """Calculates perplexity metric = 2^(entropy) or e^(entropy)"""
    return pow(2, loss(y_true=labels, y_pred=logits))


# Create multi-class, char-level text generation model
# Input Layer
X = Input(shape=(sequence_length,), batch_size=batch_size)

# Word-Embedding Layer
embedded = Embedding(input_dim=vocabulary_size, output_dim=embedding_dims)(X)

# Recurrent Layer
if config != 0:
    encoder_output, hidden_state, cell_state = CuDNNLSTM(units=512,
                                                         return_sequences=True,
                                                         return_state=True)(embedded)
    attention_input = [encoder_output, hidden_state]
else:
    encoder_output = CuDNNLSTM(units=512)(embedded)

# Optional Attention Mechanisms
if config == 1:
    encoder_output, attention_weights = SelfAttention(size=50,
                                                      num_hops=16,
                                                      use_penalization=False)(encoder_output)
elif config == 2:
    encoder_output, attention_weights = Attention(context='many-to-one',
                                                  alignment_type='global')(attention_input)
    encoder_output = Flatten()(encoder_output)
elif config == 3:
    encoder_output, attention_weights = Attention(context='many-to-one',
                                                  alignment_type='local-p*',
                                                  window_width=25)(attention_input)
    encoder_output = Flatten()(encoder_output)

# Prediction Layer
Y = Dense(units=vocabulary_size, activation='softmax')(encoder_output)

# Compile model
model = Model(inputs=X, outputs=Y)
model.compile(loss=loss, optimizer='adam', metrics=[perplexity, categorical_accuracy])
print(model.summary())

# Train multi-class classification model
model.fit(x=X_train, y=Y_train,
          validation_data=(X_test, Y_test),
          epochs=num_epochs, batch_size=batch_size)
