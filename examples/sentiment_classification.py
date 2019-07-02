import sys
sys.path.append('..')  # add parent directory to Python path for layers.py access

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
from layers import SelfAttention

# Set seeds for reproducibility
tf.random.set_seed(500)

# Set global constants
vocabulary_size = 10000  # choose 10k most-used words for truncated vocabulary
sequence_length = 500    # choose 500-word sequences, either pad or truncate sequences to this
embedding_dims = 50      # number of dimensions to represent each word in vector space
batch_size = 100         # feed in the neural network in 100-example training batches
num_epochs = 10          # number of times the neural network goes over EACH training example

config = 0
# CONFIG OPTIONS:
# 0: Simple Multi-Layer Perceptron Model
# 1: Simple Multi-Layer Perceptron Model w/ Self-Attention (Non-Penalized)
# 2: Simple Multi-Layer Perceptron Model w/ Self-Attention (Penalized)

# Load the IMDB dataset for sentiment classification
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=10000)

# Pad & truncate sequences to fixed sequence length
X_train = pad_sequences(sequences=X_train, maxlen=sequence_length)
X_test = pad_sequences(sequences=X_test, maxlen=sequence_length)

# Create binary sentiment classification model
model = Sequential()
# Input Layer
model.add(Input(shape=(sequence_length,), batch_size=batch_size))
# Word-Embedding Layer
model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dims))
# Optional Self-Attention
if config == 1:
    model.add(SelfAttention(size=500, num_hops=8, use_penalization=False))
elif config == 2:
    model.add(SelfAttention(size=500, num_hops=8, use_penalization=True, penalty_coefficient=0.1))
# Multi-Layer Perceptron
model.add(Flatten())
model.add(Dense(units=250, activation='relu'))
# Prediction Layer
model.add(Dense(units=1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train classification model
model.fit(x=X_train, y=Y_train,
          validation_data=(X_test, Y_test),
          epochs=num_epochs, batch_size=batch_size)
