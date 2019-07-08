# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 17:50:32 2019

@author: ongunuzaymacar

Script containing an example and guideline for training machine translation models
using the implemented @Attention() layer. This script is adapted from
https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention

Usage:
  # Train a Encoder-Decoder model:
  python machine_translation.py
  # Train a Encoder-Decoder model with Global Attention:
  python machine_translation.py --config=1
  # Check CONFIG OPTIONS comment block for more options
"""

import argparse
import time
import os
import unicodedata
import re
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import get_file, to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, Dense, RepeatVector, \
    TimeDistributed, Flatten, Lambda, Concatenate, Permute
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.layers import CuDNNLSTM   # CuDNNLSTM not yet released for TF 2.0

import sys
sys.path.append('..')  # add parent directory to Python path for layers.py access
from layers import Attention

# Argument specification
parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default=0,
                    help="Integer value representing a model configuration")
# CONFIG OPTIONS:
# 0: Encoder-Decoder Model
# 1: Encoder-Decoder Model w/ Global Attention
# 2: Encoder-Decoder Model w/ Local-m Attention
# 3: Encoder-Decoder Model w/ Local-p Attention
# 4: Encoder-Decoder Model w/ Local-p* Attention
args = parser.parse_args()

# Set seeds for reproducibility
np.random.seed(500)
tf.random.set_seed(500)

# Set global constants
embedding_dim = 128     # number of dimensions to represent each character in vector space
batch_size = 100        # feed in the neural network in 100-example training batches
num_epochs = 20         # number of times the neural network goes over EACH training example
config = int(args.config)  # model-configuration

# Load Spanish-to-English dataset (.zip)
zipped = get_file(
    fname='spa-eng.zip',
    origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True
)
file = os.path.join(os.path.dirname(zipped), 'spa-eng/spa.txt')


def unicode_to_ascii(string):
    """Function to convert the string from unicode file to ascii format"""
    return ''.join(char for char in unicodedata.normalize('NFD', string)
                   if unicodedata.category(char) != 'Mn')


def preprocess_sentence(sentence):
    """
    Function to preprocess sentences according to machine translation conventions. Includes
    conversion to ascii characters, general cleaning operations, and removal of accents.
    """
    sentence = unicode_to_ascii(sentence.lower().strip())
    # Creates a space between a word and the punctuation following it, ex: "hi dad." => "hi dad ."
    sentence = re.sub(r'([?.!,¿])', r' \1 ', sentence)
    sentence = re.sub(r'[" "]+', ' ', sentence)

    # Replace everything with space except (a-z, A-Z, '.', '?', '!', ',')
    sentence = re.sub(r'[^a-zA-Z?.!,¿]+', ' ', sentence)

    # Remove spaces
    sentence = sentence.rstrip().strip()

    # Add a start and an end token to the sentence for the model to recognize
    sentence = '<start> ' + sentence + ' <end>'
    return sentence


def create_dataset(path, num_examples):
    """Returns sentence pairs in [ENGLISH, SPANISH] format"""
    lines = open(path, encoding='utf8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(sentence)
                   for sentence in line.split('\t')]
                  for line in lines[:num_examples]]
    return zip(*word_pairs)


# Load and process pairwise English and Spanish sentences
english_sentences, spanish_sentences = create_dataset(path=file, num_examples=None)


def max_length(tensor):
    """Function that returns the maximum length of any element in a given tensor"""
    return max(len(tensor_unit) for tensor_unit in tensor)


def tokenize(language):
    """Function to tokenize language by mapping words to integer indices"""
    # Perform tokenization
    language_tokenizer = Tokenizer(filters='')
    language_tokenizer.fit_on_texts(language)
    tensor = language_tokenizer.texts_to_sequences(language)
    # Pad sequences to maximum found sequence length by appending 0s to end
    tensor = pad_sequences(sequences=tensor, padding='post')

    return tensor, language_tokenizer


def load_dataset(path, num_examples=None):
    """Function to load dataset"""
    # Create cleaned input-output pairs
    target_language, input_language = create_dataset(path, num_examples)
    # Create language tokenizers and extract tensors
    input_tensor, input_language_tokenizer = tokenize(input_language)
    target_tensor, target_language_tokenizer = tokenize(target_language)

    return input_tensor, target_tensor, input_language_tokenizer, target_language_tokenizer


# Get example (input) tensors, label (target) tensors, and distinct tokenizers for both languages
input_tensor, target_tensor, input_language_tokenizer, target_language_tokenizer = load_dataset(
    path=file, num_examples=None
)

# Setup more global constants
input_vocabulary_size = len(input_language_tokenizer.word_index) + 1
target_vocabulary_size = len(target_language_tokenizer.word_index) + 1
# Calculate maximum sequence lengths of the input and target tensors
input_sequence_length, target_sequence_length = max_length(input_tensor), max_length(target_tensor)

# Split data to training and validation sets
X_train, X_test, Y_train, Y_test = train_test_split(input_tensor,
                                                    target_tensor,
                                                    test_size=0.2,
                                                    random_state=500)

# Compute batch size and cutoff training & validation examples to fit
training_cutoff, test_cutoff = len(X_train) % batch_size, len(X_test) % batch_size
X_train, Y_train = X_train[:-training_cutoff], Y_train[:-training_cutoff]
X_test, Y_test = X_test[:-test_cutoff], Y_test[:-test_cutoff]

# Feed in current labels (Y) as decoder inputs, and pad current labels by 1 word
# Check https://www.tensorflow.org/images/seq2seq/attention_mechanism.jpg for better understanding
X_train_target, X_test_target = Y_train, Y_test
Y_train = np.array(pad_sequences(sequences=np.array([sequence[1:] for sequence in Y_train]),
                                 maxlen=target_sequence_length,
                                 padding='post'))
Y_test = np.array(pad_sequences(sequences=np.array([sequence[1:] for sequence in Y_test]),
                                maxlen=target_sequence_length,
                                padding='post'))

# Create word-level multi-class classification (machine translation), sequence-to-sequence model
# Input Layers
# i)  Initialize input & target sequences
X_input = Input(shape=(input_sequence_length,), batch_size=batch_size, name='input_sequences')
X_target = Input(shape=(target_sequence_length,), batch_size=batch_size, name='target_sequences')
# ii) Initialize hidden & cell states
initial_hidden_state = Input(shape=(128,), batch_size=batch_size, name='hidden_state')
initial_cell_state = Input(shape=(128,), batch_size=batch_size, name='cell_state')
hidden_state, cell_state = initial_hidden_state, initial_cell_state
# NOTE: Here hidden state refers to the recurrently propagated input to the cell, whereas cell
# state refer to the cell state directly from the previous cell.

# Word-Embedding Layers
# i)  Embed input sequences from the input language
embedded_input = Embedding(input_dim=input_vocabulary_size, output_dim=embedding_dim)(X_input)
# ii) Embed target sequences from the target language
embedded_target = Embedding(input_dim=target_vocabulary_size, output_dim=embedding_dim)(X_target)
# NOTE: The embedded target sequences (deriving from X_target) allow us to enforce Teacher Forcing:
# using the actual output (correct translation) from the training dataset at the current time step
# as input in the next time step, rather than the output generated by the network.

# Recurrent Layers
# i)  Encoder
encoder_output = CuDNNLSTM(units=128, return_sequences=True)(embedded_input)
# ii) Decoder
decoder_recurrent_layer = CuDNNLSTM(units=128, return_state=True)
# NOTE: The encoder is always fully vectorized and returns the hidden representations of the whole
# sequence at once, whereas the decoder does this step by step.

# Optional Attention Mechanism
if config == 1:
    attention_layer = Attention(context='many-to-many', alignment_type='global')
elif config == 2:
    attention_layer = Attention(context='many-to-many', alignment_type='local-m')
elif config == 3:
    attention_layer = Attention(context='many-to-many', alignment_type='local-p')
elif config == 4:
    attention_layer = Attention(context='many-to-many', alignment_type='local-p*')

# Prediction Layer
decoder_dense_layer = Dense(units=target_vocabulary_size, activation='softmax')

# Training Loop
outputs = []
for timestep in range(target_sequence_length):
    # Get current input in from embedded target sequences
    current_word = Lambda(lambda x: x[:, timestep: timestep+1, :])(embedded_target)
    # Apply optional attention mechanism
    if config != 0:
        context_vector, attention_weights = attention_layer([encoder_output, hidden_state, timestep])
    # Combine information
    decoder_input = Concatenate(axis=1)(
        [context_vector if config != 0 else encoder_output, current_word]
    )
    # Decode target word hidden representation at t = timestep
    output, hidden_state, cell_state = decoder_recurrent_layer(
        decoder_input, initial_state=[hidden_state, cell_state]
    )
    # Predict next word & append to outputs
    decoder_outputs = decoder_dense_layer(output)
    outputs.append(decoder_outputs)

# Reshape outputs to (B, S', V)
outputs = Lambda(lambda x: tf.keras.backend.permute_dimensions(tf.stack(x), pattern=(1, 0, 2)))(outputs)

# Compile model
model = Model(inputs=[X_input, X_target, initial_hidden_state, initial_cell_state],
              outputs=outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# print(model.summary())
# NOTE: Summary is omitted due to length, deriving from number of simulated recurrent connections

# Create placeholder variables of 0s
placeholder = tf.zeros(shape=(len(X_train), 128))

# Train multi-step, multi-class classification model
model.fit(x={'input_sequences': X_train, 'target_sequences': X_train_target,
             'hidden_state': placeholder, 'cell_state': placeholder},
          y=Y_train,
          validation_data=([X_test, X_test_target,
                            placeholder, placeholder],
                           Y_test),
          epochs=num_epochs, batch_size=batch_size)
