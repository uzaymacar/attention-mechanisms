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
import os
import unicodedata
import re
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import get_file, to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, Bidirectional, Dense, RepeatVector, \
    TimeDistributed
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
# 0: BiLSTM Model
# 1: BiLSTM Model w/ Global Attention
# 2: BiLSTM Model w/ Local-m Attention
# 3: BiLSTM Model w/ Local-p Attention
# 4: BiLSTM Model w/ Local-p* Attention
args = parser.parse_args()

# Set seeds for reproducibility
np.random.seed(500)
tf.random.set_seed(500)

# Set global constants
embedding_dims = 256    # number of dimensions to represent each character in vector space
batch_size = 100        # feed in the neural network in 100-example training batches
num_epochs = 10         # number of times the neural network goes over EACH training example
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
X_train, X_test, Y_train, Y_test = train_test_split(
    input_tensor, target_tensor, test_size=0.2, random_state=500
)

# Compute batch size and cutoff training & validation examples to fit
training_cutoff, test_cutoff = len(X_train) % batch_size, len(X_test) % batch_size
X_train, Y_train = X_train[:-training_cutoff], Y_train[:-training_cutoff]
X_test, Y_test = X_test[:-test_cutoff], Y_test[:-test_cutoff]

# Create word-level machine translation model -> Encoder Model + Decoder Model
model = Sequential()
# Input Layer
model.add(Input(shape=(input_sequence_length,), batch_size=batch_size))
# i) Encoder Model
# Word-Embedding Layer
model.add(Embedding(input_dim=input_vocabulary_size, output_dim=embedding_dims))
# Recurrent Layer
model.add(Bidirectional(CuDNNLSTM(units=512, return_sequences=True if config != 0 else False)))
# Optional Attention Mechanisms
if config == 1:
    model.add(Attention(size=1024, context='many-to-many', alignment_type='global'))
elif config == 2:
    model.add(Attention(size=1024, context='many-to-many', alignment_type='local-m'))
elif config == 3:
    model.add(Attention(size=1024, context='many-to-many', alignment_type='local-p'))
elif config == 4:
    model.add(Attention(size=1024, context='many-to-many', alignment_type='local-p*'))
# Connection between Encoder & Decoder (input_sequence_length -> target_sequence_length)
model.add(RepeatVector(target_sequence_length))
# ii) Decoder Model
# Recurrent Layer
model.add(Bidirectional(CuDNNLSTM(units=512, return_sequences=True)))
# Prediction Layer
model.add(TimeDistributed(Dense(units=target_vocabulary_size, activation='softmax')))

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train multi-step, multi-class classification model
model.fit(x=X_train, y=Y_train,
          validation_data=(X_test, Y_test),
          epochs=num_epochs, batch_size=batch_size)
