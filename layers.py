# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 20:08:11 2019

@author: ongunuzaymacar

Script containing custom layer implementations for a family of attention mechanisms in 
TensorFlow with Keras integration, specifically for the upcoming new version 2.0.0. It
should be noted that every layer is only tested with applications in many-to-one sequence
generation, but should theoretically be adaptable to other domains with minor tweaks.
Every layer is a subclass of tf.keras.layers.Layer(). The __init__() method of each custom
class calls the the initialization method of its parent and defines additional attributes
specific to each layer. The get_config() method calls the configuration method of its
parent and defines custom attributes introduced with the layer. If a custom layer includes
method build(), then it contains trainable parameters. Take the Attention() layer for
example, the backpropagation of the loss signals which inputs to give more care to and
hence indicates a change in weights of the layer. On the other hand, some layers are
definite and should not experience any change. Take the OneHotEncoding() layer for
example, there exists only one way to transform a vector into one-hot representation.
Finally, the call() method is the actual operation that is performed on the input tensors.
compute_output_shape() methods are avoided for spacing, and instead comments next to each
operation in each layer indicate the output shapes. For ease of notation, the following
abbreviations are used:
i)   B = batch size,      ii) S = sequence length, 
iii) V = vocabulary size, iv) H = number of hidden dimensions
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Activation, Permute
from tensorflow.keras.layers import Multiply, Lambda, Reshape, Dot, Concatenate, RepeatVector

class Attention(Layer):
    """
    Layer for implementing two common types of attention mechanisms:
    i) global (soft) attention, as discussed in "Neural Machine Translation by
    Jointly Learning to Align and Translate" by Dzmitry Bahdanau et. al. This type of
    attention attends to the entire input state space. Aims to eliminate compression
    and loss of information in encoder RNNs.
    ii) local (hard) attention mechanism, as discussed in "Effective Approaches to
    Attention-based Neural Machine Translation" by Minh-Thang Luong et. al. Aims to
    eliminate the attentive cost of global attention by instead focusing on a small subset
    of tokens in the input sequence. This window is proposed as [p_t-D, p_t+D] where
    D=width, and we disregard positions that cross sequence boundaries. The aligned
    position, p_t, is decided either through a) monotonic alignment: set p_t=t, or
    ii) predictive alignment: set p_t = S*sigmoid(FC1(tanh(FC2(h_t))) where
    fully-connected layers are trainable weight matrices. Since yielding an integer index
    value is undifferentiable due to tf.cast() and other methods, this implementation
    instead derives a aligned position float value and uses Gaussian distribution to
    adjust the attention weights of all source hidden states instead of slicing the actual
    window. We also propose an experimental alignment type, iii) completely predictive
    alignment: set p_t as in ii), but apply it to all source hidden states (h_s) instead
    of the target hidden state (h_t). Then, choose @window_width positions to build
    the context vector and zero out the rest.

    The setting use_bias=False converts the Dense() layers into annotation weight matrices.
    Softmax activation ensures that all weights sum up to 1.
    Read more here to make more sense of the code and implementations:
    i)   https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention
    ii)  https://github.com/philipperemy/keras-attention-mechanism/issues/14
    iii) https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html

    SUGGESTION: If model doesn't converge, increase either the hidden size of the RNN
    model, the batch size of the model, or the param @size. If test accuracy is low,
    decrease instead.

    NOTE: This implementation takes the hidden states associted with the last timestep as
    the target hidden state (h_t) as suggested by @felixhao28 in i), whereas originally
    attention was proposed for MANY-TO-MANY sequence tasks like machine translation.
    Hence, when trying to predict what word (token) comes after sequence ['I', 'love',
    'biscuits', 'and'], we take h('and') with shape (1, H) as the target hidden state.

    @param size (int): size of attension vector or attention length; number of hidden
           units to decode the attention to with dense layer, presumably before being fed
           to the final softmax dense layer for next token prediction
    @param alignment_type (str): type of attention mechanism to be applied, 'local-m'
           corresponds to monotonic alignment where we take the last @window_width
           timesteps, 'local-p' corresponds to having a Gaussian distribution around
           the predicted aligned position, whereas 'local-p*' corresponds to the newly
           proposed method to adaptively learning the unique timesteps to give attention
    @param window_width (int): width for set of source hidden states in 'local' attention
    @param score_function (str): alignment score function config; current implementations
           include the 'dot', 'general', and 'location' both by Luong et. al. 2015,
           'concat' by Bahdanau et. al. 2015, and 'scaled_dot' by Vaswani et. al. 2017
    """
    def __init__(self, size, alignment_type='global', window_width=None,
                 score_function='general', **kwargs):
        if alignment_type not in ['global', 'local-m', 'local-p', 'local-p*']:
            raise ValueError("Argument for param @alignment_type is not recognized")
        if alignment_type == 'global':
            if window_width is not None:
                raise ValueError("Can't use windowed approach with global attention")
        if score_function not in ['dot', 'general', 'location', 'concat', 'scaled_dot']:
            raise ValueError("Argument for param @score_function is not recognized")
        super(Attention, self).__init__(**kwargs)
        self.size = size
        self.alignment_type = alignment_type
        self.window_width = window_width # 2*D
        self.score_function = score_function

    def get_config(self):
        base_config = super(Attention, self).get_config()
        base_config['size'] = self.size
        base_config['alignment_type'] = self.alignment_type
        base_config['window_width'] = self.window_width
        base_config['score_function'] = self.score_function
        return base_config

    def build(self, input_shape): # Build weight matrices for trainable, adaptive parameters
        if 'local-p' in self.alignment_type:
            self.W_p = Dense(units=input_shape[2], use_bias=False)
            self.W_p.build(input_shape=(None, None, input_shape[2])) # (B, 1, H)
            self._trainable_weights += self.W_p.trainable_weights

            self.v_p = Dense(units=1, use_bias=False)
            self.v_p.build(input_shape=(None, None, input_shape[2])) # (B, 1, H)
            self._trainable_weights += self.v_p.trainable_weights

        if 'dot' not in self.score_function: # weight matrix not utilized for 'dot' function
            self.W_a = Dense(units=input_shape[2], use_bias=False)
            self.W_a.build(input_shape=(None, None, input_shape[2])) # (B, S*, H)
            self._trainable_weights += self.W_a.trainable_weights

        if self.score_function == 'concat': # define additional weight matrices
            self.U_a = Dense(units=input_shape[2], use_bias=False)
            self.U_a.build(input_shape=(None, None, input_shape[2])) # (B, 1, H)
            self._trainable_weights += self.U_a.trainable_weights

            self.v_a = Dense(units=1, use_bias=False)
            self.v_a.build(input_shape=(None, None, input_shape[2])) # (B, S*, H)
            self._trainable_weights += self.v_a.trainable_weights

        self.attention_vector = Dense(units=self.size, activation='tanh', use_bias=False)
        self.attention_vector.build(input_shape=(None, 2*input_shape[2])) # (B, 2*H)
        self._trainable_weights += self.attention_vector.trainable_weights

        super(Attention, self).build(input_shape)

    def call(self, inputs):
        sequence_length = inputs.shape[1]
        ## Get h_t, the current (target) hidden state ##
        target_hidden_state = Lambda(function=lambda x: x[:, -1, :])(inputs) # (B, H)
        target_hidden_state_reshaped = Reshape(target_shape=(1, inputs.shape[2]))(target_hidden_state) # (B, 1, H)

        ## Get h_s, source hidden states through specified attention mechanism ##
        if self.alignment_type == 'global': ## Global Approach ##
            source_hidden_states = inputs # (B, S*=S, H)

        elif 'local' in self.alignment_type: ## Local Approach ##
            if self.window_width == None: ## Automatically set window width ##
                self.window_width = sequence_length // 2

            if self.alignment_type == 'local-m': ## Monotonic Alignment ##
                aligned_position = sequence_length
                left_border = aligned_position - self.window_width if aligned_position - self.window_width >= 0 else 0
                source_hidden_states = Lambda(function=lambda x: x[:, left_border:, :])(inputs) # (B, S*=D, H)

            elif self.alignment_type == 'local-p': ## Predictive Alignment ##
                aligned_position = self.W_p(target_hidden_state) # (B, H)
                aligned_position = Activation('tanh')(aligned_position) # (B, H)
                aligned_position = self.v_p(aligned_position) # (B, 1)
                aligned_position = Activation('sigmoid')(aligned_position) # (B, 1)
                aligned_position = aligned_position * sequence_length # (B, 1)
                source_hidden_states = inputs # (B, S, H)

            elif self.alignment_type == 'local-p*': ## Completely Predictive Alignment ##
                aligned_position = self.W_p(inputs) # (B, S, H)
                aligned_position = Activation('tanh')(aligned_position) # (B, S, H)
                aligned_position = self.v_p(aligned_position) # (B, S, 1)
                aligned_position = Activation('sigmoid')(aligned_position) # (B, S, 1)
                ## Only keep top D values out of the sigmoid activation, and zero-out the rest ##
                aligned_position = tf.squeeze(aligned_position) # (B, S)
                top_probabilities = tf.nn.top_k(input=aligned_position, 
                                                k=self.window_width, 
                                                sorted=False) # (values:(B, D), indices:(B, D))
                onehot_vector = tf.one_hot(indices=top_probabilities.indices, 
                                           depth=sequence_length) # (B, D, S)
                onehot_vector = tf.reduce_sum(onehot_vector, axis=1) # (B, S)
                aligned_position = Multiply()([aligned_position, onehot_vector]) # (B, S)
                aligned_position = Reshape(target_shape=(sequence_length, 1))(aligned_position) # (B, S, 1)
                source_hidden_states = Multiply()([inputs, aligned_position]) # (B, S*=S(D), H)

        ## Compute alignment score through specified function ##
        if 'dot' in self.score_function:
            attention_score = Dot(axes=[2,1])([source_hidden_states, target_hidden_state]) # (B, S*)
            if self.score_function == 'scaled_dot':
                attention_score = attention_score * (1 / np.sqrt(float(inputs.shape[2]))) # (B, S*)

        elif self.score_function == 'general':
            weighted_hidden_states = self.W_a(source_hidden_states) # (B, S*, H)
            attention_score = Dot(axes=[2, 1])([weighted_hidden_states, target_hidden_state]) # (B, S*)

        elif self.score_function == 'location':
            weighted_target_state = self.W_a(target_hidden_state) # (B, H)
            attention_score = Activation('softmax')(weighted_target_state) # (B, H)
            attention_score = RepeatVector(n=inputs.shape[1]-1 if self.seperate
                                           else inputs.shape[1])(attention_score) # (B, S*, H)
            attention_score = tf.reduce_sum(attention_score, axis=-1) # (B, S*)

        elif self.score_function == 'concat':
            weighted_hidden_states = self.W_a(source_hidden_states) # (B, S*, H)
            weighted_target_state = self.U_a(target_hidden_state_reshaped) # (B, 1, H)
            weighted_sum = weighted_hidden_states + weighted_target_state # (B, S*, H)
            weighted_sum = Activation('tanh')(weighted_sum) # (B, S*, H)
            attention_score = self.v_a(weighted_sum) # (B, S*, 1)
            attention_score = attention_score[:, :, 0] # (B, S*)

        attention_weights = Activation('softmax')(attention_score) # (B, S*)
        if self.alignment_type == 'local-p': ## Gaussian Distribution ##
            gaussian_estimation = lambda s: tf.exp(-tf.square(s - aligned_position) /
                                                   (2 * tf.square(self.window_width / 2)))
            gaussian_factor = gaussian_estimation(0)
            for i in range(1, sequence_length):
                gaussian_factor = Concatenate()([gaussian_factor, gaussian_estimation(i)])
            # gaussian_factor: (B, S*)
            attention_weights = attention_weights * gaussian_factor # (B, S*)

        context_vector = Dot(axes=[1,1])([source_hidden_states, attention_weights]) # (B, H)
        combined_information = Concatenate()([context_vector, target_hidden_state]) # (B, 2*H)
        attention_vector = self.attention_vector(combined_information) # (B, self.size)
        return attention_vector

class SelfAttention(Layer):
    """
    Layer for implementing self-attention mechanism, first introduced in "Long Short-Term
    Memory-Networks for Machine Reading" by Jianpeng Cheng et. al. This type of attention
    relates different positions of the same input sequence. Aim is same as Attention()
    layers. This particular implementation follows "A Structured Self-Attentive Sentence
    Embedding" by Zhouhan Lin et. al. Weight variables were preferred over Dense() layers
    in implementation because they allow easy identification of shapes. Softmax activation
    ensures that all weights sum up to 1. The paper suggests adding an extra loss
    parameter to the model to prevent the redundancy problems of the embedding matrix if
    the attention mechanism always provides similar annotation weights. The paper also
    argues that there exists multiple components in a sentence that form the overall
    semantics, and hence propose multiple hops of attention extracted from the same input
    sequence.

    @param size (int): a.k.a attension length, number of hidden units to decode the
           attention before the softmax activation and becoming annotation weights
    @param num_hops (int): number of hops of attention, or number of distinct components
           to be extracted from each sentence.
    @param use_penalization (bool): set True to use penalization, otherwise set False
    @param penalty_coefficient (int): the weight of the extra loss
    """
    def __init__(self, size, num_hops=8, use_penalization=True, penalty_coefficient=0.1, **kwargs):
        self.size = size
        self.num_hops = num_hops
        self.use_penalization = use_penalization
        self.penalty_coefficient = penalty_coefficient
        super(SelfAttention, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        base_config['size'] = self.size
        base_config['num_hops'] = self.num_hops
        base_config['use_penalization'] = self.use_penalization
        base_config['penalty_coefficient'] = self.penalty_coefficient
        return base_config

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.size, input_shape[2]), # (self.size, H)
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(self.num_hops, self.size), # (self.num_hops, self.size)
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs): # (B, S, H)
        # Expand weights to include batch size through implicit broadcasting
        W1, W2 = self.W1[None, :, :], self.W2[None, :, :]
        hidden_states_transposed = Permute(dims=(2,1))(inputs) # (B, H, S)
        attention_score = tf.matmul(W1, hidden_states_transposed) # (B, self.size, S)
        attention_score = Activation('tanh')(attention_score) # (B, self.size, S)
        attention_weights = tf.matmul(W2, attention_score) # (B, self.num_hops, S)
        attention_weights = Activation('softmax')(attention_weights) # (B, self.num_hops, S)
        embedding_matrix = tf.matmul(attention_weights, inputs) # (B, self.num_hops, H)
        embedding_matrix_flattened = Flatten()(embedding_matrix) # (B, self.num_hops*H)

        if self.use_penalization:
            attention_weights_transposed = Permute(dims=(2,1))(attention_weights) # (B, S, self.num_hops)
            product = tf.matmul(attention_weights, attention_weights_transposed) # (B, self.num_hops, self.num_hops)
            identity = tf.eye(self.num_hops, batch_shape=(inputs.shape[0],)) # (B, self.num_hops, self.num_hops)
            frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(product - identity))) # distance
            self.add_loss(self.penalty_coefficient * frobenius_norm) # loss

        return embedding_matrix_flattened