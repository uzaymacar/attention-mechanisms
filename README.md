# Attention Mechanisms

## Table of Contents  
1. [Introduction](#introduction)
2. [Attention Types](#attention-types)
   * [Self Attention](#self-attention)
   * [Global Attention](#global-attention)
   * [Local Attention](#local-attention)
3. [Alignment Functions](#alignment-functions)
   * [Dot Product](#dot-product)
   * [Scaled Dot Product](#scaled-dot-product)
   * [General](#general)
   * [Concat](#concat)
   * [Location-Based](#location-based)
4. [Implementation Details](#implementation-details)
5. [Examples](#examples)
6. [Resources](#resources)

## Introduction
This repository includes custom layer implementations for a whole family of attention mechanisms, compatible with TensorFlow and Keras integration. Attention mechanisms have transformed the landscape of machine translation, and their utilization in other domains of NLP are increasing by day. In a broader sense, they aim to eliminate compression and loss of information due to fixed-length encoding of hidden states derived from input sequences in RNNs. The layers in this repository are tailored specifically for many-to-one sequence tasks, such as sentiment classification and language modeling.

## Attention Types
![Attention Categories](assets/attention_categories.png)

### Self Attention
First introduced in "Long Short-Term Memory-Networks for Machine Reading" by Jianpeng Cheng et al. The idea is to relate different positions of the same hidden state space derived from the input sequence, based on the argument that multiple components together form the overall semantics of a sequence. This approach brings together these differently positioned information through **multiple hops** attention. This particular implementation follows "A Structured Self-Attentive Sentence Embedding" by Zhouhan Lin et al. where authors propose an additional loss metric for regularization to prevent the redundancy problems of the embedding matrix if the attention mechanism always provides similar annotation weights.

![Self Attention](assets/self_attention.png)

### Global (Soft) Attention
First introduced in "Neural Machine Translation by Jointly Learning to Align and Translate" by Dzmitry Bahdanau et al. The idea is to derive a context vector based on **all** hidden states of the encoder RNN. Hence, it is said that this type of attention *attends* to the entire input state space.

![Global Attention](assets/global_attention.png)

### Local (Hard) Attention
First introduced in "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Kelvin Xu et al. and adapted to NLP in "Effective Approaches to Attention-based Neural Machine Translation" by Minh-Thang Luong et al. The idea is to  eliminate the attentive cost of global attention by instead focusing on a small subset of tokens in hidden states set derived from the input sequence. This window is proposed as ```[p_t-D, p_t+D]``` where ```D=width```, and we disregard positions that cross sequence boundaries. The aligned position, ```p_t```, is decided either through **a) monotonic alignment:** set ```p_t=t```, or **b) predictive alignment**: set ```p_t = S*sigmoid(FC1(tanh(FC2(h_t)))``` where fully-connected layers are trainable weight matrices. Since yielding an integer index value is undifferentiable due to ```tf.cast()``` and similar methods, this implementation instead derives a aligned position float value and uses Gaussian distribution to
adjust the attention weights of all source hidden states instead of slicing the actual
window. We also propose an experimental alignment type, **c) completely predictive
alignment:** set ```p_t``` as in ii), but apply it to all source hidden states (```h_s```) instead
of the target hidden state (```h_t```). Then, choose top ```@window_width``` positions to build
the context vector and zero out the rest.

![Local Attention](assets/local_attention.png)

## Alignment Functions
![Alignment Functions](assets/alignment_functions.png)
Each function is trying to compute an alignment score given a target hidden state (```h_t```) and source hidden states (```h_s```).

### Dot Product
![Dot Product](https://latex.codecogs.com/png.latex?\Large&space;score(h_t,&space;h_s)=h_t^\intercal&space;\cdot&space;h_s)

### Scaled Dot Product
![Scaled Dot Product](https://latex.codecogs.com/png.latex?\Large&space;score(h_t,&space;h_s)=\frac{h_t^\intercal&space;\cdot&space;h_s}{\sqrt{H}})

where ```H``` is the number of hidden states given by the encoder RNN.

### General
![General](https://latex.codecogs.com/png.latex?\Large&space;score(h_t,&space;h_s)=h_t^\intercal&space;\cdot&space;W_a&space;\cdot&space;h_s)

where ```W_a``` is a trainable weight matrix.

### Concat
![Concat](https://latex.codecogs.com/png.latex?\Large&space;score(h_t,&space;h_s)=v_a^\intercal&space;\cdot&space;\tanh(W_a[h_t:h_s]))

where ```v_a``` and ```W_a``` are trainable weight matrices.

### Location-Based
![Location Based](https://latex.codecogs.com/png.latex?\Large&space;score(h_t,&space;h_s)=W_a&space;\cdot&space;h_t)


where ```W_a``` is a trainable weight matrix.

## Implementation Details
* It should be noted that every layer is only tested with applications in many-to-one sequence
generation, but should theoretically be adaptable to other domains with minor tweaks.
* Every layer is a subclass of ```tf.keras.layers.Layer()```.
* The ```__init__()``` method of each custom class calls the the initialization method of its parent and defines additional attributes specific to each layer.
* The ```get_config()``` method calls the configuration method of its parent and defines custom attributes introduced with the layer.
* If a custom layer includes method ```build()```, then it contains trainable parameters. Take the ```Attention()``` layer for example, the backpropagation of the loss signals which inputs to give more care to and hence indicates a change in weights of the layer.
* The ```call()``` method is the actual operation that is performed on the input tensors.
* ```compute_output_shape()``` methods are avoided for spacing.

## Examples
These layers can be plugged-in to your projects (whether language models or other types of RNNs) within seconds, just like any other TensorFlow layer with Keras integration. See the example below:
```
X = Input(shape=(sequence_length,), batch_size=batch_size) # define input layer for summary
## Token Embedding (Pretrained or Not) ##
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedded_dimensions)(X)
## Encoding Recurrent Layers ##
encoder = LSTM(units=recurrent_units, return_sequences=True)(embedding) # keep timesteps
## Decoding with Attention ##
decoder = Attention(size=attention_size, alignment_type=attention_type)(encoder)
## Prediction Layer ##
Y = Dense(units=vocabulary_size, activation='softmax')(decoder)
```
where ```alignment_type``` is one of ```'global'```, ```'local-m'```, ```'local-p'```, and ```'local-p*'```. For self attention, call the ```SelfAttention(size=attention_size)``` layer instead.

## Resources
* All papers mentioned above.
* https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention
* https://github.com/philipperemy/keras-attention-mechanism/issues/14
* https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
