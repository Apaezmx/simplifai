from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

def activate(activation, tensor):
  """ Maps a string activation to the keras backend function """
  if activation == 'tanh':
    return K.tanh(tensor)
  elif activation == 'sigmoid':
    return K.sigmoid(tensor)
  elif activation == 'relu':
    return K.relu(tensor)
  return tensor

def single_activation(activations):
  def fn(x):
    to_stack = []

    for i, activation in enumerate(activations):
      to_stack.append(K.flatten(activate(activation, tf.slice(x, [0, i], [-1, 1]))))
    return K.transpose(K.stack(to_stack))
  return fn
