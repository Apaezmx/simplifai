from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

# Only availble with tf.
class CustomDense(Layer):

  def __init__(self, activations, **kwargs):
    self.activations = activations
    self.output_dim = len(activations)
    super(CustomDense, self).__init__(**kwargs)

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    self.kernel = self.add_weight(shape=(input_shape[1], self.output_dim),
                                    initializer='uniform',
                                    trainable=True)
    super(CustomDense, self).build(input_shape)  # Be sure to call this somewhere!

  def activate(self, activation, tensor):
    if activation == 'tanh':
      return K.tanh(tensor)
    elif activation == 'sigmoid':
      return K.sigmoid(tensor)
    elif activation == 'relu':
      return K.relu(tensor)
    return tensor

  def call(self, x, mask=None):
    res =  K.dot(x, self.kernel)
    to_stack = []

    for i, activation in enumerate(self.activations):
      to_stack.append(K.flatten(self.activate(activation, tf.slice(res, [0, i], [-1, 1]))))
    return K.transpose(K.stack(to_stack))

  def get_output_shape_for(self, input_shape):
    return (input_shape[0], self.output_dim)
