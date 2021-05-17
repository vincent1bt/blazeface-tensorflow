import tensorflow as tf
from tensorflow.keras import layers

class BlazeBlock(tf.keras.Model):
  def __init__(self, filters, strides=1):
    super(BlazeBlock, self).__init__()
    self.strides = strides
    self.filters = filters

    if strides == 2:
      self.pool = layers.MaxPool2D()

    self.dw_conv = layers.DepthwiseConv2D((3, 3), strides=strides, padding="same")
    self.conv = layers.Conv2D(filters, (1, 1), strides=(1, 1))

    self.norm_1 = layers.BatchNormalization()
    self.norm_2 = layers.BatchNormalization()

    self.activation = layers.ReLU()
  
  def call(self, x_input):
    x = self.dw_conv(x_input)
    x = self.norm_1(x)
    x = self.conv(x)
    x = self.norm_2(x)
    
    if self.strides == 2:
      x_input = self.pool(x_input)
    
    padding = self.filters - x_input.shape[-1]
    
    if padding != 0:
      padding_values = [[0, 0], [0, 0], [0, 0], [0, padding]]
      x_input = tf.pad(x_input, padding_values)
    
    x = x + x_input
    x = self.activation(x)

    return x