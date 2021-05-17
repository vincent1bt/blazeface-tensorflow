import tensorflow as tf
from tensorflow.keras import layers

from model_blocks import BlazeBlock

class BlazeModel(tf.keras.Model):
  def __init__(self):
    super(BlazeModel, self).__init__()

    self.conv = layers.Conv2D(24, (3, 3), strides=2, padding="same")
    self.activation = layers.ReLU()

    self.block_1 = BlazeBlock(24)
    self.block_2 = BlazeBlock(28)
    self.block_3 = BlazeBlock(32, strides=2)
    self.block_4 = BlazeBlock(36)
    self.block_5 = BlazeBlock(42)

    self.block_6 = BlazeBlock(48, strides=2)
    self.block_7 = BlazeBlock(56)
    self.block_8 = BlazeBlock(64)
    self.block_9 = BlazeBlock(72)
    self.block_10 = BlazeBlock(80)
    self.block_11 = BlazeBlock(88)

    self.block_12 = BlazeBlock(96, strides=2)
    self.block_13 = BlazeBlock(96)
    self.block_14 = BlazeBlock(96)
    self.block_15 = BlazeBlock(96)

    self.classifier_8 = layers.Conv2D(2, (1, 1), strides=(1, 1), activation="sigmoid")
    self.classifier_16 = layers.Conv2D(6, (1, 1), strides=(1, 1), activation="sigmoid")

    self.regressor_8 = layers.Conv2D(8, (1, 1), strides=(1, 1)) # 32
    self.regressor_16 = layers.Conv2D(24, (1, 1), strides=(1, 1)) # 96
  
  def call(self, x):
    B, H, W, C = x.shape

    x = self.conv(x)
    x = self.activation(x) # (B, 64, 64, 24)

    x = self.block_1(x) # (B, 64, 64, 24)
    x = self.block_2(x) # (B, 64, 64, 28)
    x = self.block_3(x) # (B, 32, 32, 32)
    x = self.block_4(x) # (B, 32, 32, 36)
    x = self.block_5(x) # (B, 32, 32, 42)

    # Double Blocks

    x = self.block_6(x) # (4, 16, 16, 48)
    x = self.block_7(x) # (4, 16, 16, 56)
    x = self.block_8(x) # (4, 16, 16, 64)
    x = self.block_9(x) # (4, 16, 16, 72)
    x = self.block_10(x) # (4, 16, 16, 80)
    x = self.block_11(x) # (4, 16, 16, 88) output size

    h = self.block_12(x) # (4, 8, 8, 96)
    h = self.block_13(h) # (4, 8, 8, 96)
    h = self.block_14(h) # (4, 8, 8, 96)
    h = self.block_15(h) # (4, 8, 8, 96) output size

    c1 = self.classifier_8(x) # B, 16, 16, 2 output size
    c1 = layers.Reshape((-1, 1))(c1) # B, 512, 1 output size

    c2 = self.classifier_16(h) # B, 8, 8, 6 output size
    c2 = layers.Reshape((-1, 1))(c2) # B, 384, 1 output size

    c = layers.concatenate([c1, c2], axis=1) # B, 896, 1

    r1 = self.regressor_8(x) # B, 16, 16, 8 output size
    r1 = layers.Reshape((-1, 4))(r1) # B, 512, 4 output size

    r2 = self.regressor_16(h) # B, 8, 8, 24 output size
    r2 = layers.Reshape((-1, 4))(r2) # B, 384, 4 output size

    r = layers.concatenate([r1, r2], axis=1) # B, 896, 4

    return r, c