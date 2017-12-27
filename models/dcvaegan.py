from .modules import DiscriminatorBase, EncoderBase, DecoderBase
from .base import VAEGANBase
import tensorflow as tf

class DCEncoder(EncoderBase):
  def __init__(self, latent_size, learning_rate):
    EncoderBase.__init__(self, latent_size, learning_rate)

  def net(self, x, reuse):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      c1 = tf.layers.conv2d(x, 64, 5, (2, 2), padding='SAME', activation=tf.nn.leaky_relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv1')
      c2 = tf.layers.conv2d(c1, 128, 5, (2, 2), padding='SAME', activation=tf.nn.leaky_relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv2')
      c3 = tf.layers.conv2d(c2, 256, 5, (2, 2), padding='SAME', activation=tf.nn.leaky_relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv3')
      c3 = tf.contrib.layers.flatten(c3)
      hidden = tf.layers.dense(c3, 1024, activation=tf.nn.leaky_relu, kernel_initializer=self.initializer,
                               kernel_regularizer=self.regularizer, name='fc')
    return hidden

class DCDecoder(DecoderBase):
  def __init__(self, learning_rate):
    DecoderBase.__init__(self, learning_rate)

  def net(self, x, reuse):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      d1 = tf.layers.dense(x, 8*8*256, activation=tf.nn.leaky_relu, kernel_initializer=self.initializer,
                           kernel_regularizer=self.regularizer, name='fc')
      d1 = tf.reshape(d1, [-1, 8, 8, 256])
      d3 = tf.layers.conv2d_transpose(d1, 128, 5, (2, 2), padding='SAME', activation=tf.nn.leaky_relu,
                                      kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                      name='deconv1')
      d4 = tf.layers.conv2d_transpose(d3, 64, 5, (2, 2), padding='SAME', activation=tf.nn.leaky_relu,
                                      kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                      name='deconv2')
      d5 = tf.layers.conv2d_transpose(d4, 3, 5, (2, 2), activation=tf.nn.sigmoid, padding='SAME',
                                      kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                      name='deconv3')
    return d5

class DCDiscriminator(DiscriminatorBase):
  def __init__(self, learning_rate):
    DiscriminatorBase.__init__(self, learning_rate)

  def net(self, x, reuse):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      c1 = tf.layers.conv2d(x, 32, 5, (2, 2), padding='SAME', activation=tf.nn.leaky_relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv1')
      c2 = tf.layers.conv2d(c1, 128, 5, (2, 2), padding='SAME', activation=tf.nn.leaky_relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv2')
      c3 = tf.layers.conv2d(c2, 256, 5, (2, 2), padding='SAME', activation=tf.nn.leaky_relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv3')
      c4 = tf.layers.conv2d(c3, 256, 5, (2, 2), padding='SAME', activation=tf.nn.leaky_relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv4')
      reconstruction_layer = c4
      c4 = tf.contrib.layers.flatten(c4)
      hidden = tf.layers.dense(c4, 256, activation=tf.nn.leaky_relu, kernel_initializer=self.initializer,
                               kernel_regularizer=self.regularizer, name='fc1')
      output = tf.layers.dense(hidden, 1, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                               name='fc2')
    return reconstruction_layer, output

class DCVAEGAN(VAEGANBase):
  def __init__(self, input_shape, gamma, learning_rate):
    VAEGANBase.__init__(self, input_shape, gamma)
    self.encoder = DCEncoder(128, learning_rate)
    self.decoder = DCDecoder(learning_rate)
    self.discriminator = DCDiscriminator(learning_rate)
