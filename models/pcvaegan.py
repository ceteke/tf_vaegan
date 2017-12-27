from .modules import DiscriminatorBase, EncoderBase, DecoderBase
from .base import VAEGANBase
import tensorflow as tf

class PCEncoder(EncoderBase):
  def __init__(self, n, latent_size, learning_rate):
    EncoderBase.__init__(self, latent_size, learning_rate)
    self.n = n

  def net(self, x, training, reuse):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      c1 = tf.layers.conv2d(x, 64, (1, 3), padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv1')
      c2 = tf.layers.conv2d(c1, 64, 1, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv2')
      c3 = tf.layers.conv2d(c2, 128, 1, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv3')
      c4 = tf.layers.conv2d(c3, 1024, 1, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv4')
      mp = tf.layers.max_pooling2d(c4, (self.n, 1), 1)
      hidden = tf.contrib.layers.flatten(mp)
    return hidden

class PCDecoder(DecoderBase):
  def __init__(self, n, learning_rate):
    DecoderBase.__init__(self, learning_rate)
    self.n = n

  def net(self, x, training, reuse):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      d1 = tf.layers.dense(x, 1024, activation=tf.nn.relu, kernel_initializer=self.initializer,
                           kernel_regularizer=self.regularizer, name='fc1')
      d2 = tf.layers.dense(d1, 2048, activation=tf.nn.relu, kernel_initializer=self.initializer,
                           kernel_regularizer=self.regularizer, name='fc2')
      d3 = tf.layers.dense(d2, self.n * 3, kernel_initializer=self.initializer,
                           kernel_regularizer=self.regularizer, name='fc3') # Sigmoid?
      d3 = tf.reshape(d3, [-1, self.n, 3, 1])
    return d3

class PCDiscriminator(DiscriminatorBase):
  def __init__(self, n, learning_rate):
    DiscriminatorBase.__init__(self, learning_rate)
    self.n = n

  def net(self, x, training, reuse):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      c1 = tf.layers.conv2d(x, 64, (1, 3), padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv1')
      c2 = tf.layers.conv2d(c1, 64, 1, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv2')
      c3 = tf.layers.conv2d(c2, 128, 1, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv3')
      c4 = tf.layers.conv2d(c3, 1024, 1, padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv4')
      mp = tf.layers.max_pooling2d(c4, (self.n, 1), 1)
      mp = tf.contrib.layers.flatten(mp)
      hidden = tf.layers.dense(mp, 512, activation=tf.nn.relu, kernel_initializer=self.initializer,
                               kernel_regularizer=self.regularizer, name='fc1')
      reconstruction_layer = hidden
      output = tf.layers.dense(hidden, 1, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                               name='fc2')
    return reconstruction_layer, output

class PCVAEGAN(VAEGANBase):
  def __init__(self, input_shape, gamma, learning_rate, tb_id):
    VAEGANBase.__init__(self, input_shape, gamma, tb_id)
    self.encoder = PCEncoder(input_shape[1], 512, learning_rate)
    self.decoder = PCDecoder(input_shape[1], learning_rate)
    self.discriminator = PCDiscriminator(input_shape[1], learning_rate)
