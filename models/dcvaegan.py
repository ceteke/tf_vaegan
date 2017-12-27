from .modules import DiscriminatorBase, EncoderBase, DecoderBase
from .base import VAEGANBase
import tensorflow as tf

class DCEncoder(EncoderBase):
  def __init__(self, latent_size, learning_rate):
    EncoderBase.__init__(self, latent_size, learning_rate)

  def net(self, x, training, reuse):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      c1 = tf.layers.conv2d(x, 64, 5, (2, 2), padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv1')
      c1 = tf.layers.batch_normalization(c1, training=training)
      c2 = tf.layers.conv2d(c1, 128, 5, (2, 2), padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv2')
      c2 = tf.layers.batch_normalization(c2, training=training)
      c3 = tf.layers.conv2d(c2, 256, 5, (2, 2), padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv3')
      c3 = tf.layers.batch_normalization(c3, training=training)
      c3 = tf.contrib.layers.flatten(c3)
      hidden = tf.layers.dense(c3, 2048, activation=tf.nn.relu, kernel_initializer=self.initializer,
                               kernel_regularizer=self.regularizer, name='fc')
      hidden = tf.contrib.layers.flatten(hidden)
    return hidden

class DCDecoder(DecoderBase):
  def __init__(self, learning_rate):
    DecoderBase.__init__(self, learning_rate)

  def net(self, x, training, reuse):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      d1 = tf.layers.dense(x, 4*4*256, activation=tf.nn.relu, kernel_initializer=self.initializer,
                           kernel_regularizer=self.regularizer, name='fc')
      d1 = tf.layers.batch_normalization(d1, training=training)
      d1 = tf.reshape(d1, [-1, 4, 4, 256])
      d3 = tf.layers.conv2d_transpose(d1, 256, 5, (2, 2), padding='SAME', activation=tf.nn.relu,
                                      kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                      name='deconv1')
      d3 = tf.layers.batch_normalization(d3, training=training)
      d4 = tf.layers.conv2d_transpose(d3, 128, 5, (2, 2), padding='SAME', activation=tf.nn.relu,
                                      kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                      name='deconv2')
      d4 = tf.layers.batch_normalization(d4, training=training)
      d5 = tf.layers.conv2d_transpose(d4, 32, 5, (2, 2), activation=tf.nn.sigmoid, padding='SAME',
                                      kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                      name='deconv3')
      d5 = tf.layers.batch_normalization(d5, training=training)
      d6 = tf.layers.conv2d_transpose(d5, 3, 5, (1, 1), activation=tf.nn.tanh, padding='SAME',
                                      kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                      name='deconv4')
      d6 = tf.layers.batch_normalization(d6, training=training)
    return d6

class DCDiscriminator(DiscriminatorBase):
  def __init__(self, learning_rate):
    DiscriminatorBase.__init__(self, learning_rate)

  def net(self, x, training, reuse):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      c1 = tf.layers.conv2d(x, 32, 5, (1, 1), padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv1')
      c1 = tf.layers.batch_normalization(c1, training=training)
      c2 = tf.layers.conv2d(c1, 128, 5, (2, 2), padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv2')
      c2 = tf.layers.batch_normalization(c2, training=training)
      c3 = tf.layers.conv2d(c2, 256, 5, (2, 2), padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv3')
      c3 = tf.layers.batch_normalization(c3, training=training)
      reconstruction_layer = c3
      c4 = tf.layers.conv2d(c3, 256, 5, (2, 2), padding='SAME', activation=tf.nn.relu,
                            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, name='conv4')
      c4 = tf.layers.batch_normalization(c4, training=training)
      c4 = tf.contrib.layers.flatten(c4)
      hidden = tf.layers.dense(c4, 512, activation=tf.nn.relu, kernel_initializer=self.initializer,
                               kernel_regularizer=self.regularizer, name='fc1')
      hidden = tf.layers.batch_normalization(hidden, training=training)
      output = tf.layers.dense(hidden, 1, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                               name='fc2')
    return reconstruction_layer, output

class DCVAEGAN(VAEGANBase):
  def __init__(self, input_shape, gamma, learning_rate, tb_id):
    VAEGANBase.__init__(self, input_shape, gamma, tb_id)
    self.encoder = DCEncoder(128, learning_rate)
    self.decoder = DCDecoder(learning_rate)
    self.discriminator = DCDiscriminator(learning_rate)
