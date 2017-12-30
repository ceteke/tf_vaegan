from .modules import DiscriminatorBase, EncoderBase, DecoderBase
from .base import VAEGANBase
import tensorflow as tf

class DCEncoder(EncoderBase):
  def __init__(self, latent_size, learning_rate):
    EncoderBase.__init__(self, latent_size, learning_rate)

  def net(self, x, training, reuse):
    c1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(x, 64, 5, (2, 2), padding='SAME', activation=None, name='conv1'), training=training))
    c2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(c1, 128, 5, (2, 2), padding='SAME', activation=None, name='conv2'), training=training))
    c3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(c2, 256, 5, (2, 2), padding='SAME', activation=None, name='conv3'), training=training))
    c3 = tf.nn.relu(tf.contrib.layers.flatten(c3))
    hidden = tf.nn.relu(tf.layers.batch_normalization(tf.layers.dense(c3, 2048, activation=None, name='fc'), training=training))
    return hidden

class DCDecoder(DecoderBase):
  def __init__(self, learning_rate):
    DecoderBase.__init__(self, learning_rate)

  def net(self, x, training, reuse):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      d1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.dense(x, 8*8*256, activation=None, name='fc'), training=training))
      d1 = tf.reshape(d1, [-1, 8, 8, 256])
      d2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(d1, 256, 5, (2, 2), padding='SAME', activation=None, name='deconv1'), training=training))
      d3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(d2, 128, 5, (2, 2), padding='SAME', activation=None, name='deconv2'), training=training))
      d4 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d_transpose(d3, 32, 5, (2, 2), padding='SAME', activation=None, name='deconv3'), training=training))
      d5 = tf.layers.conv2d_transpose(d4, 3, 5, (1, 1), padding='SAME', activation=tf.nn.tanh, name='deconv4')
    return d5

class DCDiscriminator(DiscriminatorBase):
  def __init__(self, learning_rate):
    DiscriminatorBase.__init__(self, learning_rate)

  def net(self, x, training, reuse):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      c1 = tf.nn.relu(tf.layers.conv2d(x, 32, 5, (1, 1), padding='SAME', activation=None, name='conv1'))
      c2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(c1, 128, 5, (2, 2), padding='SAME', activation=None, name='conv2'), training=training))
      c3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(c2, 256, 5, (2, 2), padding='SAME', activation=None, name='conv3'), training=training))
      c4 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(c3, 256, 5, (2, 2), padding='SAME', activation=None, name='conv4'), training=training))
      c4 = tf.contrib.layers.flatten(c4)
      reconstruction_layer = c4
      hidden = tf.nn.relu(tf.layers.batch_normalization(tf.layers.dense(c4, 512, activation=None, name='fc1'), training=training))
      output = tf.layers.dense(hidden, 1, name='fc2')
    return reconstruction_layer, output

class DCVAEGAN(VAEGANBase):
  def __init__(self, input_shape, learning_rate, gamma, tb_id):
    VAEGANBase.__init__(self, input_shape, gamma, tb_id)
    self.encoder = DCEncoder(512, learning_rate)
    self.decoder = DCDecoder(learning_rate)
    self.discriminator = DCDiscriminator(learning_rate)
