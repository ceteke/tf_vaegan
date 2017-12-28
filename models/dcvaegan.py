from .modules import DiscriminatorBase, EncoderBase, DecoderBase
from .base import VAEGANBase
import tensorflow as tf
from .mylayers import conv_bn_act, conv_transpose_bn_act

class DCEncoder(EncoderBase):
  def __init__(self, latent_size, learning_rate):
    EncoderBase.__init__(self, latent_size, learning_rate)

  def net(self, x, training, reuse):
    c1 = conv_bn_act(x, 64, 5, (2, 2), padding='SAME', activation=tf.nn.relu, name='conv1', training=training, reuse=reuse)
    c2 = conv_bn_act(c1, 128, 5, (2, 2), padding='SAME', activation=tf.nn.relu, name='conv2', training=training, reuse=reuse)
    c3 = conv_bn_act(c2, 256, 5, (2, 2), padding='SAME', activation=tf.nn.relu, name='conv3', training=training, reuse=reuse)
    c3 = tf.contrib.layers.flatten(c3)
    hidden = tf.layers.dense(c3, 2048, activation=tf.nn.relu, name='fc')
    return hidden

class DCDecoder(DecoderBase):
  def __init__(self, learning_rate):
    DecoderBase.__init__(self, learning_rate)

  def net(self, x, training, reuse):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      d1 = tf.layers.dense(x, 8*8*256, activation=tf.nn.relu, name='fc')
      d1 = tf.reshape(d1, [-1, 8, 8, 256])
      d2 = conv_transpose_bn_act(d1, 256, 5, (2, 2), padding='SAME', activation=tf.nn.relu, name='deconv1', training=training, reuse=reuse)
      d3 = conv_transpose_bn_act(d2, 128, 5, (2, 2), padding='SAME', activation=tf.nn.relu, name='deconv2', training=training, reuse=reuse)
      d4 = conv_transpose_bn_act(d3, 32, 5, (2, 2), padding='SAME', activation=tf.nn.relu, name='deconv3', training=training, reuse=reuse)
      d5 = conv_transpose_bn_act(d4, 3, 5, (1, 1), padding='SAME', activation=tf.nn.tanh, name='deconv4', training=training, reuse=reuse)
    return d5

class DCDiscriminator(DiscriminatorBase):
  def __init__(self, learning_rate):
    DiscriminatorBase.__init__(self, learning_rate)

  def net(self, x, training, reuse):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      c1 = conv_bn_act(x, 32, 5, (1, 1), padding='SAME', activation=tf.nn.relu, name='conv1', training=training, reuse=reuse)
      c2 = conv_bn_act(c1, 128, 5, (2, 2), padding='SAME', activation=tf.nn.relu, name='conv2', training=training, reuse=reuse)
      c3 = conv_bn_act(c2, 256, 5, (2, 2), padding='SAME', activation=tf.nn.relu, name='conv3', training=training, reuse=reuse)
      c4 = conv_bn_act(c3, 256, 5, (2, 2), padding='SAME', activation=tf.nn.relu, name='conv4', training=training, reuse=reuse)
      reconstruction_layer = c4
      c4 = tf.contrib.layers.flatten(c4)
      hidden = tf.layers.dense(c4, 512, activation=tf.nn.relu, kernel_initializer=self.initializer, name='fc1')
      output = tf.layers.dense(hidden, 1, name='fc2')
    return reconstruction_layer, output

class DCVAEGAN(VAEGANBase):
  def __init__(self, input_shape, learning_rate, tb_id):
    VAEGANBase.__init__(self, input_shape, tb_id)
    self.encoder = DCEncoder(512, learning_rate)
    self.decoder = DCDecoder(learning_rate)
    self.discriminator = DCDiscriminator(learning_rate)
