import tensorflow as tf

def conv_bn_act(x, channels, kernel_size, stride, name, reuse, training, activation, padding='SAME'):
  return activation(
    tf.layers.batch_normalization(
      tf.layers.conv2d(x, channels, kernel_size, stride, padding=padding, activation=None, name=name), training=training,
      reuse=reuse, name='{}_bn'.format(name))
    , name='{}_a'.format(name))

def conv_transpose_bn_act(x, channels, kernel_size, stride, name, reuse, training, activation, padding='SAME'):
  return activation(
    tf.layers.batch_normalization(
      tf.layers.conv2d_transpose(x, channels, kernel_size, stride, padding=padding, activation=None, name=name), training=training,
      reuse=reuse, name='{}_bn'.format(name))
    , name='{}_a'.format(name))