import tensorflow as tf
import numpy as np

def variable_decorator(name):
  def decorator(fn):
    with tf.variable_scope(name) as scope:
      print(name)
      def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
      return wrapper
  return decorator
EPSILON = 1e-6
def GaussianLogDensity(x, mu, log_var, name='GaussianLogDensity'):
  c = np.log(2 * np.pi)
  var = tf.exp(log_var)
  x_mu2 = tf.square(x-mu)  # [Issue] not sure the dim works or not?
  x_mu2_over_var = tf.div(x_mu2, var + EPSILON)
  log_prob = -0.5 * (c + log_var + x_mu2_over_var)
  log_prob = tf.reduce_sum(log_prob, -1, name=name)  # keep_dims=True,
  return log_prob

def get_element_from_dict(input, dict, training):
  t = dict['type']

  if t == 'conv':
    layer = tf.layers.conv2d(input, dict['units'], dict['kernel'], dict['stride'], padding='SAME')
  elif t == 'conv_t':
    layer = tf.layers.conv2d_transpose(input, dict['units'], dict['kernel'], dict['stride'], padding='SAME')
  elif t == 'fc':
    layer = tf.layers.dense(input, dict['units'])
  elif t == 'flatten':
    return tf.layers.flatten(input)
  elif t == 'reshape':
    return tf.reshape(input, dict['shape'])
  else:
    raise Exception("Unknown layer type")

  if dict['bnorm'] == 1:
    layer = tf.layers.batch_normalization(layer, training=training)

  act = dict.get('act', None)

  if act == 'relu':
    return tf.nn.relu(layer)
  elif act == 'tanh':
    return tf.nn.tanh(layer)
  elif act == 'sigmoid':
    return tf.nn.sigmoid(layer)
  elif act is None:
    return layer
  else:
    raise Exception("Unknown activation type")

def get_optimizer(name, lr):
  if name == 'adam':
    return tf.train.AdamOptimizer(learning_rate=lr)
  elif name == 'rmsprop':
    return tf.train.RMSPropOptimizer(learning_rate=lr)
  elif name == 'sgd':
    return tf.train.GradientDescentOptimizer(learning_rate=lr)
  else:
    raise Exception("Unknown optimizer")
