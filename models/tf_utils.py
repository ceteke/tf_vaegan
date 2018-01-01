import tensorflow as tf

def variable_decorator(name):
  def decorator(fn):
    with tf.variable_scope(name) as scope:
      print(name)
      def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
      return wrapper
  return decorator

def gaussian_loss(x, mu, name='feature_loss'):
  x_mu2 = tf.square(x-mu)
  log_prob = tf.reduce_mean(0.5 * tf.reduce_sum(x_mu2, -1, name=name))  # keep_dims=True,
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

  act = dict.get('act', None)

  if dict['bnorm'] == 1:
    layer = tf.layers.batch_normalization(layer, training=training)

  if act == 'relu':
    layer = tf.nn.relu(layer)
  elif act == 'tanh':
    layer = tf.nn.tanh(layer)
  elif act == 'sigmoid':
    layer = tf.nn.sigmoid(layer)
  elif act is None:
    layer = layer
  else:
    raise Exception("Unknown activation type")

  return layer

def get_optimizer(name, lr):
  if name == 'adam':
    return tf.train.AdamOptimizer(learning_rate=lr)
  elif name == 'rmsprop':
    return tf.train.RMSPropOptimizer(learning_rate=lr)
  elif name == 'sgd':
    return tf.train.GradientDescentOptimizer(learning_rate=lr)
  else:
    raise Exception("Unknown optimizer")
