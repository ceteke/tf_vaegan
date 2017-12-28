import tensorflow as tf
import numpy as np

EPSILON = 1e-6

def variable_decorator(name):
  def decorator(fn):
    with tf.variable_scope(name) as scope:
      print(name)
      def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
      return wrapper
  return decorator


def NLLNormal(x, mu, log_var, name='GaussianLogDensity'):
  c = np.log(2 * np.pi)
  var = tf.exp(log_var)
  x_mu2 = tf.square(tf.sub(x, mu))  # [Issue] not sure the dim works or not?
  x_mu2_over_var = tf.div(x_mu2, var + EPSILON)
  log_prob = -0.5 * (c + log_var + x_mu2_over_var)
  log_prob = tf.reduce_sum(log_prob, -1, name=name)  # keep_dims=True,
  return log_prob