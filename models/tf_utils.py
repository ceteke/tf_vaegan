import tensorflow as tf

def variable_decorator(name):
  def decorator(fn):
    with tf.variable_scope(name) as scope:
      print(name)
      def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
      return wrapper
  return decorator