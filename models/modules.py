import tensorflow as tf
from .tf_utils import get_element_from_dict, get_optimizer

class ModuleBase(object):
  def __init__(self, arch, optimizer_name, lr, decay, total_steps, scope_name, dtype):
    self.dtype = dtype
    self.arch = arch
    self.scope_name = scope_name
    self.global_step, self.optimizer = get_optimizer(optimizer_name, lr, decay, total_steps, scope_name)

  def __call__(self, x, training, reuse=True):
    with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
      return self.net(x, training)

  def net(self, x, training):
    for layer in self.arch['net']:
      x = get_element_from_dict(x, layer, training)
    return x

class EncoderBase(ModuleBase):
  def __init__(self, arch, optimizer_name, lr, decay, total_steps, dtype=tf.float32):
    ModuleBase.__init__(self, arch, optimizer_name, lr, decay, total_steps, 'enc', dtype)
    self.latent_size = arch['z_dim']

  def __call__(self, x, training, reuse=True):
    with tf.variable_scope(self.scope_name, reuse=reuse):
      hidden = self.net(x, training)
      mu = tf.layers.dense(hidden, self.latent_size, name='mu')
      log_sigm_sq = tf.layers.dense(hidden, self.latent_size, name='log_sigm_sq')
      sigma = tf.sqrt(tf.exp(log_sigm_sq), name='sigma')
      eps = tf.random_normal(shape=tf.shape(sigma), name='epsilon')
      z = mu + sigma * eps
    return z, mu, sigma


class DecoderBase(ModuleBase):
  def __init__(self, arch, optimizer_name, lr, decay, total_steps, dtype=tf.float32):
    ModuleBase.__init__(self, arch, optimizer_name, lr, decay, total_steps, 'dec', dtype)


class DiscriminatorBase(ModuleBase):
  def __init__(self, arch, optimizer_name, lr, decay, total_steps, dtype=tf.float32):
    ModuleBase.__init__(self, arch, optimizer_name, lr, decay, total_steps, 'dis', dtype)
    self.feature_layer = arch['feature_layer']

  def net(self, x, training):
    for i, layer in enumerate(self.arch['net']):
      x = get_element_from_dict(x, layer, training)
      if i == self.feature_layer - 1:
        feature_layer = tf.layers.flatten(x)
    return feature_layer, x