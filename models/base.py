import tensorflow as tf
from os.path import join
from .tf_utils import gaussian_loss

class VAEGANBase(object):
  def __init__(self, input_shape, gamma, tb_id, dtype=tf.float32, verbose=1):
    self.input_shape = input_shape
    self.dtype = dtype
    self.encoder = None
    self.decoder = None
    self.discriminator = None
    self.verbose = verbose
    self.global_step = 0
    self.tb_id = tb_id
    self.gamma = gamma

  def save(self, path):
    self.saver.save(self.sess, join(path, 'vaegan'))

  def form_variables(self):
    self.input = tf.placeholder(dtype=self.dtype, shape=self.input_shape, name='input')
    self.training = tf.placeholder(dtype=tf.bool, name='training')

  def kl_divergence(self, mu, sigma):
    with tf.name_scope("kl_div"):
      return -0.5 * tf.reduce_sum(1 + tf.log(tf.square(sigma)) - tf.square(mu) - tf.square(sigma), -1)

  def sample_latent(self):
    return tf.random_normal(shape=[self.input_shape[0], self.encoder.latent_size], name='prior')

  def zero_uniform(self):
    return tf.random_uniform([self.input_shape[0], 1], 0, 0.3, self.dtype)

  def one_uniform(self):
    return tf.random_uniform([self.input_shape[0], 1], 0.7, 1.2, self.dtype)

  def build_graph(self):
    z, mu, sigma = self.encoder(self.input, training=self.training, reuse=False)

    x_tilda = self.decoder(z, training=self.training, reuse=False)
    dis_l_tilda, fake_disc = self.discriminator(x_tilda, training=self.training, reuse=False)
    dis_l_x, real_disc = self.discriminator(self.input, training=self.training)

    z_p = self.sample_latent()
    x_p = self.decoder(z_p, training=self.training)

    _, sampled_disc = self.discriminator(x_p, training=self.training)

    # Reconstruction Loss (not pixelwise but featurewise)
    feature_loss = -tf.reduce_mean(gaussian_loss(dis_l_x, dis_l_tilda, tf.zeros_like(dis_l_tilda)))
    # Encoder Loss
    prior_loss = tf.reduce_mean(self.kl_divergence(mu, sigma))

    self.encoder_loss = prior_loss + feature_loss

    # Discriminator Loss
    d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.one_uniform(), logits=real_disc))
    d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.zero_uniform(), logits=fake_disc))
    d_sampled = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.zero_uniform(), logits=sampled_disc))
    self.discriminator_loss = d_real + d_fake + d_sampled

    # Genereator Loss
    g_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.one_uniform(), logits=fake_disc))
    g_sampled = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.one_uniform(), logits=sampled_disc))
    generator_dis_loss = g_fake + g_sampled
    self.generator_loss = generator_dis_loss + self.gamma*feature_loss

    # Get each network's parameters
    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if self.discriminator.scope_name in var.name]
    self.g_vars = [var for var in t_vars if self.decoder.scope_name in var.name]
    self.e_vars = [var for var in t_vars if self.encoder.scope_name in var.name]
    if self.verbose == 1:
      print("Discrimnator Variables:", flush=True)
      for d_var in self.d_vars:
        print(d_var, flush=True)
      print("Generator (Decoder) Variables:", flush=True)
      for g_var in self.g_vars:
        print(g_var, flush=True)
      print("Encoder Variables:", flush=True)
      for e_var in self.e_vars:
        print(e_var, flush=True)
    # Update parameters
    enc_grads = self.encoder.optimizer.compute_gradients(self.encoder_loss, var_list=self.e_vars)
    self.enc_update_op = self.encoder.optimizer.apply_gradients(enc_grads, self.encoder.global_step)
    dec_grads = self.decoder.optimizer.compute_gradients(self.generator_loss, var_list=self.g_vars)
    self.dec_update_op = self.decoder.optimizer.apply_gradients(dec_grads, global_step=self.decoder.global_step)
    dis_grads = self.discriminator.optimizer.compute_gradients(self.discriminator_loss, var_list=self.d_vars)
    self.dis_update_op = self.discriminator.optimizer.apply_gradients(dis_grads, global_step=self.discriminator.global_step)

    tf.summary.scalar('enc_loss', self.encoder_loss)
    tf.summary.scalar('kl_div', prior_loss)
    tf.summary.scalar('feature_loss', feature_loss)
    tf.summary.scalar('generator_loss', generator_dis_loss)
    tf.summary.scalar('dec_loss', self.generator_loss)
    tf.summary.scalar('dis_loss', self.discriminator_loss)
    # tf.summary.image('reconstructed', x_tilda)
    # tf.summary.image('sampled', x_p)
    # tf.summary.image('input', self.input)
    self.merged = tf.summary.merge_all()

  def compile(self):
    self.form_variables()
    self.build_graph()
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.train_writer = tf.summary.FileWriter('log/{}/train'.format(self.tb_id), self.sess.graph)
    self.saver = tf.train.Saver()

  def fit(self, X):
    self.global_step += 1
    t = self.sess.run([self.encoder_loss, self.generator_loss, self.discriminator_loss,
                       self.enc_update_op, self.dec_update_op, self.dis_update_op, self.merged],
                      feed_dict={self.input: X,
                                 self.training: True})
    self.train_writer.add_summary(t[-1], self.global_step)
    return t[0], t[1], t[2]

  def eval(self, X):
    t = self.sess.run([self.encoder_loss, self.generator_loss, self.discriminator_loss,
                       self.enc_update_op, self.dec_update_op, self.dis_update_op],
                      feed_dict={self.input: X,
                                 self.training: False})
    return t[0], t[1], t[2]