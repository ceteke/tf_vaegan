import tensorflow as tf

class VAEGANBase(object):
  def __init__(self, input_shape, gamma, tb_id, dtype=tf.float32, verbose=1):
    self.input_shape = input_shape
    self.dtype = dtype
    self.gamma = gamma
    self.encoder = None
    self.decoder = None
    self.discriminator = None
    self.verbose = verbose
    self.global_step = 0
    self.tb_id = tb_id

  def form_variables(self):
    self.input = tf.placeholder(dtype=self.dtype, shape=self.input_shape, name='input')
    self.training = tf.placeholder(dtype=tf.bool, name='training')

  def kl_divergence(self, mu, sigma):
    return -0.5 * tf.reduce_sum(1 + tf.log(tf.square(sigma)) - tf.square(mu) - tf.square(sigma))

  def sample_latent(self, latent_size):
    return tf.random_normal(shape=[self.input_shape[0], latent_size])

  def build_graph(self):
    z, mu, sigma = self.encoder(self.input, training=self.training, reuse=False)

    x_tilda = self.decoder(z, training=self.training, reuse=False)

    dis_l_tilda, fake_disc = self.discriminator(x_tilda, training=self.training, reuse=False)
    dis_l_x, real_disc = self.discriminator(self.input, training=self.training)

    z_p = self.sample_latent(self.encoder.latent_size)
    x_p = self.decoder(z_p, training=self.training)

    _, sampled_disc = self.discriminator(x_p, training=self.training)

    # Reconstruction Loss (not pixelwise but featurewise)
    feature_loss = tf.losses.mean_squared_error(dis_l_x, dis_l_tilda) # Gaussian loss has some 1/sqrt(pi) stuff but since we are optimizing with the same constants everytime this is simply squared error

    # Encoder Loss
    prior_loss = self.kl_divergence(mu, sigma)
    self.encoder_loss = prior_loss + feature_loss

    # Discriminator Loss
    d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_disc), logits=real_disc))
    d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_disc), logits=fake_disc))
    d_sampled = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(sampled_disc), logits=sampled_disc))
    self.discriminator_loss = d_real + d_fake + d_sampled

    # Genereator Loss
    g_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_disc), logits=fake_disc))
    g_sampled = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(sampled_disc), logits=sampled_disc))
    self.generator_loss = g_fake + g_sampled + self.gamma*feature_loss

    # Get each network's parameters
    t_vars = tf.trainable_variables()
    if self.verbose == 1:
      for var in t_vars:
        print(var)
    self.d_vars = [var for var in t_vars if 'dis' in var.name]
    self.g_vars = [var for var in t_vars if 'dec' in var.name]
    self.e_vars = [var for var in t_vars if 'enc' in var.name]

    # Update parameters
    enc_grads = self.encoder.optimizer.compute_gradients(self.encoder_loss, var_list=self.e_vars)
    self.enc_update_op = self.encoder.optimizer.apply_gradients(enc_grads)
    dec_grads = self.decoder.optimizer.compute_gradients(self.generator_loss, var_list=self.g_vars)
    self.dec_update_op = self.decoder.optimizer.apply_gradients(dec_grads)
    dis_grads = self.discriminator.optimizer.compute_gradients(self.discriminator_loss, var_list=self.d_vars)
    self.dis_update_op = self.discriminator.optimizer.apply_gradients(dis_grads)

    tf.summary.scalar('enc_loss', self.encoder_loss)
    tf.summary.scalar('dec_loss', self.generator_loss)
    tf.summary.scalar('dis_loss', self.discriminator_loss)
    tf.summary.image('reconstructed', x_tilda)
    tf.summary.image('sampled', x_p)
    tf.summary.image('input', self.input)
    self.merged = tf.summary.merge_all()

  def compile(self):
    self.form_variables()
    self.build_graph()
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.train_writer = tf.summary.FileWriter('log/{}/train'.format(self.tb_id), self.sess.graph)

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