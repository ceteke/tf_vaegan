from .modules import DiscriminatorBase, EncoderBase, DecoderBase
from .base import VAEGANBase

class VAEGAN(VAEGANBase):
  def __init__(self, input_shape, arch_json, tb_id):
    gamma = arch_json['gamma']
    VAEGANBase.__init__(self, input_shape, gamma, tb_id)
    self.encoder = EncoderBase(arch_json['enc'])
    self.decoder = DecoderBase(arch_json['dec'])
    self.discriminator = DiscriminatorBase(arch_json['dis'])
