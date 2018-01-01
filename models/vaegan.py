from .modules import DiscriminatorBase, EncoderBase, DecoderBase
from .base import VAEGANBase
from .tf_utils import get_optimizer

class VAEGAN(VAEGANBase):
  def __init__(self, input_shape, arch_json, total_steps, tb_id):
    gamma = arch_json['gamma']
    VAEGANBase.__init__(self, input_shape, gamma, tb_id)
    self.encoder = EncoderBase(arch_json['enc'], arch_json['optim'], arch_json['lr'], arch_json['decay'], total_steps)
    self.decoder = DecoderBase(arch_json['dec'], arch_json['optim'], arch_json['lr'], arch_json['decay'], total_steps)
    self.discriminator = DiscriminatorBase(arch_json['dis'], arch_json['optim'], arch_json['lr'], arch_json['decay'], total_steps)
