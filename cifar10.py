from dataset.datasets import CIFAR10
from dataset.utils import batchify
from models.dcvaegan import DCVAEGAN

cifar = CIFAR10('cifar10')
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar.build_dataset(grayscale=False, flat=False)

model = DCVAEGAN([64, 32, 32, 3], 1e-5, 0.0003, tb_id=1)
model.compile()

for e in range(100):
  print("Epoch", e+1)
  for x_b in batchify(X_train_cifar, 64):
    enc_loss, dec_loss, dis_loss = model.fit(x_b)
