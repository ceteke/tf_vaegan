from dataset.datasets import CelebA
from models.dcvaegan import DCVAEGAN
from dataset.utils import batchify
import pickle

print("Loading dataset")
celeba = CelebA(path='/home/ceteke/Documents/datasets/img_align_celeba')
#X = celeba.preprocess_dataset(n_jobs=128)
X = celeba.load_dataset(128)

#pickle.dump(X, open('celeba_processed.pk', 'wb'), protocol=4)

model = DCVAEGAN([64, 64, 64, 3], 0.0003, tb_id=6, gamma=1e-1)
model.compile()
model.save('storage/5')
for e in range(500):
  print("Epoch", e+1)
  for x_b in batchify(X, 64):
    enc_loss, dec_loss, dis_loss = model.fit(x_b)
  model.save('storage/5')