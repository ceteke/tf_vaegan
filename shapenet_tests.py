from models.vaegan import VAEGAN
import argparse
from dataset.datasets import ModelNet10
import os, json
from dataset.utils import batchify
import numpy as np
from sklearn.svm import SVC
from utils import save_pc

print("Loading dataset", flush=True)
modelnet = ModelNet10('model10_train.pk', 'model10_test.pk', 1024)
X_train, y_train, X_test, y_test = modelnet.process()

total_steps = int(len(X_train)/32)
print("Steps/epoch:", total_steps, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('--arch', dest='arch', help='Path of the architecture file', required=True)
parser.add_argument('--tb_id', dest='tb_id', help='Tensorboard log id and model id to save to storege/id', required=True)
parser.add_argument('--n_jobs', dest='n_jobs', default=128, help='Jobs to run in parallel for reading the dataset', type=int)
parser.add_argument('--device', dest='device_id', default=0, help='CUDA device to run the model', type=int)
FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device_id)

arch_json = json.load(open(FLAGS.arch))

vaegan = VAEGAN([32, 1024, 3, 1], arch_json, total_steps, FLAGS.tb_id)
vaegan.compile()

print("Restoring model")
vaegan.load('storage/{}/vaegan'.format(FLAGS.tb_id))

X_train_trans = []
X_test_trans = []

recons = vaegan.reconstruct(X_train[:32])
save_pc(recons[0], 'test_recons.pcd')
save_pc(X_train[0], 'test_orig.pcd')

print("Getting Representations")
for x_b in batchify(X_train, 32):
  latent = vaegan.representation(x_b)
  X_train_trans.append(latent)

for x_b in batchify(X_test, 32):
  latent = vaegan.representation(x_b)
  X_test_trans.append(latent)

X_train_trans = np.concatenate(X_train_trans)
X_test_trans = np.concatenate(X_test_trans)

y_train = y_train[:len(X_train_trans)]
y_test = y_test[:len(X_test_trans)]

print("Training SVM")
lsvc = SVC()
lsvc.fit(X_train_trans, y_train)
print(lsvc.score(X_test_trans, y_test))