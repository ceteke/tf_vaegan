import os, struct
from array import array as pyarray
import numpy as np

class MNIST(object):
  def __init__(self, path):
    self.path = path

  def build_dataset(self, flat=False):
    X_train, y_train = self.load('training')
    X_test, y_test = self.load('testing')

    X_train, X_test = X_train / 255, X_test / 255

    if flat:
      X_train, X_test = X_train.reshape(-1, 784), X_test.reshape(-1, 784)
    else:
      X_train, X_test = X_train.reshape(-1,1,28,28), X_test.reshape(-1,1,28,28)

    return (X_train, y_train), (X_test, y_test)

  def load(self, dataset, label_idxs=np.arange(10)):
    if dataset == "training":
      fname_img = os.path.join(self.path, 'train-images-idx3-ubyte')
      fname_lbl = os.path.join(self.path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
      fname_img = os.path.join(self.path, 't10k-images-idx3-ubyte')
      fname_lbl = os.path.join(self.path, 't10k-labels-idx1-ubyte')
    else:
      raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in label_idxs]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.float16)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
      images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
      labels[i] = lbl[ind[i]]

    return images, labels