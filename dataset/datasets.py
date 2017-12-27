import os, struct
from array import array as pyarray
import numpy as np
import pickle

def batchify(X, y, n):
  l = len(X)
  for ndx in range(0, l, n):
    yield X[ndx:min(ndx + n, l)], y[ndx:min(ndx + n, l)]

def batchify_single(X, n):
  l = len(X)
  for ndx in range(0, l, n):
    yield X[ndx:min(ndx + n, l)]

def get_subset(X_train, y_train, train_size):
  labels = [np.where(y_train == label)[0] for label in range(10)]
  for i, l in enumerate(labels):
    np.random.shuffle(l)

  train_indexes = np.concatenate([l[:train_size] for l in labels])
  np.random.shuffle(train_indexes)
  X_train_subset = np.take(X_train, train_indexes, axis=0)
  y_train_subset = np.take(y_train, train_indexes, axis=0)

  return X_train_subset, y_train_subset

class MNIST(object):
  def __init__(self, path):
    self.path = path

  def build_dataset(self, flat=False):
    X_train, y_train = self.load('training')
    X_test, y_test = self.load('testing')

    X_train, X_test = X_train / 255, X_test / 255
    y_train = y_train.ravel()
    y_test = y_test.ravel()

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

class CIFAR10(object):
  def __init__(self, path):
    self.path = path

  def build_dataset(self, grayscale=False, flat=False):
    X_train, y_train, X_test, y_test = self.load()
    X_train, X_test = X_train/255, X_test/255
    if grayscale:
      X_train = np.dot(X_train, [0.299, 0.587, 0.114])
      X_test = np.dot(X_test, [0.299, 0.587, 0.114])
      X_train = X_train.reshape((-1, 1, 32, 32))
      X_test = X_test.reshape((-1, 1, 32, 32))
    else:
      X_train = X_train.reshape((-1, 32, 32, 3))
      X_test = X_test.reshape((-1, 32, 32, 3))
    y_train, y_test = np.array(y_train), np.array(y_test)
    if flat:
      if grayscale:
        return(X_train.reshape(-1, 1024), y_train), (X_test.reshape(-1, 1024), y_test)
      else:
        return(X_train.reshape(-1, 1024*3), y_train), (X_test.reshape(-1, 1024*3), y_test)
    return (X_train, y_train), (X_test, y_test)

  def open_jar(self, file):
    with open(os.path.join(self.path, '{}'.format(file)), 'rb') as fo:
      dict = pickle.load(fo, encoding='bytes')
      X_batch = dict[b'data']
      X_batch = np.transpose(np.reshape(X_batch, (-1, 3, 32, 32)), (0, 2, 3, 1))
      y_batch = dict[b'labels']
    return X_batch, y_batch

  def load(self):
    X = []
    y = []
    for i in range(1, 6):
      X_batch, y_batch = self.open_jar('data_batch_{}'.format(i))
      X.append(X_batch)
      y.append(y_batch)
    X_test, y_test = self.open_jar('test_batch')
    return np.concatenate(X), np.concatenate(y), X_test, y_test

class AliveNet(object):
  def __init__(self, path):
    self.path = path

  def build_dataset(self, flat=False):
    X, y = self.load()
    X = np.array(X)
    y = np.array(y)
    if flat:
      return X.reshape(-1, 32*32), y
    return X, y

  def load(self):
    X = pickle.load(open(os.path.join(self.path, 'images.pk'), 'rb'))
    y = pickle.load(open(os.path.join(self.path, 'features.pk'), 'rb'))
    return X, y