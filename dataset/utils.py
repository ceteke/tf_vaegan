import numpy as np

def batchify(X, n):
  l = len(X)
  for ndx in range(0, l, n):
    x = X[ndx:min(ndx + n, l)]
    if len(x) == n:
      yield x

def get_subset(X_train, y_train, train_size):
  labels = [np.where(y_train == label)[0] for label in range(10)]
  for i, l in enumerate(labels):
    np.random.shuffle(l)

  train_indexes = np.concatenate([l[:train_size] for l in labels])
  np.random.shuffle(train_indexes)
  X_train_subset = np.take(X_train, train_indexes, axis=0)
  y_train_subset = np.take(y_train, train_indexes, axis=0)

  return X_train_subset, y_train_subset