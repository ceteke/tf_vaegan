import numpy as np

def normalize_unit_sphere(points):
  centroid = np.mean(points, axis=0)
  points -= centroid
  furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
  points /= furthest_distance
  return points

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