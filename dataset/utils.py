def batchify(X, n):
  l = len(X)
  for ndx in range(0, l, n):
    x = X[ndx:min(ndx + n, l)]
    if len(x) == n:
      yield x