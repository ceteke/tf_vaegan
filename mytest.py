from dataset.mnist import MNIST
from models.dcvaegan import DCVAEGAN

mnist = MNIST('digit')
(X_train, y_train), (X_test, y_test) = mnist.build_dataset(flat=False)

X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

model = DCVAEGAN([32, 64, 64, 3], 1e-5, 0.0002)
model.compile()