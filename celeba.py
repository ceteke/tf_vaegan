from dataset.datasets import CelebA

print("Loading dataset")
celeba = CelebA(path='/home/ceteke/Documents/datasets/img_align_celeba')
X = celeba.build_dataset()

print(X.shape)