from dataset.datasets import CelebA
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--db', dest='db_path', help='Path of the database', required=True)
parser.add_argument('--n_jobs', dest='n_jobs', default=128, help='Jobs to run in parallel for reading the dataset', type=int)
FLAGS = parser.parse_args()

print("Forming dataset", flush=True)
# /home/ceteke/Documents/datasets/img_align_celeba
celeba = CelebA(path=FLAGS.db_path)
X = celeba.preprocess_dataset(n_jobs=FLAGS.n_jobs)