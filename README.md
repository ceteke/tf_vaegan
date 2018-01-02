# Tensorflow VAE/GAN
This is an implementation of [this](https://arxiv.org/pdf/1512.09300.pdf) paper.  
Theory behind it: [VAE/GAN Part 1](https://ceteke.github.io/vae-gan-p1/)  
Some results: [VAE/GAN Part 2](https://ceteke.github.io/vae-gan-p2/)  
### How to use this code?
#### Requirements
* Tensorflow: 1.4
* Python 3.5
* Joblib

If you want to use it with [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) download the "img_align_celeba.zip" file.  
After downloading unzip it.  
Before training you need to preprocess the data. Run  
```bash
python --db [dataset path] --n_jobs [num of jobs]
```
This will create a file "celeba" under the project directory. Now we are ready for training.  
```bash
python celeba.py --arch celeba_architecture.json --tb_id [tensorboard id] --n_jobs [number of jobs] --device [device id]
```
**tb_id** is for saving the model and logs with the given id.  
**n_jobs** is number of jobs to read the data in parallel.  
**device** is the device id to run the model.  
That's it. To see the progress run
```bash
tensorboard --logdir=log
```
You can play with ```decay```, ```lr``` and ```gamma``` parameters in the json file.
