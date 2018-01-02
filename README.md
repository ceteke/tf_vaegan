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
### Your architecture
If you want to use your own architecture with a custom dataset. You have to write your own data processing and reading code. But you can easyly use the model by providing a json file.
#### JSON Format
At the highest levet you have to include following
```json
{"lr": "learning rate",
"decay": "decay rate of learning rate, give 0 for no decay",
"optim": "optimizer to use. Possible options are adam, sgd and rmsprop",
"gamma": "weight of reconstruction error of the decoder"
}
```
Then you have to define 3 keys: ```enc```, ```dec``` and ```dis```. These will be the architectures of the encoder, decoder and discriminator respectively. All of them has to include a key ```net``` which has a vlaue of array. This array's elements will be dictionaries containing the info about the layers. ```enc``` has to have a key ```z_dim``` which is the dimension of the latent representation. ```dis``` has to have a ```feature_layer``` key that denotes which layer to use for feature reconstruction.  
#### Layers
Each layer has a ```type``` key. Those can be: ```flatten```, ```fc```, ```conv```,  ```conv_t``` and ```reshape```.  
```bnorm```: This denotes Batch Normalization. If it is 1 BN is used if 0 BN is not used. This has to be provided for layers that has parameters.  
```act```: This the activation. ```relu```, ```sigmoid``` and ```tanh``` can be used. Again this has to be provided for layers that has parameters.  
For an example please see ```celeba_architecture.json```.  
After you have developed your own architecture in order to use it in the model:  
```python
from models.vaegan import VAEGAN
import json
arch_json = json.load(open(<your_json>))
model = VAEGAN(<input_size>, arch_json, total_steps, tb_id)
model.compile()
...
enc_loss, dec_loss, dis_loss = model.fit(x)
```
```input_size```: [batch_size, image_height, image_width, channel_size]  
```total_steps```: Number of steps to perform learning rate decayin.
