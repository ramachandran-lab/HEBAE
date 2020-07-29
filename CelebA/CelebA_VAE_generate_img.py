from __future__ import absolute_import, division, print_function
from preprocess import load_image_batch
from CelebA_VAE import *
from math import *
from imageio import imwrite
import os
import sys
import pickle
import numpy as np
from PIL import Image
# lamb = sys.argv[1]
# lamb = float(lamb)
lamb = 0.001
#initialize model
u = tf.convert_to_tensor(np.zeros(z_dim), dtype = 'float32')
z_dim = 64
temp = np.zeros((z_dim, z_dim))
for i in range(z_dim):
    temp[i,i] = 1.0
v = tf.convert_to_tensor(temp, dtype = 'float32')
vae = VAE(z_dim, u, v, lamb)
batch = tf.random.normal((100,64,64,3))
temp = vae.call(batch)
vae.load_weights('pretrained_models/pretrained_vae.h5')

#generate images
batch_size = 1000
for j in range(10):
    print(j)
    temp_logits = np.random.normal(loc=0, scale = 1, size = (batch_size, z_dim))
    temp_logits = tf.convert_to_tensor(temp_logits, dtype='float32')
    img = vae.decoder(temp_logits)
    img = ((img / 2) - 0.5) * 255
    img = np.asarray(img, dtype = 'uint8')
    for i in range(0, batch_size):
        img_i = img[i]
        s = 'test_img/vae_' + str(i+j*100)+'.png'
        imwrite(s, img_i)










