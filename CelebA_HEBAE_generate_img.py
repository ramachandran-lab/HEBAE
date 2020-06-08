from preprocess import load_image_batch
from CelebA_HEBAE import *
from math import *
import os
import argparse
import sys
import pickle
from imageio import imwrite

#ind = sys.argv[1]
ind = 1
ind = int(ind)

# lamb = sys.argv[2]
lamb = 0.001
lamb = float(lamb)

#defein z_dim
z_dim = 64



all_loss = list()
u = tf.convert_to_tensor(np.zeros(z_dim), dtype = 'float32')
temp = np.zeros((z_dim, z_dim))
for i in range(z_dim):
    temp[i,i] = 1.0
v = tf.convert_to_tensor(temp, dtype = 'float32')

#intialize model
hebae = HEBAE(z_dim, u, v, ind, lamb)
batch = tf.random.normal((500,64,64,3), mean = 0.0, stddev = 10.0)
logits, outputs = hebae.call(batch)

#load in pretrained param
hebae.load_weights('pretrained_models/pretrained_hebae.h5')    
batch_size = 1000

#generate images
for j in range(10):
    print(j)
    temp_logits = np.random.normal(loc=0, scale = 1, size = (batch_size, z_dim))
    temp_logits = tf.convert_to_tensor(temp_logits, dtype='float32')
    img = hebae.decoder(temp_logits)
    img = ((img / 2) - 0.5) * 255
    img = np.asarray(img, dtype = 'uint8')
    for i in range(0, batch_size):
        img_i = img[i]
        s = 'test_img/'+ str(i+j*1000)+'.png'
        imwrite(s, img_i)


