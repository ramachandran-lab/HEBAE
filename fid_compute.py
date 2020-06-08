#!/usr/bin/env python3
#This code is adopted and modified from https://github.com/bioinf-jku/TTUR
from __future__ import absolute_import, division, print_function
import os
import glob
import numpy as np
import fid
import tensorflow.compat.v1 as tf
from PIL import Image
import sys

#specify image path
image_path = sys.argv[1]
# print(image_path)

def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]

inception_path = fid.check_or_download_inception(None) # download inception network
image_list = glob.glob(os.path.join(image_path, '*.png'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

fid.create_inception_graph(inception_path)  # load the graph into the current TF graph

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)


#load pre_computed real data statistics
# mu_real = np.loadtxt('mu_real.txt')
# sigma_real = np.loadtxt('sigma_real.txt')

fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
print("FID: %s" % fid_value)
