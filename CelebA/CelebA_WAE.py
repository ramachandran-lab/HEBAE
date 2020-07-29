from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, ReLU
class Encoder(tf.keras.Model):
    """
    encoder model
    """
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        #  z_dim 
        self.z_dim = z_dim
        #  Initialize all
        self.c1 = tf.keras.layers.Conv2D(128, (5,5), strides = (2,2), padding = "SAME", use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02))
        self.b1 = tf.keras.layers.BatchNormalization()
        self.r1 = tf.keras.layers.ReLU()
        self.c2 = tf.keras.layers.Conv2D(256, (5,5), strides = (2,2), padding = "SAME", use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02))
        self.b2 = tf.keras.layers.BatchNormalization()
        self.r2 = tf.keras.layers.ReLU()
        self.c3 = Conv2D(512, (5,5), strides = (2,2), padding = "SAME", use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02))
        self.b3 = tf.keras.layers.BatchNormalization()
        self.r3 = tf.keras.layers.ReLU()
        self.c4 = tf.keras.layers.Conv2D(1024, (5,5), strides = (2,2), padding = "SAME", use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02))
        self.b4 = tf.keras.layers.BatchNormalization()
        self.r4 = tf.keras.layers.ReLU()
        self.f5 = tf.keras.layers.Flatten()
        self.d6 = tf.keras.layers.Dense(self.z_dim)
        """
        """
    def call(self, inputs):
        """
        pass inputs
        """
        inputs = self.c1(inputs)
        inputs = self.b1(inputs)
        inputs = self.r1(inputs)
        inputs = self.c2(inputs)
        inputs = self.b2(inputs)
        inputs = self.r2(inputs)
        inputs = self.c3(inputs)
        inputs = self.b3(inputs)
        inputs = self.r3(inputs)
        inputs = self.c4(inputs)
        inputs = self.b4(inputs)
        inputs = self.r4(inputs)
        inputs = self.f5(inputs)
        means = self.d6(inputs)
        return means









class Decoder(tf.keras.Model):
    """
    decoder model
    """
    def __init__(self):
        super(Decoder, self).__init__()
        #  use sequantial model
        self.model = tf.keras.models.Sequential()
        self.model.add(Dense(8*8*1024, use_bias=False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))        
        self.model.add(BatchNormalization())
        self.model.add(ReLU()) 
        self.model.add(Reshape((8,8,1024)))
        self.model.add(Conv2DTranspose(512,(5, 5), strides = (2,2), padding = "SAME", use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))
        self.model.add(BatchNormalization())
        self.model.add(ReLU())
        self.model.add(Conv2DTranspose(256,(5, 5), strides = (2,2), padding = "SAME", use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))
        self.model.add(BatchNormalization())
        self.model.add(ReLU())
        self.model.add(Conv2DTranspose(128,(5, 5), strides = (2,2), padding = "SAME", use_bias = False, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))
        self.model.add(BatchNormalization())
        self.model.add(ReLU())
        self.model.add(Conv2DTranspose(3,(5, 5), strides = (1,1), padding = "SAME", use_bias = False, activation = tf.nn.tanh, kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.02)))
        self.model.compile()
        """
        """
    def call(self, inputs):
        """
        pass inputs
        """
        outputs = self.model(inputs)
        return outputs
        """
        """






class WAE(tf.keras.Model):
    """docstring for PAE"""
    def __init__(self,z_dim, u, v, lamb):
        super(WAE, self).__init__()
        self.z_dim = z_dim
        #prior mean, normally it's just 0
        self.u = u
        #prior covariance, normally it's I
        self.v = v
        #lambda weight for MMD
        self.lamb = lamb
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1 = 0.5, beta_2 = 0.999)
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder()
        """
        """
    def call(self, inputs):
        """
        pass inputs
        """
        means = self.encoder(inputs)
        eps = tf.random.normal((means.shape[0],self.z_dim), mean = tf.zeros(self.z_dim), stddev = tf.ones(self.z_dim))
        outputs = self.decoder(means)
        return means, eps, outputs
        """
        """
    def imq_kernel(self, sample_qz, sample_pz):
        """
        this code is adopted from https://github.com/tolstikhin/wae/
        """
        sigma2_p = 2.0
        n = sample_qz.shape[0]
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1)
        norms_pz = tf.reshape(norms_pz, (int(n),1))
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1)
        norms_qz = tf.reshape(norms_qz, (int(n),1))
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods
        Cbase = 2. *self.z_dim*sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
        return stat
        """
        """
    def loss(self, sample_qz, sample_pz, inputs, outputs):
        #reconstrauction loss
        rec = 0.05*tf.reduce_mean(tf.reduce_sum(tf.square(inputs-outputs), axis = [1,2,3]))
        #mmd loss
        mmd = self.imq_kernel(sample_qz, sample_pz)
        #total loss
        loss = rec + self.lamb*mmd
        return loss








