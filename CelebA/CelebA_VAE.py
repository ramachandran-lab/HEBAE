from __future__ import absolute_import
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
        # z_dim
        self.z_dim = z_dim
        # Initialize weights and biases
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
        self.d7 = tf.keras.layers.Dense(self.z_dim)
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
        logvar = self.d7(inputs)
        return means, logvar









class Decoder(tf.keras.Model):
    """
    decoder model
    """
    def __init__(self):
        super(Decoder, self).__init__()
        #  use sequential model
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






class VAE(tf.keras.Model):
    """
    whole model
    """
    def __init__(self,z_dim, u, v, lamb):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        #prior mean, normally it's just 0
        self.u = u
        #prior covariance, normally it's I
        self.v = v
        #lambda weight for KL divergvence
        self.lamb = lamb
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1 = 0.9, beta_2 = 0.999)
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder()
        """
        """
    def call(self, inputs):
        """
        pass inputs with reparameterization trick
        """
        means, logvar = self.encoder(inputs)
        eps = tf.random.normal(means.shape, mean=0.0, stddev = 1.0)
        z = means+eps*tf.math.sqrt(tf.math.exp(logvar))
        outputs = self.decoder(z)
        return means, logvar, outputs
        """
        """
    def loss(self, means, logvar, inputs,outputs):
        #reconstruction loss
        rec = tf.reduce_mean(tf.square(outputs - inputs))
        #kl loss
        kl = -0.5*tf.reduce_sum(logvar+1-tf.math.square(means)-tf.math.exp(logvar))/means.shape[0]
        #total loss
        negelbo = rec + self.lamb*kl
        return rec, kl, negelbo








