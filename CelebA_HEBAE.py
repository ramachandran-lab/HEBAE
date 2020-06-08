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
        # initialize all 
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
        mus = self.d6(inputs)
        sigma = self.d7(inputs)
        return mus, sigma









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






class HEBAE(tf.keras.Model):
    """
    whole model
    """
    def __init__(self,z_dim, u, v, ind, lamb):
        super(HEBAE, self).__init__()
        self.z_dim = z_dim
        #prior mean, normally it's just 0
        self.u = u
        #prior covariance, normally it's I
        self.v = v
        #sampling indicator, in variational inference framework, this should be 1
        self.ind = ind
        #lambda weight for KL divergvence
        self.lamb = lamb
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1 = 0.9, beta_2 = 0.999)
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder()
        """
        """
    def call(self, inputs):
        """
        pass inputs
        """
        mus, sigma = self.encoder(inputs)
        if self.ind == 1:
            #covariance matrix Sigma
            cov_mus = tfp.stats.covariance(mus, sample_axis = 0, event_axis = 1)
            #sample noise
            eps = tf.random.normal((mus.shape[0],self.z_dim), mean = tf.zeros(self.z_dim), stddev = tf.ones(self.z_dim))
            #encoder will output log(sigma), here take the exp to get sigma
            dis = tf.math.exp(sigma)
            #approxiamted reparameterization
            z = mus + eps*np.sqrt(np.diag(cov_mus))*dis
            #the following code is for reparameterization trick incoporating covariances, however, as cov_mus will converge to I, we found the approximated reparameterization won't make much difference and will save some computation time
            #r = tf.linalg.cholesky(cov_mus)
            #z = mus + tf.linalg.matmul(eps, r)*dis
            outputs = self.decoder(z)
        else:
            outputs = self.decoder(mus)
        return mus, outputs
        """
        """
    def loss(self, mus, inputs, outputs):
        cov_mus = tfp.stats.covariance(mus, sample_axis = 0, event_axis = 1)
        #kl without constraint
        loss_encoder = (0.5*(tf.linalg.trace(cov_mus) - mus.shape[1] - tf.linalg.logdet(cov_mus)))
        #constraint
        mean_mus = tf.reduce_mean(mus, axis = 0)
        #loss_encoder is equal to kl of mu with N(0,I) = kl + constraint
        loss_encoder += 0.5*tf.math.reduce_sum(tf.math.square(mean_mus))
        #reconstruction mse loss
        loss_decoder = tf.reduce_mean(tf.square(outputs - inputs))
        #loss_decoder = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_mus(mus=decoder_mus, labels=inputs), axis = 1))
        loss = self.lamb*loss_encoder+loss_decoder
        return self.lamb*loss_encoder, loss_decoder, loss




