from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data
import tensorflow_probability as tfp
import tensorflow as tf

class Encoder(tf.keras.Model):
    """
    encoder model
    """
    def __init__(self, hidden_sizes, z_dim):
        super(Encoder, self).__init__()
        # hidden size stores the number of nodes for each dense layer
        self.hidden_sizes = hidden_sizes
        self.z_dim = z_dim
        #  Initialize architectures
        self.NN = list()
        for i in range(len(self.hidden_sizes)):
            self.NN.append(tf.keras.layers.Dense(self.hidden_sizes[i]))
        self.outer_mean = tf.keras.layers.Dense(self.z_dim)
        self.outer_var = tf.keras.layers.Dense(self.z_dim)
        """
        """
    def call(self, inputs):
        """
        pass inputs
        """
        for i in range(len(self.hidden_sizes)):
            inputs = tf.nn.relu(self.NN[i](inputs))
        means = self.outer_mean(inputs)
        logvar = self.outer_var(inputs)
        return means, logvar









class Decoder(tf.keras.Model):
    """
    decoder model
    """
    def __init__(self, output_size, hidden_sizes):
        super(Decoder, self).__init__()
        #  output_size = 784, hidden size stores the number of nodes for each dense layer
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        #  Initialize architectures
        self.NN = list()
        for i in range(len(self.hidden_sizes)):
            self.NN.append(tf.keras.layers.Dense(self.hidden_sizes[i]))
        self.outer = tf.keras.layers.Dense(self.output_size)
        """
        """
    def call(self, inputs):
        """
        pass inputs
        """
        for i in range(len(self.hidden_sizes)):
            inputs = tf.nn.relu(self.NN[i](inputs))
        logits = self.outer(inputs)
        outputs = tf.nn.sigmoid(logits)
        return logits, outputs
        """
        """


class VAE(tf.keras.Model):
    """
    whole model
    u is the mean for priors, normally it's 0
    v is the covairance for prior, normally is I
    lamb is the weight parameter for KL loss
    """
    def __init__(self, output_size, hidden_sizes_encoder, hidden_sizes_decoder, z_dim, u, v, lamb):
        super(VAE, self).__init__()
        self.u = u
        self.v = v
        self.lamb = lamb
        self.output_size = output_size
        self.z_dim = z_dim
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.encoder = Encoder(hidden_sizes_encoder, self.z_dim)
        self.decoder = Decoder(self.output_size, hidden_sizes_decoder)
        """
        """
    def call(self, inputs):
        """
        pass inputs
        """
        means, logvar = self.encoder(inputs)
        eps = tf.random.normal(means.shape, mean=0.0, stddev = 1.0)
        z = means+eps*tf.math.sqrt(tf.math.exp(logvar))
        logits, outputs = self.decoder(z)
        return means, logvar, z, logits, outputs
        """
        """

    def loss(self, means, logvar, inputs, logits, outputs, z):
        #recontrauction loss
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=inputs)
        logpxz = -tf.reduce_sum(cross_ent, axis = 1)
        rec = -tf.reduce_mean(logpxz)
        #kl loss
        kl = -0.5*tf.reduce_sum(logvar+1-tf.math.square(means)-tf.math.exp(logvar))/z.shape[0]
        #total loss
        loss = rec + self.lamb*kl
        return rec, kl, loss


def train_vae(vae, train_inputs, batch_size):
    num_examples = train_inputs.shape[0]
    nbatch = int(round(num_examples/batch_size))
    for i in range(0, nbatch):
        temp_id = batch_size*i + np.array(range(batch_size))
        temp_inputs = train_inputs[np.min(temp_id):(np.max(temp_id)+1), :]
        with tf.GradientTape() as tape:
            means, logvar, z, logits, outputs = vae.call(temp_inputs)
            rec, kl, loss = vae.loss(means, logvar, temp_inputs, logits, outputs, z)
        gradients = tape.gradient(loss, vae.trainable_variables)
        vae.optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        print('loss', loss)


