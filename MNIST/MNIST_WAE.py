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
        """
        """
    def call(self, inputs):
        """
        pass inputs
        """
        for i in range(len(self.hidden_sizes)):
            inputs = tf.nn.relu(self.NN[i](inputs))
        means = self.outer_mean(inputs)
        return means









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


class WAE(tf.keras.Model):
    """
    whole model
    u is the mean for priors, normally it's 0
    v is the covairance for prior, normally is I
    lamb is the weight parameter for MMD loss
    """
    def __init__(self, output_size, hidden_sizes_encoder, hidden_sizes_decoder, z_dim, u, v, lamb):
        super(WAE, self).__init__()
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
        means = self.encoder(inputs)
        eps = tf.random.normal((means.shape[0],self.z_dim), mean = tf.zeros(self.z_dim), stddev = tf.ones(self.z_dim))
        logits, outputs = self.decoder(means)
        return means, eps, logits, outputs
        """
        """
    def imq_kernel(self, sample_qz, sample_pz):
        """
        this code is adopted from https://github.com/tolstikhin/wae/
        """
        sigma2_p = 1.0
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
    def loss(self,sample_qz, sample_pz, inputs, logits):
        #reconstruaction loss
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=inputs)
        logpxz = -tf.reduce_sum(cross_ent, axis = 1)
        rec = -tf.reduce_mean(logpxz)
        #mmd loss
        mmd = self.imq_kernel(sample_qz, sample_pz)
        #total loss
        loss = rec + self.lamb*mmd
        return rec, mmd, loss


def train_wae(wae, train_inputs, batch_size):
    num_examples = train_inputs.shape[0]
    nbatch = int(round(num_examples/batch_size))
    for i in range(0, nbatch):
        temp_id = batch_size*i + np.array(range(batch_size))
        temp_inputs = train_inputs[np.min(temp_id):(np.max(temp_id)+1), :]
        with tf.GradientTape() as tape:
            means, eps, logits, outputs = wae.call(temp_inputs)
            rec, mmd, loss = wae.loss(means, eps, temp_inputs, logits)
        gradients = tape.gradient(loss, wae.trainable_variables)
        wae.optimizer.apply_gradients(zip(gradients, wae.trainable_variables))
        print('loss', rec, mmd, loss)

