from __future__ import absolute_import
import numpy as np
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
        self.outer1 = tf.keras.layers.Dense(self.z_dim)
        self.outer2 = tf.keras.layers.Dense(self.z_dim)
        """
        """
    def call(self, inputs):
        """
        pass inputs
        """
        for i in range(len(self.hidden_sizes)):
            inputs = tf.nn.relu(self.NN[i](inputs))
        mus = self.outer1(inputs)
        sigma = self.outer2(inputs)
        return mus, sigma









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






class HEBAE(tf.keras.Model):
    """
    whole model
    u is the mean for priors, normally it's 0
    v is the covairance for prior, normally is I
    ind is the indicator for sampling. If it's 1, it's for varitional inference
    lamb is the weight parameter for KL loss
    """
    def __init__(self,input_size, output_size, hidden_sizes_encoder, hidden_sizes_decoder, z_dim, u, v, ind, lamb): 
        super(HEBAE, self).__init__()
        self.z_dim = z_dim
        self.u = u
        self.v = v
        self.ind = ind
        self.lamb = lamb
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.encoder = Encoder(hidden_sizes_encoder, z_dim)
        self.decoder = Decoder(output_size, hidden_sizes_decoder)
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
            eps = np.random.normal(loc = np.zeros(self.z_dim), scale = np.ones(self.z_dim), size = (mus.shape[0],self.z_dim))
            #encoder will output log(sigma), here take the exp to get sigma
            dis = tf.math.exp(sigma)
            #approxiamted reparameterization
            z = mus + eps*np.sqrt(np.diag(cov_mus))*dis
            #the following code is for reparameterization trick incoporating covariances, however, as cov_mus will converge to I, we found the approximated reparameterization won't make much difference and will save some computation time
            #r = tf.linalg.cholesky(cov_mus)
            #z = mus + tf.linalg.matmul(eps, r)*dis
            decoder_logits, outputs = self.decoder(z)
        else:
            decoder_logits, outputs = self.decoder(mus)
        return mus, decoder_logits, outputs
        """
        """
    def loss(self, mus, inputs, decoder_logits, outputs):
        #covariance matrix Sigma
        cov_mus = tfp.stats.covariance(mus, sample_axis = 0, event_axis = 1)
        #kl loss
        kl = 0.5*(tf.linalg.trace(cov_mus) - mus.shape[1] - tf.linalg.logdet(cov_mus))
        #constraint
        mean_mus = tf.reduce_mean(mus, axis = 0)
        loss_con = 0.5*tf.math.reduce_sum(tf.math.square(mean_mus))
        #reconstruction loss
        rec = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=decoder_logits, labels=inputs), axis = 1))
        #total loss
        loss = self.lamb*(kl+loss_con)+rec
        return kl, loss_con, rec, loss

def train(hebae, train_inputs, batch_size):
    num_examples = train_inputs.shape[0]
    nbatch = int(round(num_examples/batch_size))
    for i in range(0, nbatch):
        temp_id = batch_size*i + np.array(range(batch_size))
        temp_inputs = train_inputs[np.min(temp_id):(np.max(temp_id)+1), :]
        with tf.GradientTape() as tape:
            mus, decoder_logits,outputs = hebae.call(temp_inputs)
            kl, loss_con, rec, loss = hebae.loss(mus, temp_inputs, decoder_logits, outputs)
        gradients = tape.gradient(loss, hebae.trainable_variables)
        hebae.optimizer.apply_gradients(zip(gradients, hebae.trainable_variables))
        print('loss', loss)
        










