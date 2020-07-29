from preprocess import load_image_batch
from CelebA_VAE import *
from math import *
import tensorflow_hub as hub
import tensorflow_gan as tfgan
from imageio import imwrite
import os
import argparse
import sys
import pickle


#lamb = sys.argv[1]
lamb = 0.001

#train vae
def train(vae, dataset_iterator, epoch):
    # all_loss = list()
    for iteration, batch in enumerate(dataset_iterator):
        with tf.GradientTape() as tape:
            means, logvar, outputs = vae.call(batch)
            rec, kl, loss = vae.loss(means, logvar, batch, outputs)
        gradients = tape.gradient(loss, vae.trainable_variables)
        vae.optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        print(iteration, 'loss', loss)
        # all_loss.append([rec, kl, loss])
        # #save trained model
        # if iteration % 1000 == 0:
        #     vae.save_weights(filepath = 'trained_models/vae_'+str(vae.lamb)+'_'+str(epoch)+'_'+str(iteration)+'.h5')
        #     print('**** LOSS: %g ****' % loss)



#load in data
dataset_iterator = load_image_batch('celebA', batch_size=100, n_threads=2)

#defime z_dim
z_dim = 64

#batch size
batch_size= 100
all_loss = list()
#prior mean u and v
u = tf.convert_to_tensor(np.zeros(z_dim), dtype = 'float32')
temp = np.zeros((z_dim, z_dim))
for i in range(z_dim):
    temp[i,i] = 1.0

v = tf.convert_to_tensor(temp, dtype = 'float32')
vae = VAE(z_dim, u, v, lamb)
nepoch = 70
for epoch in range(70):
    if epoch == 30:
        vae.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001*0.5, beta_1=0.9, beta_2 = 0.999)
    if epoch == 50:
        vae.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001*0.5*0.2, beta_1=0.9, beta_2 = 0.999)
    if epoch == 70:
        vae.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001*0.5*0.2*0.1, beta_1=0.9, beta_2 = 0.999)
    print('========================== EPOCH %d  ==========================' % epoch)
    train(vae, dataset_iterator, epoch)












