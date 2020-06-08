from preprocess import load_image_batch
from CelebA_WAE import *
from math import *
import tensorflow_hub as hub
import tensorflow_gan as tfgan
from imageio import imwrite
import os
import argparse
import sys
import pickle

# lamb = sys.argv[1]
# lamb = float(lamb)
lamb = 0.001



def train(wae, dataset_iterator, epoch):
    for iteration, batch in enumerate(dataset_iterator):
        with tf.GradientTape() as tape:
            means, eps, outputs = wae.call(batch)
            loss = wae.loss(means, eps, batch, outputs)
        gradients = tape.gradient(loss, wae.trainable_variables)
        wae.optimizer.apply_gradients(zip(gradients, wae.trainable_variables))
        print(iteration, 'loss', loss)
        #save model
        # if iteration % 1000 == 0:
        #     with open('loss_0505/loss_wae_'+str(wae.lamb)+'_'+str(epoch)+'_'+str(iteration), 'wb') as fp:
        #         pickle.dump(all_loss,fp)
        #     wae.save_weights(filepath = 'trained_models/wae_'+str(wae.lamb)+'_'+str(epoch)+'_'+str(iteration)+'.h5')
        #     print('**** LOSS: %g ****' % loss)



#load in data
dataset_iterator = load_image_batch('celebA', batch_size=100, n_threads=2)

#defime z_dim
z_dim = 64

#batch size
batch_size= 100
u = tf.convert_to_tensor(np.zeros(z_dim), dtype = 'float32')
z_dim = 64
temp = np.zeros((z_dim, z_dim))
for i in range(z_dim):
    temp[i,i] = 1.0

v = tf.convert_to_tensor(temp, dtype = 'float32')
#lamb = 0.001WW
wae = WAE(z_dim, u, v, lamb)

nepoch = 100
for epoch in range(70):
    if epoch == 30:
        wae.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001*0.5, beta_1=0.5, beta_2 = 0.999)
    if epoch == 50:
        wae.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001*0.5*0.2, beta_1=0.5, beta_2 = 0.999)
    if epoch == 70:
        wae.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001*0.5*0.2*0.1, beta_1=0.9, beta_2 = 0.999)
    print('========================== EPOCH %d  ==========================' % epoch)
    train(wae, dataset_iterator, epoch)












