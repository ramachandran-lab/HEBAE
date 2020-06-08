from preprocess import load_image_batch
from CelebA_HEBAE import *
from math import *
import os
import argparse
import sys
import pickle
#
ind = 1
#
#lamb = sys.argv[1]
lamb = 0.001

#train hebae
def train(hebae, dataset_iterator, epoch, z_dim):
    # all_loss = list()
    for iteration, batch in enumerate(dataset_iterator):
        with tf.GradientTape() as tape:
            mus, outputs = hebae.call(batch)
            loss_encoder, loss_decoder, loss = hebae.loss(mus, batch, outputs)
        gradients = tape.gradient(loss, hebae.trainable_variables)
        hebae.optimizer.apply_gradients(zip(gradients, hebae.trainable_variables))
        print(iteration, 'loss', loss_encoder, loss_decoder, loss)
        # all_loss.append([loss_encoder, loss_decoder, loss])
        # save trained model
        # if iteration % 1000 == 0:
        #     hebae.save_weights(filepath = 'trained_models/hebae_'+ str(z_dim) + '_' +str(hebae.lamb)+'_'+str(ind)+'_'+str(epoch)+'_'+str(iteration)+ '_' +'.h5')
        #     print('**** LOSS: %g ****' % loss)


#load in data
dataset_iterator = load_image_batch('celebA', batch_size=100, n_threads=2)

#defime z_dim
z_dim = 64

#batch size
batch_size= 100
#prior mean u and v
u = tf.convert_to_tensor(np.zeros(z_dim), dtype = 'float32')
temp = np.zeros((z_dim, z_dim))
for i in range(z_dim):
    temp[i,i] = 1.0

v = tf.convert_to_tensor(temp, dtype = 'float32')
hebae = HEBAE(z_dim, u, v, ind, lamb)
#train for 100 epochs
nepoch = 100
for epoch in range(100):
    if epoch == 30:
       hebae.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001*0.5, beta_1=0.9, beta_2 = 0.999)
    if epoch == 50:
       hebae.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001*0.5*0.2, beta_1=0.9, beta_2 = 0.999)
    if epoch == 70:
       hebae.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001*0.5*0.2*0.1, beta_1=0.9, beta_2 = 0.999)
    print('========================== EPOCH %d  ==========================' % epoch)
    train(hebae, dataset_iterator, epoch, z_dim)
    










