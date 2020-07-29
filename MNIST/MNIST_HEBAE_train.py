from MNIST_HEBAE import *
from utils import *
from math import *
import sys
from scipy.stats import norm
from matplotlib import pyplot as plt

# load in data
train_inputs, train_labels = get_data('MNIST_data/train-images-idx3-ubyte.gz', 'MNIST_data/train-labels-idx1-ubyte.gz', 60000)
test_inputs, test_labels = get_data('MNIST_data/t10k-images-idx3-ubyte.gz', 'MNIST_data/t10k-labels-idx1-ubyte.gz', 10000)
#get black background images
# train_inputs=-1*train_inputs+1
# test_inputs=-1*test_inputs+1



#define hyper-parameters
# z_dim = sys.argv[1]
# ind = sys.argv[2]
# lamb = sys.argv[3]
# z_dim = int(z_dim)
# ind = int(ind)
# w = float(w)
z_dim = 10
ind = 1
lamb = 1.0
batch_size = 128
nepoch = 100
u = tf.convert_to_tensor(np.zeros(z_dim), dtype = 'float32')
temp = np.zeros((z_dim, z_dim))
for i in range(z_dim):
    temp[i,i] = 1.0
v = tf.convert_to_tensor(temp, dtype = 'float32')
hebae = HEBAE(784, 784, [784,800],[800,800], z_dim, u, v, ind, lamb)
for epoch in range(nepoch):
    hebae.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001*0.995**epoch)
    all_indices = range(train_inputs.shape[0])
    shuffled_indices = tf.random.shuffle(all_indices)
    shuffled_inputs = tf.gather(train_inputs, shuffled_indices)
    shuffled_labels = tf.gather(train_labels, shuffled_indices)
    train(hebae, shuffled_inputs, batch_size)


#save model
# hebae.save_weights(filepath = 'trained_models/MNIST_'+str(z_dim)+'hebae.h5')


#random samples
#temp_z = np.random.normal(loc=0, scale = 1, size = (100, z_dim))
#temp_z = tf.convert_to_tensor(temp_z)
#outputs_logits,output_images = hebae.decoder(temp_z)
#images_sample = np.reshape(output_images, (-1, 28, 28))
#fig, axs = plt.subplots(nrows=10, ncols=10)
#for i in range(10):
#    for x, ax in enumerate(axs[i]):
#        ax.imshow(images_sample[x+i*10], cmap="Greys")
#        plt.setp(ax.get_xticklabels(), visible=False)
#        plt.setp(ax.get_yticklabels(), visible=False)
#        ax.tick_params(axis='both', which='both', length=0)
#
#plt.savefig('test_img/hebae_random_samples_'+str(z_dim)+'.png', dpi=500)

