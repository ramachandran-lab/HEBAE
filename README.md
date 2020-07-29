# Hierarchical Empirical Bayes Auto-Encoder (HEBAE)
This is the official implementation for the Hierarchical Empirical Bayes Auto-Encoder (HEBAE). The repository also contains code for training and comparing HEBAE to the Variational Autoencoder (VAE), and the Wassertein Autoencoder (WAE) on the CelebA and MNIST datasets.

## Dependencies:
Python >= 3.7.4;
tensorflow >= 2.1.0;
tensorflow-probability >= 0.9.0;
keras >= 2.3.1;
matplotlib >= 3.1.2;
numpy >= 1.17.2;
Pillow >= 7.1.0;
scikit-learn >= 0.21.3;
scipy >= 1.4.1

## Datasets:
The MNIST dataset can be downloaded here: http://yann.lecun.com/exdb/mnist/.

The CelebA dataset can be downloaded here: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. Note that for the paper cited at the bottom, we center, crop, and resize images to have a 64-by-64 resolution.

## For Usage
The names of the files have the following formats:
1. `[datasets]_[model].py` - specifies the architecture of the model;
2. `[datasets]_[model]_train.py` - script to train the corresponding model.

Please check the `[datasets]_[model]_train.py` scripts to see specifc details for each model. For both sets of analyses, all hyper-parameter settings (e.g., latent dimension size, batch size, number of training epochs, etc) can be easily changed. The default numbers are set to the values used in the paper. Note that in the MNIST dataset, the size and number of hidden layers can be changed without compiling additional errors. However, for analyses with the CelebA dataset, one will need to manually change the architecture in `CelebA_[model].py` to ensure that the convolutional layers are compatible with each other. 

## Other Guidelines
1. `utils.py` contains helper function to load in data.
2. `fid.py` and `fid_compute.py` contain code for computing the Frechet Inception Distance (FID) to evaluate images. 
2. `CelebA_[model]_generate_img.py` contains the code to generate images using a pre-trained model for the CelebA dataset. 
3. The code for generating images for MNIST are commented out in the `MNIST_[model]_train.py` file. This model can be trained quickly.

## Example Images
![alt text](CelebA_images.png)

## Relevant Citations:
W. Cheng, G. Darnell, S. Ramachandran, L. Crawford (2020). Generalizing Variational Autoencoders with Hierarchical Empirical Bayes. _arXiv_. 2007.10389
