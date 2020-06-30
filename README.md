# Hierarchical Empirical Bayes Auto-Encoder (HEBAE)
This is the official implementation for Hierarchical Empirical Bayes Auto-Encoder (HEBAE). It contains the code for training the Hierarchical Empirical Bayes AutoEncoders(HEBAE)/VAE/WAE on CelebA and MNIST datasets.

# Dependencies:
Python >= 3.7.4,
tensorflow >= 2.1.0,
tensorflow-probability >= 0.9.0,
keras >= 2.3.1,
matplotlib >= 3.1.2,
numpy >= 1.17.2,
Pillow >= 7.1.0,
scikit-learn >= 0.21.3,
scipy >= 1.4.1

# Datasets:
MNIST dataset can be downloaded here:http://yann.lecun.com/exdb/mnist/

CelebA dataset can be downloaded here:http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (For the paper, we center crop and resize images to 64by64 resolution.)

# Notes
The names for code have the following formats:
1. *`[datasets]_[model].py` - specify the model architectures
2. *`[datasets]_[model]_train.py` - script to train the corresponding model

# Example Images
![alt text](all_samples_CelebA.pdf)

# Other Guidelines:
1. *`utils.py` contains helper function to load in data.
2. *`CelebA_[model]_generate_img.py` contains the code to generate images using a pre-trained model for the CelebA dataset. The code for generating images for MNIST are commented out in the *`MNIST_[model]_train.py` file. This model can be trained quickly.

