import numpy as np
import tensorflow as tf
import os
import gzip


def get_data(inputs_file_path, labels_file_path, num_examples):
    """
    Takes in an inputs file path and labels file path, unzips both files,
    normalizes the inputs, and returns (NumPy array of inputs, NumPy
    array of labels). Read the data of the file into a buffer and use
    np.frombuffer to turn the data into a NumPy array. Keep in mind that
    each file has a header of a certain size. This method should be called
    within the main function of the assignment.py file to get BOTH the train and
    test data. If you change this method and/or write up separate methods for
    both train and test data, we will deduct points.
    :param inputs_file_path: file path for inputs, something like
    'MNIST_data/t10k-images-idx3-ubyte.gz'
    :param labels_file_path: file path for labels, something like
    'MNIST_data/t10k-labels-idx1-ubyte.gz'
    :param num_examples: used to read from the bytestream into a buffer. Rather
    than hardcoding a number to read from the bytestream, keep in mind that each image
    (example) is 28 * 28, with a header of a certain number.
    :return: NumPy array of inputs as float32 and labels as int8
    """
    inputs_data = np.zeros((num_examples, 784))
    with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        temp_header = bytestream.read(16)
        for i in range(0,num_examples):
            inputs_data[i,:] = (1.0/255.0)*np.frombuffer(bytestream.read(784), dtype = np.uint8)
    inputs_data = np.array(inputs_data, dtype = np.float32)
    
    #Read in labels     
    inputs_label = np.zeros(num_examples)
    with open(labels_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        temp_header = bytestream.read(8)
        for i in range(0,num_examples):
            inputs_label[i] = np.frombuffer(bytestream.read(1), dtype = np.uint8)
    inputs_label = np.array(inputs_label, dtype = np.uint8)
    return inputs_data, inputs_label



# This code is adopted from xxxxxxxxxxxx
# Sets up tensorflow graph to load images
# (This is the version using new-style tf.data API)
def load_image_batch(dir_name, batch_size=128, shuffle_buffer_size=250000, n_threads=2):
    """
    Given a directory and a batch size, the following method returns a dataset iterator that can be queried for 
    a batch of images

    :param dir_name: a batch of images
    :param batch_size: the batch size of images that will be trained on each time
    :param shuffle_buffer_size: representing the number of elements from this dataset from which the new dataset will 
    sample
    :param n_thread: the number of threads that will be used to fetch the data

    :return: an iterator into the dataset
    """
    # Function used to load and pre-process image files
    # (Have to define this ahead of time b/c Python does allow multi-line
    #    lambdas, *grumble*)
    def load_and_process_image(file_path):
        """
        Given a file path, this function opens and decodes the image stored in the file.

        :param file_path: a batch of images

        :return: an rgb image
        """
        # Load image
        image = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3)
        # Convert image to normalized float (0, 1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Rescale data to range (-1, 1)
        image = (image - 0.5) * 2
        return image

    # List file names/file paths
    dir_path = dir_name + '/*.jpg'
    dataset = tf.data.Dataset.list_files(dir_path)

    # Shuffle order
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Load and process images (in parallel)
    dataset = dataset.map(map_func=load_and_process_image, num_parallel_calls=n_threads)

    # Create batch, dropping the final one which has less than batch_size elements and finally set to reshuffle
    # the dataset at the end of each iteration
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch the next batch while the GPU is training
    dataset = dataset.prefetch(1)

    # Return an iterator over this dataset
    return dataset
