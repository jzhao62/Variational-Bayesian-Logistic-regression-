from scipy import misc
import glob
import os
import gzip
import _pickle as cPickle
import cv2 as cv
import numpy as np
import pickle


def resize_and_scale(img, size, scale):
    img = cv.resize(img, size)
    return 1 - np.array(img, "float32")/scale
def reformat(labels):
    num_labels = 10
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return labels
def import_MNIST():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    save = cPickle.load(f, encoding='latin1')
    train_dataset = save[0][0]
    train_labels = reformat(save[0][1])
    raw_train_labels = save[0][1]

    validation_data_input = save[1][0]
    valid_labels = reformat(save[1][1])
    raw_valid_labels = save[1][1]

    test_dataset = save[2][0]
    test_labels = reformat(save[2][1])
    raw_test_labels = save[2][1]
    print(raw_train_labels.shape)
    print('Training set', train_dataset.shape, 'Training label',train_labels.shape)
    print(raw_valid_labels.shape)
    print('Validation set', validation_data_input.shape,'val label', valid_labels.shape)
    print(raw_test_labels.shape)
    print('Test set', test_dataset.shape, 'test label', test_labels.shape)

    f.close()

    return train_dataset, train_labels, raw_train_labels, validation_data_input, valid_labels, raw_valid_labels, test_dataset, test_labels, raw_test_labels


def load_usps(pkl_image, pkl_label):
    pkl1 = open('data/usps_test_image.pkl', 'rb')
    pkl2 = open('data/usps_test_label.pkl', 'rb')
    usps_image = pickle.load(pkl1)
    usps_label = pickle.load(pkl2)

    pkl1.close()
    pkl2.close()
    return usps_image, usps_label