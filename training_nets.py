'''
By: Joel Devey

IMAGE ANN:
    1024 x 50 x 2
    epochs = 20
    eta = 0.01
    lambda = 1
    Notes:
        Uses the SGD function provided previously. Approximately 91% accuracy.

IMAGE CNN:
    32 x 32 input
    convolutional layer 1:
        filters = 16
        filter size = 5
        pooling factor = 2
    convoltional layer 2:
        filters = 32
        filter size = 5
        pooling factor = 2
    fully connected layer 1:
        activation = tanh
    fully connected layer 2:
        activation = softmax
    epochs = 30
    eta = 0.1
    Notes:
        Approximately 99% accuracy

AUDIO ANN:
    sixty 1000 x 50 x 2 networks, each with same parameters
    epochs = 7
    eta = 0.1
    lambda = 1
    Notes:
        Approximately 78% accuracy for data in the training directory, and 55%
        for data in the testing directory. In both cases, the network hasn't
        seen the data before.

AUDIO CNN:
    10078 input
    convolutional layer 1:
        filters = 8
        filter size = 79
        pooling factor = 100
    convolutional layer 2:
        filters = 16
        filter size = 11
        pooling factor = 5
    fully connected layer 1:
        activation = tanh
    fully connected layer 2:
        activation = softmax
    epochs = 30
    eta = 0.1
    Notes: Approximately 90% accuracy

'''

import cv2
import glob
import pickle as cPickle
import numpy as np
import scipy as sp
from scipy.io import wavfile

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, conv_1d, max_pool_1d
from tflearn.layers.estimator import regression

from helper import *

MAX_AUDIO = 60000
AUDIO_INC = 1000
MAX_CONV = 10078

bee_train_d = []
bee_test_d = []
bee_train_d_conv = [np.array([]), np.array([])]
bee_test_d_conv = [np.array([]), np.array([])]
buzz_train_d = []
buzz_test_d = []
buzz_train_d_conv = [[], []]
buzz_test_d_conv = [[], []]

def pickle_object(obj, filename):
    with open(filename, 'wb') as fp:
        cPickle.dump(obj, fp)

def unpickle_object(filename):
    with open(filename, 'rb') as fp:
        ff = cPickle.load(fp)
        return ff

def persist_ann_bee_data():
    pickle_object(bee_train_d, 'beedata/bee_train_d.pck')
    pickle_object(bee_test_d, 'beedata/bee_test_d.pck')

def retrieve_ann_bee_data():
    global bee_train_d, bee_test_d
    bee_train_d = unpickle_object('beedata/bee_train_d.pck')
    bee_test_d = unpickle_object('beedata/bee_test_d.pck')

def persist_ann_buzz_data():
    pickle_object(buzz_train_d, 'beedata/buzz_train_d.pck')
    pickle_object(buzz_test_d, 'beedata/buzz_test_d.pck')

def retrieve_ann_buzz_data():
    buzz_train_d = unpickle_object('beedata/buzz_train_d.pck')
    buzz_test_d = unpickle_object('beedata/buzz_test_d.pck')

def persist_conv_bee_data():
    pickle_object(bee_train_d_conv, 'beedata/bee_train_d_conv.pck')
    pickle_object(bee_test_d_conv, 'beedata/bee_test_d_conv.pck')

def retrieve_conv_bee_data():
    bee_train_d_conv = unpickle_object('beedata/bee_train_d_conv.pck')
    bee_test_d_conv = unpickle_object('beedata/bee_test_d_conv.pck')

def persist_conv_buzz_data():
    pickle_object(buzz_train_d_conv, 'beedata/buzz_train_d_conv.pck')
    pickle_object(buzz_test_d_conv, 'beedata/buzz_test_d_conv.pck')

def retrieve_conv_buzz_data():
    buzz_train_d_conv = unpickle_object('beedata/buzz_train_d_conv.pck')
    buzz_test_d_conv = unpickle_object('beedata/buzz_test_d_conv.pck')

def make_image(filename):
    img = cv2.imread(filename)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image / 255.0

def make_audio(filename):
    samplerate, audio = wavfile.read(filename)
    return audio / float(np.max(audio))

def append_to_dat(makefunc, dat, fp, y, limit, makesquare=False):
    i = 0
    for filename in glob.iglob(fp, recursive=True):
        dat.append((np.array([[i] for i in (makefunc(filename).flatten())]), y))
        i += 1
        print(i)
        if (i >= limit): break
    np.random.shuffle(dat)

def append_to_dat_square(makefunc, dat, fp, y, limit):
    i = 0
    xx = []
    yy = []
    for filename in glob.iglob(fp, recursive=True):
        xx.append(makefunc(filename))
        yy.append(y)
        i += 1
        print(i)
        if (i >= limit): break
    dat[0] = np.append(dat[0], np.array(xx)).reshape([-1, 32, 32, 1])
    dat[1] = np.append(dat[1], np.array(yy)).reshape([-1, 2])

def append_to_dat_long(makefunc, dat, fp, y, limit):
    i = 0
    xx = []
    for filename in glob.iglob(fp, recursive=True):
        xx.append(makefunc(filename)[:MAX_CONV].reshape([-1, 1]))
        i += 1
        print(i)
        if (i >= limit):
            break
    for i in xx:
        dat[0].append(i)
    for i in range(limit):
        dat[1].append(y)

def make_bee_training_data():
    append_to_dat(make_image, bee_train_d, 'BEE2Set/bee_train/**/*.png', np.array([[1], [0]]), 5000)
    append_to_dat(make_image, bee_train_d, 'BEE2Set/no_bee_train/**/*.png', np.array([[0], [1]]), 5000)
    append_to_dat(make_image, bee_train_d, 'BEE2Set/bee_test/**/*.png', np.array([[1], [0]]), 5000)
    append_to_dat(make_image, bee_train_d, 'BEE2Set/no_bee_test/**/*.png', np.array([[0], [1]]), 5000)
    global bee_test_d
    bee_test_d = bee_train_d

def make_bee_conv_training_data():
    append_to_dat_square(make_image, bee_train_d_conv, 'BEE2Set/bee_train/**/*.png', np.array([1, 0]), 5000)
    append_to_dat_square(make_image, bee_train_d_conv, 'BEE2Set/no_bee_train/**/*.png', np.array([0, 1]), 5000)
    append_to_dat_square(make_image, bee_train_d_conv, 'BEE2Set/bee_test/**/*.png', np.array([1, 0]), 5000)
    append_to_dat_square(make_image, bee_train_d_conv, 'BEE2Set/no_bee_test/**/*.png', np.array([0, 1]), 5000)
    global bee_test_d_conv
    bee_test_d_conv = bee_train_d

def make_buzz_training_data():
    append_to_dat(make_audio, buzz_train_d, 'BUZZ2Set/train/bee_train/**/*.wav', np.array([[1], [0], [0]]), 100)
    append_to_dat(make_audio, buzz_train_d, 'BUZZ2Set/train/cricket_train/**/*.wav', np.array([[0], [1], [0]]), 100)
    append_to_dat(make_audio, buzz_train_d, 'BUZZ2Set/train/noise_train/**/*.wav', np.array([[0], [0], [1]]), 100)
    append_to_dat(make_audio, buzz_train_d, 'BUZZ2Set/test/bee_test/**/*.wav', np.array([[1], [0], [0]]), 100)
    append_to_dat(make_audio, buzz_train_d, 'BUZZ2Set/test/cricket_test/**/*.wav', np.array([[0], [1], [0]]), 100)
    append_to_dat(make_audio, buzz_train_d, 'BUZZ2Set/test/noise_test/**/*.wav', np.array([[0], [0], [1]]), 100)
    global buzz_test_d
    buzz_test_d = buzz_train_d

def make_buzz_conv_training_data():
    append_to_dat_long(make_audio, buzz_train_d_conv, 'BUZZ2Set/train/bee_train/**/*.wav', np.array([1, 0, 0]), 300)
    append_to_dat_long(make_audio, buzz_train_d_conv, 'BUZZ2Set/train/cricket_train/**/*.wav', np.array([0, 1, 0]), 300)
    append_to_dat_long(make_audio, buzz_train_d_conv, 'BUZZ2Set/train/noise_train/**/*.wav', np.array([0, 0, 1]), 300)
    append_to_dat_long(make_audio, buzz_train_d_conv, 'BUZZ2Set/test/bee_test/**/*.wav', np.array([1, 0, 0]), 300)
    append_to_dat_long(make_audio, buzz_train_d_conv, 'BUZZ2Set/test/cricket_test/**/*.wav', np.array([0, 1, 0]), 300)
    append_to_dat_long(make_audio, buzz_train_d_conv, 'BUZZ2Set/test/noise_test/**/*.wav', np.array([0, 0, 1]), 300)
    global buzz_test_d_conv
    buzz_test_d_conv = buzz_train_d_conv

def make_image_ann():
    n = Network([1024, 50, 2])
    n.SGD2(bee_train_d, 20, 10, 0.01, 1, bee_test_d, True, True, True, True)
    pickle_object(n, 'pck_nets/ImageANN.pck')

def make_audio_ann():
    n = []
    for i in range(0, MAX_AUDIO, AUDIO_INC):
        print("SECTION", i)
        n.append(Network([AUDIO_INC, 80, 3]))
        n[-1].SGD3(buzz_train_d, 6, 10, 0.01, i, 1, buzz_test_d, False, True, False, True)
    pickle_object(n, 'pck_nets/AudioANN.pck')

def make_image_convnet():
    input_layer = input_data(shape=[None, 32, 32, 1])
    conv_layer_1  = conv_2d(input_layer, nb_filter=16, filter_size=5, activation='relu', name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2  = conv_2d(pool_layer_1, nb_filter=32, filter_size=5, activation='relu', name='conv_layer_2')
    pool_layer_2  = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1  = fully_connected(pool_layer_2, 100, activation='tanh', name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2, activation='softmax', name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.1)
    model = tflearn.DNN(network)

    model.fit(bee_train_d_conv[0], bee_train_d_conv[1], n_epoch=30,
          shuffle=True,
          validation_set=(bee_test_d_conv[0], bee_test_d_conv[1]),
          show_metric=True,
          batch_size=100,
          run_id='image_convnet')
    model.save('pck_nets/ImageConvNet.tfl')
'''
def make_audio_convnet():
    #l = len(buzz_train_d_conv[0])
    l = 25
    buzz_train_d_conv[0] = buzz_train_d_conv[0].reshape([-1, 25, 25, 1])
    dd = []
    for i in range(l):
        for j in range(96):
            dd.append(buzz_train_d_conv[1][i])
    buzz_train_d_conv[1] = np.array(dd)
    global buzz_test_d_conv
    buzz_test_d_conv = buzz_train_d_conv
    
    input_layer = input_data(shape=[None, 25, 25, 1])
    conv_layer_1  = conv_2d(input_layer, nb_filter=8, filter_size=5, activation='relu', name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2  = conv_2d(pool_layer_1, nb_filter=16, filter_size=5, activation='relu', name='conv_layer_2')
    pool_layer_2  = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1  = fully_connected(pool_layer_2, 100, activation='tanh', name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3, activation='softmax', name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.1)
    model = tflearn.DNN(network)

    model.fit(buzz_train_d_conv[0], buzz_train_d_conv[1], n_epoch=10,
          shuffle=True,
          validation_set=(buzz_test_d_conv[0], buzz_test_d_conv[1]),
          show_metric=True,
          batch_size=100,
          run_id='audio_convnet')
    model.save('pck_nets/AudioConvNet.tfl')
'''
def make_audio_convnet():
    global buzz_train_d_conv
    buzz_test_d_conv = buzz_train_d_conv
    
    input_layer = input_data(shape=[None, MAX_CONV, 1])
    conv_layer_1  = conv_1d(input_layer, nb_filter=8, filter_size=79, activation='relu', name='conv_layer_1')
    pool_layer_1  = max_pool_1d(conv_layer_1, 100, name='pool_layer_1')
    conv_layer_2  = conv_1d(pool_layer_1, nb_filter=16, filter_size=11, activation='relu', name='conv_layer_2')
    pool_layer_2  = max_pool_1d(conv_layer_2, 5, name='pool_layer_2')
    fc_layer_1  = fully_connected(pool_layer_2, 100, activation='tanh', name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3, activation='softmax', name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.1)
    model = tflearn.DNN(network)

    model.fit(buzz_train_d_conv[0], buzz_train_d_conv[1], n_epoch=30,
          shuffle=True,
          validation_set=(buzz_test_d_conv[0], buzz_test_d_conv[1]),
          show_metric=True,
          batch_size=100,
          run_id='audio_convnet')
    model.save('pck_nets/AudioConvNet.tfl')
