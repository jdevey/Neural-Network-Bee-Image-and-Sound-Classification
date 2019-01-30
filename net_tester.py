import cv2
#import glob
import pickle as cPickle
import numpy as np
import scipy as sp
from scipy.io import wavfile

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, conv_1d, max_pool_1d
from tflearn.layers.estimator import regression

MAX_AUDIO = 60000
AUDIO_INC = 1000
MAX_CONV = 10078

sample_image_path = 'BEE2Set/bee_train/img9/192_168_4_5-2017-05-12_18-38-06_413_6_68.png'
sample_audio_path = 'BUZZ2Set/train/bee_train/bee999_192_168_4_6-2017-08-29_09-15-01.wav'

def load_image_ann(filename):
    with open(filename, 'rb') as fp:
        ff = cPickle.load(fp)
        return ff

def load_image_convnet(filename):
    input_layer = input_data(shape=[None, 32, 32, 1])
    conv_layer_1  = conv_2d(input_layer, nb_filter=16, filter_size=5, activation='relu', name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2  = conv_2d(pool_layer_1, nb_filter=32, filter_size=5, activation='relu', name='conv_layer_2')
    pool_layer_2  = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1  = fully_connected(pool_layer_2, 100, activation='tanh', name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2, activation='softmax', name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.1)
    model = tflearn.DNN(network)
    model.load(filename)
    return model

def load_audio_ann(filename):
    with open(filename, 'rb') as fp:
        ff = cPickle.load(fp)
        return ff

def load_audio_convnet(filename):
    input_layer = input_data(shape=[None, MAX_CONV, 1])
    conv_layer_1  = conv_1d(input_layer, nb_filter=8, filter_size=79, activation='relu', name='conv_layer_1')
    pool_layer_1  = max_pool_1d(conv_layer_1, 100, name='pool_layer_1')
    conv_layer_2  = conv_1d(pool_layer_1, nb_filter=16, filter_size=11, activation='relu', name='conv_layer_2')
    pool_layer_2  = max_pool_1d(conv_layer_2, 5, name='pool_layer_2')
    fc_layer_1  = fully_connected(pool_layer_2, 100, activation='tanh', name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3, activation='softmax', name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.1)
    model = tflearn.DNN(network)
    model.load(filename)
    return model

def fit_image_ann(ann, image_path):
    img = make_image(image_path)
    ind = np.argmax(ann.feedforward(img.reshape([1024, 1])))
    ret = np.array([0, 0])
    ret[ind] = 1
    return ret

def fit_image_convnet(convnet, image_path):
    img = make_image(image_path)
    ind = np.argmax(convnet.predict(img.reshape([-1, 32, 32, 1])))
    ret = np.array([0, 0])
    ret[ind] = 1
    return ret

def fit_audio_ann(ann, audio_path):
    aud = make_audio(audio_path)
    aud = aud[:MAX_AUDIO]
    votes = [0, 0, 0]
    for i in range(0, int(MAX_AUDIO / AUDIO_INC)):
        votes[np.argmax(ann[i].feedforward(np.array(aud[i * AUDIO_INC : (i + 1) * AUDIO_INC]).reshape([AUDIO_INC, 1])))] += 1
    ret = np.array([0, 0, 0])
    ret[np.argmax(votes)] = 1
    return ret

def fit_audio_convnet(convnet, audio_path):
    aud = make_audio(audio_path)
    aud = aud[:MAX_CONV]
    ind = np.argmax(convnet.predict(aud.reshape([-1, MAX_CONV, 1])))
    ret = np.array([0, 0, 0])
    ret[ind] = 1
    return ret

def make_image(filename):
    img = cv2.imread(filename)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image / 255.0

def make_audio(filename):
    samplerate, audio = wavfile.read(filename)
    return audio / float(np.max(audio))

def eqArrs(a, b):
    for i in range(len(a)):
        if (a[i] != b[i]):
            return False
    return True

def rec_image_ann(ann, path, y):
    correct = 0
    i = 0
    '''
    for filename in glob.iglob(path, recursive=True):
        if (i > 5000):
            if (eqArrs(fit_image_ann(ann, filename), y)): correct += 1
            if (i > 5500): break
        print(i)
        i += 1
    '''
    return correct

def test_image_ann():
    ann = load_image_ann('pck_nets/92ImageANN.pck')
    total = 1000
    correct = 0
    correct += rec_image_ann(ann, 'BEE2Set/bee_train/**/*.png', np.array([1, 0]))
    correct += rec_image_ann(ann, 'BEE2Set/no_bee_train/**/*.png', np.array([0, 1]))
    return correct, total

def rec_image_conv(ann, path, y):
    correct = 0
    i = 0
    '''
    for filename in glob.iglob(path, recursive=True):
        if (i > 5000):
            if (eqArrs(fit_image_convnet(ann, filename), y)): correct += 1
            if (i > 5500): break
        print(i)
        i += 1
    '''
    return correct

def test_image_conv():
    model = load_image_convnet('pck_nets/ImageConvNet.tfl')
    total = 1000
    correct = 0
    correct += rec_image_conv(model, 'BEE2Set/bee_test/**/*.png', np.array([1, 0]))
    correct += rec_image_conv(model, 'BEE2Set/no_bee_test/**/*.png', np.array([0, 1]))
    return correct, total
    
def rec_audio_ann(ann, path, y):
    correct = 0
    i = 0
    '''
    for filename in glob.iglob(path, recursive=True):
        if (i > 500):
            if (eqArrs(fit_audio_ann(ann, filename), y)): correct += 1
            if (i > 600): break
        print(i)
        i += 1
    '''
    return correct

def test_audio_ann():
    ann = load_audio_ann('pck_nets/82AudioANN.pck')
    total = 300
    correct = 0
    correct += rec_audio_ann(ann, 'BUZZ2Set/train/bee_train/**/*.wav', np.array([1, 0, 0]))
    correct += rec_audio_ann(ann, 'BUZZ2Set/train/cricket_train/**/*.wav', np.array([0, 1, 0]))
    correct += rec_audio_ann(ann, 'BUZZ2Set/train/noise_train/**/*.wav', np.array([0, 0, 1]))
    return correct, total

def rec_audio_conv(ann, path, y):
    correct = 0
    i = 0
    '''
    for filename in glob.iglob(path, recursive=True):
        if (i > 300):
            if (eqArrs(fit_audio_convnet(ann, filename), y)): correct += 1
            if (i > 400): break
        print(i)
        i += 1
    '''
    return correct

def test_audio_conv():
    model = load_audio_convnet('pck_nets/AudioConvNet.tfl')
    total = 300
    correct1, correct2 = 0, 0
    correct1 += rec_audio_conv(model, 'BUZZ2SET/train/bee_train/**/*.wav', np.array([1, 0, 0]))
    correct1 += rec_audio_conv(model, 'BUZZ2Set/train/cricket_train/**/*.wav', np.array([0, 1, 0]))
    correct1 += rec_audio_conv(model, 'BUZZ2Set/train/noise_train/**/*.wav', np.array([0, 0, 1]))
    correct2 += rec_audio_conv(model, 'BUZZ2SET/test/bee_test/**/*.wav', np.array([1, 0, 0]))
    correct2 += rec_audio_conv(model, 'BUZZ2Set/test/cricket_test/**/*.wav', np.array([0, 1, 0]))
    correct2 += rec_audio_conv(model, 'BUZZ2Set/test/noise_test/**/*.wav', np.array([0, 0, 1]))
    return correct1, correct2, total

