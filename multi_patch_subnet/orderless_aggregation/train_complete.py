#!/usr/bin/env python
# coding: utf-8

# ### Imports


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate, Average, Maximum, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend  as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import BinaryAccuracy, AUC
import cv2
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import pickle
import os
import glob2 as glob
import pandas as pd
import numpy as np


# ### Paths

img_dir = 'AVA_images'
df_pickle_path = 'df.pickle'


# ### Import the patches bounding boxes


with open(df_pickle_path, "rb") as openfile:
    df = pickle.load(openfile)

print(df.shape)

# ### Setting hyperparameters

def get_param():
    param = {}
    param['img_size'] = 224
    param['batch_size'] = 4
    param['nb_crops'] = 5
    param['nb_channels'] = 3

    #param['nb_epochs'] = 35
    param['nb_epochs'] = 8
    param['learning_rate'] = 0.000001
    param['weight_decay_coeff'] = 1e-5
    return param

param = get_param()
# ### Function to keep for later

def pad_small_image(self, img):
        old_size = np.array(img.shape[:2], dtype = int)
        max_size = old_size.max()
        ratio = self.img_size / max_size
        if ratio > 1:
            new_size = np.array([int(x*ratio)+1 for x in old_size])
            img = cv2.resize(img, (new_size[1], new_size[0]))
        else:
            new_size = old_size

        delta = self.img_size - new_size
        delta = np.where(delta < 0, 0, delta)
        top_left = delta // 2
        bottom_right = delta - (delta//2)
        outputImage = cv2.copyMakeBorder(img, top_left[0], bottom_right[0], top_left[1], bottom_right[1], cv2.BORDER_CONSTANT, value=[0,0,0])
        img = outputImage

        return img

# #### Processing functions
def preprocess_input(x):
    mean = [103.939, 116.779, 123.68]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    return x


def unpreprocess_input(x):
    mean = [103.939, 116.779, 123.68]
    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]
    return x.astype(np.uint8)

# ### Training Generator

class TrainingGenerator(tf.keras.utils.Sequence):
    """ Generates inputs for the Keras model during training.
    Attributes: - batch_size (int): batch size during training
                - df (pandas.DataFrame): dataframe with columns: idImage, labels
                - img_size (int): shape of the inputs of the model
                - indexes (np.array): indexes used for splitting the dataset in batches
    Functions:
                - __init__
                - __len__
                - __get_item__
                - pad_small_image
                - data_generation
    """
    def __init__(self, df, img_dir, augmentation = False):
        param = get_param()
        self.df = df
        self.img_dir = img_dir
        self.fns, self.bboxes_list, self.labels = self.sample_dataframe()
        print('proportion positives: {}'.format(str(np.sum(self.labels)/len(self.labels))))
        self.batch_size = param['batch_size']
        self.img_size = param['img_size']
        self.nb_crops = self.bboxes_list[0].shape[0]
        self.nb_channels = 3

        self.indexes = np.arange(len(self.fns))
        np.random.shuffle(self.indexes)
        self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(len(self.fns) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data'
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:min((index+1)*self.batch_size,len(self.fns))]

        # Find list of IDs
        fns_batch = [self.fns[k] for k in indexes]
        bboxes_list_batch = [self.bboxes_list[k] for k in indexes]
        labels = self.labels[indexes]

        # Generate data
        X = self.data_generation(fns_batch, bboxes_list_batch)

        #augmentation
        if self.augmentation: # custum hflip
            idx_hflip = np.random.randint(2, size = X.shape[0]) == 1
            if idx_hflip.any():
                X[idx_hflip] = X[idx_hflip][..., ::-1, :]

        X = preprocess_input(X)

        return X, labels

    def sample_dataframe(self):
        idImage = list(self.df['idImage'].values)
        fns = [os.path.join(self.img_dir, str(id_im) + '.jpg') for id_im in idImage]
        bboxes_list = list(self.df['BBoxes'].values)
        labels = self.df['labels'].values.astype(np.bool_)
        return fns, bboxes_list, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.fns))
        np.random.shuffle(self.indexes)

    def data_generation(self, fns_batch, list_bboxes_batch):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        X = np.zeros((len(fns_batch), self.nb_crops, self.img_size, self.img_size, self.nb_channels), dtype = np.float32)
        # load data
        for i in range(len(fns_batch)):
            fn = fns_batch[i]
            img = cv2.imread(fn)
            bboxes = list_bboxes_batch[i]
            for j in range(bboxes.shape[0]):
                bbox = bboxes[j]
                crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :].copy()
                X[i, j] = crop

        return X

# ### Model definition


def get_model(global_version = True):
    """ Return the parallelized VGG16 for the multi-patch subnet
    Inputs: - global_version (bool): True when training only the multi-path subnet
    Outputs: - model: keras model
    """
    K.clear_session()
    param = get_param()
    input_shape = (param['nb_crops'], param['img_size'], param['img_size'], param['nb_channels'])

    vgg16 = VGG16(include_top = False, weights = 'imagenet', input_shape = (param['img_size'], param['img_size'], param['nb_channels']))
    flatten = Flatten()(vgg16.output)
    fc1_vgg = Dense(4096, activation = 'relu', name = 'fc1_vgg')(flatten)
    dropout_vgg = Dropout(rate = 0.5, name = 'dropout_vgg')(fc1_vgg)
    fc2_vgg = Dense(4096, activation = 'relu', name = 'fc2_vgg')(dropout_vgg)
    backbone = Model(vgg16.input, fc2_vgg, name = 'backbone')
    backbone.trainable = True
    for idx, layer in enumerate(backbone.layers):
        if idx < 15:
            layer.trainable = False
        else:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer.add_loss(lambda: l2(param['weight_decay_coeff'])(layer.kernel))
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                layer.add_loss(lambda: l2(param['weight_decay_coeff'])(layer.bias))


    inputs = Input(shape = input_shape, name = 'input')
    in1, in2, in3, in4, in5 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3], inputs[:, 4]
    out1, out2, out3, out4, out5 = backbone(in1), backbone(in2), backbone(in3), backbone(in4), backbone(in5)

    # agregation
    agg_avg = Average(name = 'average')([out1, out2, out3, out4, out5])
    agg_max = Maximum(name = 'max')([out1, out2, out3, out4, out5])

    agg = concatenate([agg_avg, agg_max], name = 'agregation_layer')

    fc1 = Dense(units = 4096, activation = 'relu', name = 'fc1')(agg)
    dropout = Dropout(rate = 0.5, name = 'dropout')(fc1)
    fc2 = Dense(units = 4096, activation = 'relu', name = 'fc2')(dropout) # output of the multi-patch subnet in the A-Lamp model
    if global_version:
        out = fc2
    else:
        out = Dense(units = 1, activation = 'sigmoid', name = 'predictions')(fc2) # we first train the multi-patch subnet alone

    model = Model(inputs = inputs, outputs = out)

    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            layer.add_loss(lambda: l2(param['weight_decay_coeff'])(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(lambda: l2(param['weight_decay_coeff'])(layer.bias))

    return model


model = get_model(global_version = False)




model.summary()

"""
from tensorflow.keras.losses import Loss


class BinaryFocalLoss_fixed(Loss):
    def __init__(self, gamma = 2., alpha = .75):
        super().__init__()
        self.gamma = gamma
        self.alpha= alpha

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return self.binary_focal_loss_fixed(y_true, y_pred)

    def binary_focal_loss_fixed(self, y_true, y_pred):

        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
        loss_value = -K.sum(self.alpha * K.pow(1. - pt_1, self.gamma) * K.log(pt_1)) \
                -K.sum((1- self.alpha) * K.pow(pt_0, self.gamma) * K.log(1. - pt_0)),
        return loss_value
"""

#opt = Adam(learning_rate = lr)
opt = SGD(learning_rate = param['learning_rate'], momentum = 0.9)
model.load_weights('best_weights.hdf5')

for idx, layer in enumerate(model.layers[6].layers):
    if idx < 15:
        layer.trainable= False

model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = [BinaryAccuracy(), AUC()])

for layer in model.layers[6].layers:
    print(layer, layer.trainable)

# ### Callbacks



callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3),
             ModelCheckpoint("weights.{epoch:02d}-acc{val_binary_accuracy:.2f}.hdf5", monitor='val_binary_accuracy', mode = 'max', save_best_only=True, verbose=1),
             ModelCheckpoint("weights.{epoch:02d}-auc{val_auc:.2f}.hdf5", monitor='val_auc', mode = 'max', save_best_only=True, verbose=1),
             TensorBoard(log_dir='./logs/'+str(datetime.datetime.now()).replace(' ', '/'))
]


# ### Training

other_df, test_df = train_test_split(df, train_size = 0.9, stratify = df['labels'], random_state=42)

train_df, val_df = train_test_split(other_df, train_size = 0.9, stratify = other_df['labels'], random_state=42)
del other_df



train_generator = TrainingGenerator(train_df, img_dir, augmentation = True)
validation_generator = TrainingGenerator(val_df, img_dir, augmentation = False)


model.fit(train_generator, epochs = param['nb_epochs'], verbose = 1, callbacks = callbacks, validation_data = validation_generator, shuffle = False)
