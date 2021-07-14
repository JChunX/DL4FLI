# -*- coding: utf-8 -*-

# Relevant libraries and functions
import argparse

parser = argparse.ArgumentParser(description='Training pipeline for DL_FLIM')

parser.add_argument('data_dir', type=str,
    help='Data directory')
parser.add_argument('train_link', type=str,
    help='Link to training data')
parser.add_argument('test_link', type=str,
    help='Link to testing data')
parser.add_argument('checkpoint', default=None, type=str, 
    help='Model checkpoint filename')

parser.add_argument('--epochs', default=250, type=int, 
    help='Number of epochs to train for')
parser.add_argument('--batch_size', default=64, type=int, 
    help='Batch size')

import numpy as np 
import os
import zipfile
import tensorflow as tf

from pathlib import Path
from flim_datagen import VoxelGenerator
from flim_net import IRFDeconv3D, resblock_2D_BN, resblock_3D_BN
from data_extraction import extract_link 

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import  (
    Activation, 
    AveragePooling2D, 
    BatchNormalization,
    Conv2D,Conv3D, 
    Conv3DTranspose,
    Dense, 
    Flatten,
    Input,
    MaxPool3D,  
    Reshape,
)

mirrored_strategy = tf.distribute.MirroredStrategy()

def main(args):

    data_dir = args.data_dir

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(os.path.join(data_dir,'checkpoints')):
        os.makedirs(os.path.join(data_dir,'checkpoints'))
    if not os.path.exists(os.path.join(data_dir,'logs')):
        os.makedirs(os.path.join(data_dir,'logs'))

    checkpoint = os.path.join(args.data_dir,'checkpoints', args.checkpoint)

    train_link = args.train_link
    test_link = args.test_link

    train_dir = extract_link(data_dir, train_link)
    test_dir = extract_link(data_dir, test_link)

    nTG = 256
    xX = 28
    yY = 28
    batch_size = args.batch_size * mirrored_strategy.num_replicas_in_sync
    val_split = 0.2
    voxels = VoxelGenerator(train_dir,test_dir,nTG,xX,yY,batch_size,val_split)
    
    with mirrored_strategy.scope():
        modelD = None

        dk_data = Input(shape=(xX, yY, nTG,1))
        irf_data = Input(shape=(1, 1, nTG,1))
        tpsf = dk_data
        irf = irf_data
        # # # # # # # # 3D-Model # # # # # # # #
        tpsf = IRFDeconv3D(xX,yY,nTG)((tpsf,irf))
        tpsf = Conv3D(64,kernel_size=(1,1,15),strides=(1,1,5), activation=None, data_format="channels_last")(tpsf)
        tpsf = BatchNormalization()(tpsf)
        tpsf = Activation(tf.nn.relu)(tpsf)
        tpsf = MaxPool3D((1,1,3), strides=(1,1,2))(tpsf)

        tpsf = resblock_3D_BN(64, 32, (1,1,5), tpsf)
        tpsf = resblock_3D_BN(64, 32, (1,1,5), tpsf)

        tpsf = Conv3D(128,kernel_size=(1,1,15),strides=(1,1,3), activation=None, data_format="channels_last")(tpsf)
        tpsf = BatchNormalization()(tpsf)
        tpsf = Activation(tf.nn.relu)(tpsf)
        tpsf = MaxPool3D((1,1,3), strides=(1,1,2))(tpsf)

        tpsf_ = Reshape((xX,yY,128))(tpsf)
        tpsf_ = resblock_2D_BN(128,64, 1,tpsf_)
        tpsf_ = resblock_2D_BN(128,64, 1,tpsf_)
        tpsf_ = resblock_2D_BN(128,64, 1,tpsf_)

        # Short-lifetime branch
        imgT1 = Conv2D(64, 5, activation=None)(tpsf_)
        imgT1 = BatchNormalization()(imgT1)
        imgT1 = Activation(tf.nn.relu)(imgT1)
        imgT1 = Conv2D(32, 1, activation=None)(imgT1)
        imgT1 = BatchNormalization()(imgT1)
        imgT1 = Activation(tf.nn.relu)(imgT1)
        imgT1 = Conv2D(1, 3, activation=None)(imgT1)
        imgT1 = BatchNormalization()(imgT1)
        imgT1 = Activation(tf.nn.relu)(imgT1)
        t1 = Flatten()(imgT1)
        t1 = Dense(512, activation=tf.nn.relu)(t1)
        t1 = Dense(1, activation=tf.nn.relu)(t1)

        # Long-lifetime branch
        imgT2 = Conv2D(64, 5, activation=None)(tpsf_)
        imgT2 = BatchNormalization()(imgT2)
        imgT2 = Activation(tf.nn.relu)(imgT2)
        imgT2 = Conv2D(32, 1, activation=None)(imgT2)
        imgT2 = BatchNormalization()(imgT2)
        imgT2 = Activation(tf.nn.relu)(imgT2)
        imgT2 = Conv2D(1, 3, activation=None)(imgT2)
        imgT2 = BatchNormalization()(imgT2)
        imgT2 = Activation(tf.nn.relu)(imgT2)
        t2 = Flatten()(imgT2)
        t2 = Dense(512, activation=tf.nn.relu)(t2)
        t2 = Dense(1, activation=tf.nn.relu)(t2)

        # Amplitude-Ratio branch
        imgTR = Conv2D(64, 5, padding='same', activation=None)(tpsf_)
        imgTR = BatchNormalization()(imgTR)
        imgTR = Activation(tf.nn.relu)(imgTR)
        imgTR = Conv2D(64, 1, padding='same', activation=None)(imgTR)
        imgTR = BatchNormalization()(imgTR)
        imgTR = Activation(tf.nn.relu)(imgTR)
        imgTR = Conv2D(1, 3, padding='same', activation=None)(imgTR)
        imgTR = BatchNormalization()(imgTR)
        imgTR = Activation(tf.nn.relu)(imgTR)

        # Decoder
        decoded = Conv3DTranspose(64,kernel_size=(1,1,15),strides=(1,1,3))(tpsf)
        decoded = Conv3DTranspose(128,kernel_size=(1,1,15),strides=(1,1,5))(decoded)
        decoded = Conv3DTranspose(1,kernel_size=(1,1,4),strides=(1,1,3))(decoded)

        modelD = Model(inputs=(dk_data,irf_data), outputs=(t1, t2, imgTR, decoded))

        adam = Adam(learning_rate=1e-5,epsilon=0.001)

    if Path(checkpoint).exists():
        modelD.load_weights(checkpoint)
    modelD.compile(loss='mae',
                optimizer=adam,
                metrics=['mse'])

    # Setting patience (patience = 15 recommended)
    earlyStopping = EarlyStopping(monitor='val_loss', 
                                  patience = 25, 
                                  verbose = 0,
                                  mode = 'auto')

    learning_curve = os.path.join(data_dir,'logs','learning-curve.log')

    # Save loss curve (mse) and MAE information over all trained epochs. (monitor = '' can be changed to focus on other tau parameters)
    modelCheckPoint = ModelCheckpoint(filepath=checkpoint, 
                                      monitor='val_loss', 
                                      save_best_only=True, 
                                      verbose=0)
    # Train network (80/20 train/validation split, batch_size=20 recommended, nb_epoch may vary based on application)
    history = History()
    csv_logger = CSVLogger(learning_curve)
    history = modelD.fit(voxels.train, validation_data=voxels.val,
              epochs=args.epochs, verbose=1, callbacks=[earlyStopping,csv_logger,modelCheckPoint])
    modelD.evaluate(voxels.test)
    voxels.plot(modelD, max_subplots=15)

if __name__ == '__main__':
    args = parser.parse_args()
    print('Training args:')
    print('    Number of epochs: {}'.format(args.epochs))
    print('    Batch size: {}'.format(args.batch_size))
    print('    Data directory: {}'.format(args.data_dir))
    print('    Training data link: {}'.format(args.train_link))
    print('    Testing data link: {}'.format(args.test_link))
    print('    Checkpoint: {}'.format(os.path.join(args.data_dir,args.checkpoint)))
    main(args)

