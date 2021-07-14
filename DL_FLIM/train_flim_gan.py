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

parser.add_argument('--epochs', default=250, type=int, 
    help='Number of epochs to train for')
parser.add_argument('--batch_size', default=64, type=int, 
    help='Batch size')

import numpy as np 
import os
import zipfile
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import tensorflow_gan as tfgan

from pathlib import Path
from flim_datagen import DecayGenerator
from data_extraction import extract_link
from flim_net_gan import conditional_generator, unconditional_critic

mirrored_strategy = tf.distribute.MirroredStrategy()

def main(args):
    data_dir = args.data_dir

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(os.path.join(data_dir,'logs')):
        os.makedirs(os.path.join(data_dir,'logs'))
    train_link = args.train_link
    test_link = args.test_link

    train_dir = 'C:\\Users\\xieji\\Dropbox\\Documents\\Data\\DL-FLIM\\train_global_gan'#extract_link(data_dir, train_link)
    test_dir = 'C:\\Users\\xieji\\Dropbox\\Documents\\Data\\DL-FLIM\\test_global_gan'#extract_link(data_dir, test_link)

    nTG = 256

    batch_size = args.batch_size * mirrored_strategy.num_replicas_in_sync
    val_split = 0.2
    ds_gan = DecayGenerator(train_dir,test_dir,nTG,batch_size,val_split,'gan')
    train(ds_gan,(batch_size, os.path.join(data_dir,'logs'), args.epochs))

def train(ds_gan, hparams):

    batch_size, train_log_dir, epochs = hparams

    dk_low, irf, dk_high = tf.data.make_one_shot_iterator(ds_gan.train).get_next()
    gan_model = tfgan.gan_model(
        generator_fn = conditional_generator,
        discriminator_fn = unconditional_critic,
        real_data = tf.concat([dk_high, irf],0),
        generator_inputs = (dk_low, irf)
        )

    with tf.name_scope('loss'):
        gan_loss = tfgan.gan_loss(gan_model, add_summaries=True)
        tfgan.eval.add_regularization_loss_summaries(gan_model)

    with tf.name_scope('train'):
        gen_lr, dis_lr = (1e-5, 1e-4)
        train_ops = tfgan.gan_train_ops(
            gan_model,
            gan_loss,
            generator_optimizer=tf.train.AdamOptimizer(gen_lr, 0.01),
            discriminator_optimizer=tf.train.AdamOptimizer(dis_lr, 0.01),
            summarize_gradients=True,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    # Run the alternating training loop. Skip it if no steps should be taken
    # (used for graph construction tests).
    status_message = tf.strings.join([
        'Starting train step: ',
        tf.as_string(tf.train.get_or_create_global_step())], 
        name='status_message')
    if epochs == 0:
        return

    tfgan.gan_train(
        train_ops,
        hooks=[
            tf.estimator.StopAtStepHook(num_steps=epochs),
            tf.estimator.LoggingTensorHook([status_message], every_n_iter=10)
            ],
        logdir=train_log_dir,
        get_hooks_fn=tfgan.get_joint_train_hooks(),
        save_checkpoint_secs=60)

if __name__ == '__main__':
    args = parser.parse_args()
    print('GAN Training args:')
    print('    Number of epochs: {}'.format(args.epochs))
    print('    Batch size: {}'.format(args.batch_size))
    print('    Data directory: {}'.format(args.data_dir))
    print('    Training data link: {}'.format(args.train_link))
    print('    Testing data link: {}'.format(args.test_link))
    main(args)

