# -*- coding: utf-8 -*-

import argparse
import numpy as np 
import os
import zipfile
import tensorflow as tf
import datetime

from pathlib import Path
from flim_datagen import DecayGenerator
from data_extraction import extract_link
from flim_net_gan import conditional_generator, conditional_critic


#python train_flim_gan.py --steps 30000 --batch_size 2 D:\Data\DL-FLIM https://www.dropbox.com/s/0lpsp1r9ma3nvmr/train_gan.zip https://www.dropbox.com/s/hb9xhjl2flt5hrw/test_gan.zip?dl=0

class FLIMGAN:

    def __init__(self, data_dir,
                    train_link,
                    test_link,
                    nTG,
                    val_split,
                    hparams):
        self.strategy = tf.distribute.get_strategy()

        self.data_dir = data_dir
        self.train_link = train_link
        self.test_link = test_link

        self.nTG = nTG
        self.batch_size = hparams['batch_size'] * self.strategy.num_replicas_in_sync

        self.alpha = hparams['alpha']
        self.clipping = hparams['clipping']
        self.ncritic = hparams['ncritic']
        self.batch_size = hparams['batch_size']
        self.val_split = val_split

        self.logdir = os.path.join(data_dir,'logs')
        self.checkpt = os.path.join(data_dir,'checkpt')

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        if not os.path.exists(self.checkpt):
            os.makedirs(self.checkpt)

        #train_dir = 'C:\\Users\\xieji\\Dropbox\\Documents\\Data\\DL-FLIM\\train_gan'
        #test_dir = 'C:\\Users\\xieji\\Dropbox\\Documents\\Data\\DL-FLIM\\test_gan'
        train_dir = extract_link(data_dir, train_link)
        test_dir = extract_link(data_dir, test_link)

        self.ds_gan = DecayGenerator(train_dir,test_dir,nTG,self.batch_size,self.val_split,'gan')
        with self.strategy.scope():
            # generator: (dk_lowcount, irf) -> dk_high
            self.generator = conditional_generator((tf.keras.Input(shape=(nTG,1)),
                                            tf.keras.Input(shape=(nTG,1)))) 
            # critic: (dk_high, irf) -> score
            self.critic = conditional_critic((tf.keras.Input(shape=(nTG,1)),
                                    tf.keras.Input(shape=(nTG,1))))
            self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.alpha)
            self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.alpha, clipvalue=self.clipping)

            self.checkpoint_prefix = os.path.join(self.checkpt, "ckpt")
            self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         critic_optimizer=self.critic_optimizer,
                                         generator=self.generator,
                                         critic=self.critic)

    @tf.function
    def generator_loss(self, crit_generated_output):
        # Mean absolute error (sum) from critic output
        return -tf.reduce_mean(tf.abs(crit_generated_output))

    @tf.function
    def critic_loss(self, crit_generated_output, crit_real_output):
        return -tf.reduce_mean(tf.abs(crit_real_output)) + tf.reduce_mean(tf.abs(crit_generated_output))

    def train_critic_step(self, critic_inputs):
        dk_low, dk_high = critic_inputs
        # Sample prior, real data, compute loss on critic
        with tf.GradientTape() as crit_tape:
            gen_output = self.generator(dk_low, training=True)
            crit_real_output = self.critic([dk_high, dk_low[1]], training=True)
            crit_generated_output = self.critic([gen_output, dk_low[1]], training=True)
            crit_loss = self.critic_loss(crit_generated_output, crit_real_output)
        critic_gradients = crit_tape.gradient(crit_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        return crit_loss

    def train_generator_step(self,dk_low):
        with tf.GradientTape() as gen_tape:
            gen_output = self.generator(dk_low, training=True)
            crit_generated_output = self.critic([gen_output, dk_low[1]],training=True)
            gen_loss = self.generator_loss(crit_generated_output)
        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        return gen_loss

    @tf.function
    def distributed_train_critic_step(self, critic_inputs):
        per_replica_losses = self.strategy.run(self.train_critic_step, args=(critic_inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_train_generator_step(self, generator_inputs):
        per_replica_losses = self.strategy.run(self.train_generator_step, args=(generator_inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    def train_wasserstein(self, steps):

        summary_writer = tf.summary.create_file_writer(
            os.path.join(self.checkpt, "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        train_ds = self.ds_gan.train
        test_ds = self.ds_gan.test

        self.critic.summary()

        for step in range(steps):
            for _ in range(self.ncritic):
                critic_inputs = next(iter(train_ds))
                crit_loss = self.distributed_train_critic_step(critic_inputs)
            generator_inputs, _ = next(iter(train_ds))
            gen_loss = self.distributed_train_generator_step(generator_inputs)

            with summary_writer.as_default():
                tf.summary.scalar('gen_loss', gen_loss, step=step//1000)
                tf.summary.scalar('crit_loss', crit_loss, step=step//1000)

            # Training step
            if (step+1) % 10 == 0:
                print('.', end='', flush=True)

            # Save (checkpoint) the model every 500 steps
            if (step + 1) % 100 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print('gen_loss: {}'.format(gen_loss))
                print('crit_loss: {}'.format(crit_loss))
                self.ds_gan.plot(self.generator, savedir=self.logdir)
                dk_low, dk_high = self.ds_gan.example
                dk_super = self.generator(dk_low,training=False)

                critique_gen = self.critic([dk_super,dk_low[1]],training=False)
                critique_real = self.critic([dk_high,dk_low[1]],training=False)

                print('Generator critic score:\n{}'.format(critique_gen))
                print('Real decay critic score:\n{}'.format(critique_real))


        return


def main(args):
    nTG = 256
    hparams = {'alpha':5e-5,
           'clipping':0.01,
           'ncritic':5,
           'batch_size':args.batch_size}

    gan = FLIMGAN(args.data_dir, args.train_link, args.test_link, nTG, 0.2, hparams)
    steps = args.steps
    gan.train_wasserstein(steps)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline for DL_FLIM')
    parser.add_argument('data_dir', type=str,
        help='Data directory')
    parser.add_argument('train_link', type=str,
        help='Link to training data')
    parser.add_argument('test_link', type=str,
        help='Link to testing data')

    parser.add_argument('--steps', default=250, type=int, 
        help='Number of steps to train for')
    parser.add_argument('--batch_size', default=64, type=int, 
        help='Batch size')
    args = parser.parse_args()

    print('GAN Training args:')
    print('    Number of steps: {}'.format(args.steps))
    print('    Batch size: {}'.format(args.batch_size))
    print('    Data directory: {}'.format(args.data_dir))
    print('    Training data link: {}'.format(args.train_link))
    print('    Testing data link: {}'.format(args.test_link))
    main(args)

