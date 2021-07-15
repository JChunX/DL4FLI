import tensorflow as tf
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt 

from tensorflow.experimental.numpy import moveaxis
from abc import ABC, abstractmethod

class TcspcDataGenerator(ABC):
    def __init__(self, train, test,         
                 batch_size, val_split):
        self.train_dir = train
        self.test_dir = test


        self.batch_size = batch_size
        self.val_split = val_split

        train_files = os.listdir(self.train_dir)
        num_samples = len(train_files)
        list_ds = tf.data.Dataset.list_files(self.train_dir + '/*')
        val_size = int(num_samples * self.val_split)
        self.train_ds = list_ds.skip(val_size)
        self.val_ds = list_ds.take(val_size)
        self.test_ds = tf.data.Dataset.list_files(self.test_dir + '/*')

    @abstractmethod
    def wrap_process_path(self):
        pass

    def configure_for_performance(self,ds):
        ds = ds.cache()
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=1)
        ds = ds.repeat()
        return ds

    def get_dataset(self, ds):
        ds = ds.map(self.wrap_process_path, num_parallel_calls=os.cpu_count())
        ds = self.configure_for_performance(ds)
        return ds

    @property
    def train(self):
        return self.get_dataset(self.train_ds)

    @property
    def val(self):
        return self.get_dataset(self.val_ds)

    @property
    def test(self):
        return self.get_dataset(self.test_ds)

    @property
    def example(self):
        return next(iter(self.test))

    def benchmark(self, num_epochs=1):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            for i, sample in enumerate(self.val):
                # Performing a training step
                print(i)
        print("Execution time:", time.perf_counter() - start_time)


class DecayGenerator(TcspcDataGenerator):
    def __init__(self, train, test,         
                 nTG,
                 batch_size, val_split,
                 type):
        super().__init__(train,test,batch_size,val_split)

        self.nTG = nTG 
        self.type = type

    def __repr__(self):
        return '\n'.join([
            f'nTG: {self.nTG}'
        ])   

    def process_path(self, path):
        f = h5py.File(path,'r')
        decay_lowcount = np.array(f['dk_low'],dtype=np.float32)
        decay_highcount = np.array(f['dk_high'],dtype=np.float32)
        t1 = np.array(f['t1'],dtype=np.float32)
        t2 = np.array(f['t2'],dtype=np.float32)
        rT = np.array(f['rT'],dtype=np.float32)
        irf = np.array(f['irf'],dtype=np.float32)
        
        return (decay_lowcount, decay_highcount, t1, t2, rT, irf)

    def wrap_process_path(self, path):
        tensors = tf.numpy_function(self.process_path, 
            [path], [tf.float32, 
                    tf.float32,
                    tf.float32, 
                    tf.float32, 
                    tf.float32,
                    tf.float32])
        decay_lowcount = tf.transpose(tensors[0])
        decay_highcount = tf.transpose(tensors[1])
        t1 = tensors[2]
        t2 = tensors[3]
        rT = tensors[4]
        irf = tf.transpose(tensors[5])

        if self.type == 'gan':
            return (decay_lowcount, irf), decay_highcount
        else:
            return (decay_highcount), (t1,t2,rT)

    def plot(self, model=None, max_subplots=1):
        dk, labels = self.example
        for i in range(min(max_subplots,self.batch_size)):

            if self.type == 'gan':
                plt.figure(figsize=(30,10))
                plt.subplot(1,3,1)
                plt.title('Low count decay example')
                plt.plot(dk[0][i])
                plt.subplot(1,3,2)
                plt.title('IRF')
                plt.plot(dk[1][i])
                if model is not None:
                    dk_highcount = model(dk)
                    plt.subplot(1,3,3)
                    plt.title('Upscaled decay')
                    plt.plot(dk_highcount[i])

            if self.type == 'est':
                print('========Single decay parameters========')
                print('tau1: {}'.format(labels[0][i,0,0].numpy())) #TODO
                print('tau2: {}'.format(labels[1][i,0,0].numpy())) #TODO
                print('alpha%: {}'.format(labels[2][i,0,0].numpy())) #TODO
                if model is not None:
                    predictions = model(dk)
                    print('tau1_pred: {}'.format(predictions[0][i,0,0].numpy())) #TODO
                    print('tau2_pred: {}'.format(predictions[1][i,0,0].numpy())) #TODO
                    print('alpha%_pred: {}'.format(predictions[2][i,0,0].numpy())) #TODO
            plt.show()
            plt.pause(0.001)


class VoxelGenerator(TcspcDataGenerator):

    def __init__(self, train, test,         
                 nTG, xX, yY, 
                 batch_size, val_split):
        super().__init__(train,test,batch_size,val_split)

        self.nTG = nTG
        self.xX = xX
        self.yY = yY

    def __repr__(self):
        return '\n'.join([
            f'nTG: {self.nTG}',
            f'xX: {self.xX}',
            f'yY: {self.yY}'
        ])

    def process_path(self, path):
        f = h5py.File(path,'r')
        decay = np.array(f['sigD'],dtype=np.float32)
        t1 = np.array(f['t1'],dtype=np.float32)
        t2 = np.array(f['t2'],dtype=np.float32)
        rT = np.array(f['rT'],dtype=np.float32)
        irf = np.array(f['irf'],dtype=np.float32)
        
        return (decay, t1, t2, rT, irf)

    def wrap_process_path(self, path):
        tensors = tf.numpy_function(self.process_path, [path], [tf.float32, 
                                                        tf.float32, 
                                                        tf.float32, 
                                                        tf.float32,
                                                        tf.float32])
        decay = tf.reshape(tf.transpose(tensors[0],[2,1,0]), (self.xX, self.yY, self.nTG, 1))
        t1 = tensors[1]
        t2 = tensors[2]
        rT = tf.reshape(tf.transpose(tensors[3]), (self.xX, self.yY, 1))
        irf = tf.reshape(tensors[4],(1,1,self.nTG,1,1))
        return (decay, irf), (t1, t2, rT, decay)

    def plot(self, model=None, max_subplots=5):
        voxels, labels = self.example
        for i in range(min(max_subplots,self.batch_size)):
            tau1 = labels[0][i,0,0].numpy()
            tau2 = labels[1][i,0,0].numpy()
            A = labels[2][i].numpy()[:,:,0]
            tau_m = A*tau1 + (1-A)*tau2
            tau_m[A == 0] = 0.
            plt.figure(figsize=(10,30))
            tau_min, tau_max = 0.0, np.max(tau_m)
            plt.subplot(411)
            plt.title('tau_m: (tau1={0},tau2={1})'.format(tau1,tau2))
            plt.imshow(tau_m, vmin=tau_min, vmax=tau_max)
            
            if model is not None:
                predictions = model(voxels)
                tau1_pred = predictions[0][i,0].numpy()
                tau2_pred = predictions[1][i,0].numpy()
                A_pred = predictions[2][i].numpy()[:,:,0]

                print('Prediction Errors:')
                print(np.sum(np.abs(A_pred-A))/(self.xX*self.yY))
                print(np.abs(tau1_pred-tau1))
                print(np.abs(tau2_pred-tau2))
                
                tau_m_pred = A_pred*tau1_pred + (1-A_pred)*tau2_pred
                tau_m_pred[A == 0] = 0.
                plt.subplot(412)
                plt.title('Predicted tau_m: (tau1={0},tau2={1})'.format(
                                                        tau1_pred,tau2_pred))
                plt.imshow(tau_m_pred, 
                           vmin=tau_min, vmax=tau_max)
                
                plt.subplot(413)
                plt.imshow(np.abs(tau_m_pred-tau_m), vmin=tau_min, vmax=tau_max)
                plt.title('Mean Lifetime Absolute Error')
                plt.colorbar()
                plt.subplots_adjust(right=0.8)

                plt.subplot(414)
                plt.title('Decay Reconstruction')
                plt.plot(np.sum(labels[3][i,:,:,:].numpy().reshape(28,28,256),(0,1))/(28*28), 'b', label='Original')
                plt.plot(np.sum(predictions[3][i,:,:,:].numpy().reshape(28,28,256),(0,1))/(28*28), 'r', label='Reconstructed')
                plt.legend()