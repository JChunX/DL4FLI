import tensorflow as tf
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt 

from tensorflow.experimental.numpy import moveaxis
from tensorflow.keras.layers import  (
    add, 
    Activation, 
    BatchNormalization,
    Conv2D,Conv3D, 
    Conv3DTranspose,
    Layer, 
    MaxPool3D,  
    SeparableConv2D
)


class VoxelGenerator():

    def __init__(self, train, test,         
                 nTG, xX, yY, 
                 batch_size, val_split):
        self.train_dir = train
        self.test_dir = test

        self.nTG = nTG
        self.xX = xX
        self.yY = yY
        self.batch_size = batch_size
        self.val_split = val_split

        train_files = os.listdir(self.train_dir)
        num_samples = len(train_files)
        list_ds = tf.data.Dataset.list_files(self.train_dir + '/*')
        val_size = int(num_samples * self.val_split)
        self.train_ds = list_ds.skip(val_size)
        self.val_ds = list_ds.take(val_size)
        self.test_ds = tf.data.Dataset.list_files(self.test_dir + '/*')

    def __repr__(self):
        return '\n'.join([
            f'nTG: {self.nTG}',
            f'xX: {self.xX}',
            f'yY: {self.yY}'
        ])

    def process_path(self, path):
        f = h5py.File(path,'r')
        sigD = np.array(f['sigD'],dtype=np.float32)
        t1 = np.array(f['t1'],dtype=np.float32)
        t2 = np.array(f['t2'],dtype=np.float32)
        rT = np.array(f['rT'],dtype=np.float32)
        irf = np.array(f['irf'],dtype=np.float32)
        
        return (sigD, t1, t2, rT, irf)

    def wrap_process_path(self, path):
        tensors = tf.numpy_function(self.process_path, [path], [tf.float32, 
                                                        tf.float32, 
                                                        tf.float32, 
                                                        tf.float32,
                                                        tf.float32])
        sigD = tf.reshape(tf.transpose(tensors[0],[2,1,0]), (self.xX, self.yY, self.nTG, 1))
        t1 = tensors[1]
        t2 = tensors[2]
        tR = tf.reshape(tf.transpose(tensors[3]), (self.xX, self.yY, 1))
        irf = tf.reshape(tensors[4],(1,1,self.nTG,1,1))
        return (sigD, irf), (t1, t2, tR, sigD)

    def configure_for_performance(self,ds):
        ds = ds.cache()
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=1)
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

    def benchmark(self, num_epochs=2):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            for i, sample in enumerate(self.val):
                # Performing a training step
                print(i)
                time.sleep(0.01)
        print("Execution time:", time.perf_counter() - start_time)


class IRFDeconv3D(tf.keras.layers.Layer):
    def __init__(self, x, y, t):
        super().__init__()
        self.x = x
        self.y = y
        self.t = t
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'x_shape': self.x,
            'y_shape': self.y,
            'time_bins': self.t,
        })
        return config

    def single_batch_deconv(self,tup):
        d, k = tup
        return tf.nn.conv3d_transpose(d,k,(1,self.x,self.y,self.t-1+self.t,1),strides=(1,1,1),padding='VALID')

    def call(self, input, *args, **kwargs):
        return tf.squeeze(tf.map_fn(
            self.single_batch_deconv, (tf.expand_dims(input[0],1),tf.expand_dims(input[1],-1)), dtype=tf.float32),axis=1)[:,:,:,:256,:]

def resblock_2D(num_filters, size_filter, x):
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(x)
    Fx = Activation(tf.nn.leaky_relu)(Fx)
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(Fx)
    output = add([Fx, x])
    output = Activation(tf.nn.leaky_relu)(output)
    return output

def resblock_2D_BN(nfilt_outer, nfilt_inner, size_filter, x, newstage=False):
    Fx = None
    if (newstage):
        Fx = Conv2D(nfilt_inner, 1, strides=2, activation=None)(x)
        x = Conv2D(nfilt_outer, 1, strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)    
    else:
        Fx = Conv2D(nfilt_inner, 1, activation=None)(x)
    Fx = BatchNormalization()(Fx)
    Fx = Activation(tf.nn.leaky_relu)(Fx)
    Fx = Conv2D(nfilt_inner, size_filter, padding='same', activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    Fx = Activation(tf.nn.leaky_relu)(Fx)
    Fx = Conv2D(nfilt_outer, 1, activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    Fx = Activation(tf.nn.leaky_relu)(Fx)
    output = add([Fx, x])
    return output

def resblock_3D_BN(nfilt_outer, nfilt_inner, size_filter, x, newstage=False):
    Fx = None
    if (newstage):
        Fx = Conv3D(nfilt_inner, (1,1,1), strides=(1,1,2), activation=None)(x)
        x = Conv3D(nfilt_outer, (1,1,1), strides=(1,1,2))(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)    
    else:
        Fx = Conv3D(nfilt_inner, (1,1,1), activation=None)(x)
    Fx = BatchNormalization()(Fx)
    Fx = Activation(tf.nn.leaky_relu)(Fx)
    Fx = Conv3D(nfilt_inner, size_filter, padding='same', activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    Fx = Activation(tf.nn.leaky_relu)(Fx)
    Fx = Conv3D(nfilt_outer, (1,1,1), activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    Fx = Activation(tf.nn.leaky_relu)(Fx)
    output = add([Fx, x])
    return output

def resblock_decode_3D(nfilt_outer, nfilt_inner, size_filter, x, newstage=False):
    Fx = None
    if (newstage):
        Fx = Conv3DTranspose(nfilt_inner, (1,1,1), strides=(1,1,2), activation=None)(x)
        x = Conv3DTranspose(nfilt_outer, (1,1,1), strides=(1,1,2))(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)    
    else:
        Fx = Conv3D(nfilt_inner, (1,1,1), activation=None) (x)
    Fx = BatchNormalization()(Fx)
    Fx = Activation(tf.nn.leaky_relu)(Fx)
    Fx = Conv3DTranspose(nfilt_inner, size_filter, padding='same', activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    Fx = Activation(tf.nn.leaky_relu)(Fx)
    Fx = Conv3DTranspose(nfilt_outer, (1,1,1), activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    Fx = Activation(tf.nn.leaky_relu)(Fx)
    output = add([Fx, x])
    return output

def xCeptionblock_2D_BN(num_filters, size_filter, x):
    Fx = SeparableConv2D(num_filters, size_filter, padding='same', activation=None)(x)
    Fx = BatchNormalization()(Fx)
    Fx = Activation(tf.nn.leaky_relu)(Fx)
    Fx = SeparableConv2D(num_filters, size_filter, padding='same', activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    output = add([Fx, x])
    output = Activation(tf.nn.leaky_relu)(output)
    return output