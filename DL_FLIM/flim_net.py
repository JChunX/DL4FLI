import tensorflow as tf
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
        return tf.nn.conv3d_transpose(d,k/tf.reduce_max(k),(1,self.x,self.y,self.t-1+self.t,1),strides=(1,1,1),padding='VALID')

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

if __name__ == '__main__':
    pass