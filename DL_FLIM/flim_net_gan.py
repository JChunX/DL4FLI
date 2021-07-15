import tensorflow as tf
import tensorflow_gan as tfgan

"""
Components for GAN
"""

def _dense(inputs, units, l2_weight):
    return tf.keras.layers.Dense(units)(inputs)

def _batch_norm(inputs):
    return tf.keras.layers.BatchNormalization(
        momentum=0.999, epsilon=0.001)(inputs)

def _deconv1d(inputs, filters, kernel_size, stride, l2_weight):
    return tf.keras.layers.Conv1DTranspose(
        filters, kernel_size, strides=stride, 
        activation=tf.nn.relu, padding='valid',
        kernel_initializer=tf.keras.initializers.glorot_uniform,
        kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
        bias_regularizer=tf.keras.regularizers.l2(l=l2_weight))(inputs)

def _conv1d(inputs, filters, kernel_size, stride, l2_weight):
    return tf.keras.layers.Conv1D(
        filters, kernel_size, strides=stride, 
        activation=None, padding='valid',
        kernel_initializer=tf.keras.initializers.glorot_uniform,
        kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
        bias_regularizer=tf.keras.regularizers.l2(l=l2_weight))(inputs)

def _maxpool1d(inputs):
    return tf.keras.layers.MaxPooling1D(2,strides=2)(inputs)

"""

1D batchwise reconvolution
Given input of shape (batch, width, 1) and
kernal of shape (batch, width, 1)
Map batchwise s.t. conv1d receives (fakebatch=1,width,1) and (width,1,1)
Input is expanded to shape of (batch, 1, width, 1)
Kernal is expanded to shape of (batch, width, 1, 1)

"""
class IRFReconv1D(tf.keras.layers.Layer):
    def __init__(self,t):
        super().__init__()
        self.t = t

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'time_bins': self.t
        })
        return config

    def single_batch_reconv(self, tup):
        d, k = tup
        return tf.nn.conv1d(d,k/tf.reduce_max(k),stride=1,padding='SAME')

    def call(self, dk, irf, *args, **kwargs):
        return tf.squeeze(tf.map_fn(
            self.single_batch_reconv, 
            (tf.expand_dims(dk,1),tf.expand_dims(irf,-1)), 
            dtype=tf.float32),axis=1)

"""

Generator maps low-count decay to high-count decay
Uses modified U-net architecture
Input and output length is the same

"""
def conditional_generator(inputs, weight_decay=2.5e-5):
    dk_lowcount = inputs[0]
    irf = inputs[1]
    u_stage1 = _conv1d(dk_lowcount,64,3,1,weight_decay)
    u_stage1 = _batch_norm(u_stage1)
    u_stage1 = tf.nn.relu(u_stage1)
    u_stage1 = _conv1d(u_stage1,64,3,1,weight_decay)
    u_stage1 = _batch_norm(u_stage1)
    u_stage1 = tf.nn.relu(u_stage1)

    u_stage2 = _maxpool1d(u_stage1)
    u_stage2 = _conv1d(u_stage2,128,3,1,weight_decay)
    u_stage2 = _batch_norm(u_stage2)
    u_stage2 = tf.nn.relu(u_stage2)
    u_stage2 = _conv1d(u_stage2,128,3,1,weight_decay)
    u_stage2 = _batch_norm(u_stage2)
    u_stage2 = tf.nn.relu(u_stage2)

    u_stage3 = _maxpool1d(u_stage2)
    u_stage3 = _conv1d(u_stage3,256,3,1,weight_decay)
    u_stage3 = _batch_norm(u_stage3)
    u_stage3 = tf.nn.relu(u_stage3)
    u_stage3 = _conv1d(u_stage3,256,3,1,weight_decay)
    u_stage3 = _batch_norm(u_stage3)
    u_stage3 = tf.nn.relu(u_stage3)

    u_stage4 = _maxpool1d(u_stage3)
    u_stage4 = _conv1d(u_stage4,512,3,1,weight_decay)
    u_stage4 = _batch_norm(u_stage4)
    u_stage4 = tf.nn.relu(u_stage4)
    u_stage4 = _conv1d(u_stage4,512,3,1,weight_decay)
    u_stage4 = _batch_norm(u_stage4)
    u_stage4 = tf.nn.relu(u_stage4)

    u_stage5 = _maxpool1d(u_stage4)
    u_stage5 = _conv1d(u_stage5,1024,3,1,weight_decay)
    u_stage5 = _batch_norm(u_stage5)
    u_stage5 = tf.nn.relu(u_stage5)
    u_stage5 = _conv1d(u_stage5,1024,3,1,weight_decay)
    u_stage5 = _batch_norm(u_stage5)
    u_stage5 = tf.nn.relu(u_stage5)
    """
    Save for estimator training (lock other layers)
    lifetimes = _maxpool1d(u_stage5)
    lifetimes = tf.layers.flatten(lifetimes)
    t1 = _dense(lifetimes,1024,weight_decay)
    t1 = _batch_norm(t1,is_training)
    t1 = tf.nn.relu(t1)
    t2 = _dense(lifetimes,1024,weight_decay)
    t2 = _batch_norm(t2,is_training)
    t2 = tf.nn.relu(t2)
    rT = _dense(lifetimes,1024,weight_decay)
    rT = _batch_norm(rT,is_training)
    rT = tf.nn.relu(rT)
    """

    u_stage6 = tf.concat([_deconv1d(u_stage5,512,3,3,weight_decay), u_stage4],-1)
    u_stage6 = _conv1d(u_stage6,512,3,1,weight_decay)
    u_stage6 = _batch_norm(u_stage6)
    u_stage6 = tf.nn.relu(u_stage6)
    u_stage6 = _conv1d(u_stage6,512,3,1,weight_decay)
    u_stage6 = _batch_norm(u_stage6)
    u_stage6 = tf.nn.relu(u_stage6)

    u_stage7 = tf.concat([tf.keras.layers.Cropping1D((1,2))(_deconv1d(u_stage6,128,2,3,weight_decay)), u_stage3],-1)
    u_stage7 = _conv1d(u_stage7,256,3,1,weight_decay)
    u_stage7 = _batch_norm(u_stage7)
    u_stage7 = tf.nn.relu(u_stage7)
    u_stage7 = _conv1d(u_stage7,256,3,1,weight_decay)
    u_stage7 = _batch_norm(u_stage7)
    u_stage7 = tf.nn.relu(u_stage7)

    u_stage8 = tf.concat([tf.keras.layers.Cropping1D((19,18))(_deconv1d(u_stage7,128,2,3,weight_decay)),u_stage2],-1)
    u_stage8 = _conv1d(u_stage8,128,3,1,weight_decay)
    u_stage8 = _batch_norm(u_stage8)
    u_stage8 = tf.nn.relu(u_stage8)
    u_stage8 = _conv1d(u_stage8,128,3,1,weight_decay)
    u_stage8 = _batch_norm(u_stage8)
    u_stage8 = tf.nn.relu(u_stage8)

    u_stage9 = tf.concat([tf.keras.layers.Cropping1D(51)(_deconv1d(u_stage8,64,3,3,weight_decay)), u_stage1],-1)
    u_stage9 = _conv1d(u_stage9,64,3,1,weight_decay)
    u_stage9 = _batch_norm(u_stage9)
    u_stage9 = tf.nn.relu(u_stage9)
    u_stage9 = _conv1d(u_stage9,64,3,1,weight_decay)
    u_stage9 = _batch_norm(u_stage9)
    u_stage9 = tf.nn.relu(u_stage9)
    u_stage9 = _deconv1d(u_stage9,1,9,1,weight_decay)

    dk_super = IRFReconv1D(256)(u_stage9,irf)
    dk_super = _conv1d(dk_super,1,1,1,weight_decay)

    generative_model = tf.keras.Model(inputs=(dk_lowcount,irf), outputs=dk_super)
    return generative_model

"""

Critic maps high-count input and IRF and produces a score

"""

def conditional_critic(inputs, weight_decay=2.5e-5):
    dk_highcount = inputs[0]
    irf = inputs[1]
    inputs = tf.concat([dk_highcount,irf/tf.reduce_max(irf)],0)
    net = _conv1d(inputs,64,15,7,weight_decay)
    net = _leaky_relu(net)
    net = _conv1d(net,128,4,2,weight_decay)
    net = _leaky_relu(net)
    net = _conv1d(net,128,4,2,weight_decay)
    net = _leaky_relu(net)

    net = tf.keras.layers.Flatten()(net)
    net = _dense(net,128,weight_decay)
    net = _leaky_relu(net)

    net = _dense(net,1,weight_decay)
    critic = tf.keras.Model(inputs=(dk_highcount, irf), outputs=net)
    return critic
