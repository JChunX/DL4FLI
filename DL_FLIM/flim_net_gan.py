import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import tensorflow_gan as tfgan

def _dense(inputs, units, l2_weight):
    return tf.layers.Dense(units)(inputs)

def _batch_norm(inputs, is_training):
    return tf.layers.batch_normalization(
        inputs, momentum=0.999, epsilon=0.001, training=is_training)

def _deconv1d(inputs, filters, kernel_size, stride, l2_weight):
    temp = tf.layers.Conv2DTranspose(
        filters, [kernel_size,1], strides=[stride,1], 
        activation=tf.nn.relu, padding='valid',
        kernel_initializer=tf.keras.initializers.glorot_uniform,
        kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
        bias_regularizer=tf.keras.regularizers.l2(l=l2_weight))(tf.expand_dims(inputs,axis=-2))
    return tf.squeeze(temp,axis=2)

def _conv1d(inputs, filters, kernel_size, stride, l2_weight):
    return tf.layers.Conv1D(
        filters, kernel_size, strides=stride, 
        activation=None, padding='valid',
        kernel_initializer=tf.keras.initializers.glorot_uniform,
        kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
        bias_regularizer=tf.keras.regularizers.l2(l=l2_weight))(inputs)

def _maxpool1d(inputs):
    return tf.layers.MaxPooling1D(2,strides=2)(inputs)

def _deconv2d(inputs, filters, kernel_size, stride, l2_weight):
    return tf.layers.conv2d_transpose(
        inputs, filters, kernel_size, strides=stride, 
        activation=tf.nn.relu, padding='valid',
        kernel_initializer=tf.keras.initializers.glorot_uniform,
        kernel_regularizer=tf.keras.regularizers.l2(l=l2_weight),
        bias_regularizer=tf.keras.regularizers.l2(l=l2_weight))

def reconvolution1d(input, kernel):

    def single_batch_reconv(tup):
        d, k = tup
        return tf.nn.conv1d(d,k/tf.reduce_max(k),stride=1,padding='SAME')

    def call(input, *args, **kwargs):
        return tf.squeeze(tf.map_fn(
            single_batch_reconv, (tf.expand_dims(input[0],1),tf.expand_dims(input[1],-1)), dtype=tf.float32),axis=1)

    return call((input,kernel))

def conditional_generator(inputs, weight_decay=2.5e-5, is_training=True):
    dk_lowcount, irf = inputs
    u_stage1 = _conv1d(dk_lowcount,64,3,1,weight_decay)
    u_stage1 = _batch_norm(u_stage1, is_training)
    u_stage1 = tf.nn.relu(u_stage1)
    u_stage1 = _conv1d(u_stage1,64,3,1,weight_decay)
    u_stage1 = _batch_norm(u_stage1, is_training)
    u_stage1 = tf.nn.relu(u_stage1)

    u_stage2 = _maxpool1d(u_stage1)
    u_stage2 = _conv1d(u_stage2,128,3,1,weight_decay)
    u_stage2 = _batch_norm(u_stage2, is_training)
    u_stage2 = tf.nn.relu(u_stage2)
    u_stage2 = _conv1d(u_stage2,128,3,1,weight_decay)
    u_stage2 = _batch_norm(u_stage2, is_training)
    u_stage2 = tf.nn.relu(u_stage2)

    u_stage3 = _maxpool1d(u_stage2)
    u_stage3 = _conv1d(u_stage3,256,3,1,weight_decay)
    u_stage3 = _batch_norm(u_stage3, is_training)
    u_stage3 = tf.nn.relu(u_stage3)
    u_stage3 = _conv1d(u_stage3,256,3,1,weight_decay)
    u_stage3 = _batch_norm(u_stage3, is_training)
    u_stage3 = tf.nn.relu(u_stage3)

    u_stage4 = _maxpool1d(u_stage3)
    u_stage4 = _conv1d(u_stage4,512,3,1,weight_decay)
    u_stage4 = _batch_norm(u_stage4, is_training)
    u_stage4 = tf.nn.relu(u_stage4)
    u_stage4 = _conv1d(u_stage4,512,3,1,weight_decay)
    u_stage4 = _batch_norm(u_stage4, is_training)
    u_stage4 = tf.nn.relu(u_stage4)

    u_stage5 = _maxpool1d(u_stage4)
    u_stage5 = _conv1d(u_stage5,1024,3,1,weight_decay)
    u_stage5 = _batch_norm(u_stage5, is_training)
    u_stage5 = tf.nn.relu(u_stage5)
    u_stage5 = _conv1d(u_stage5,1024,3,1,weight_decay)
    u_stage5 = _batch_norm(u_stage5, is_training)
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
    u_stage6 = _batch_norm(u_stage6, is_training)
    u_stage6 = tf.nn.relu(u_stage6)
    u_stage6 = _conv1d(u_stage6,512,3,1,weight_decay)
    u_stage6 = _batch_norm(u_stage6, is_training)
    u_stage6 = tf.nn.relu(u_stage6)

    u_stage7 = tf.concat([tf.keras.layers.Cropping1D((1,2))(_deconv1d(u_stage6,128,2,3,weight_decay)), u_stage3],-1)
    u_stage7 = _conv1d(u_stage7,256,3,1,weight_decay)
    u_stage7 = _batch_norm(u_stage7, is_training)
    u_stage7 = tf.nn.relu(u_stage7)
    u_stage7 = _conv1d(u_stage7,256,3,1,weight_decay)
    u_stage7 = _batch_norm(u_stage7, is_training)
    u_stage7 = tf.nn.relu(u_stage7)

    u_stage8 = tf.concat([tf.keras.layers.Cropping1D((19,18))(_deconv1d(u_stage7,128,2,3,weight_decay)),u_stage2],-1)
    u_stage8 = _conv1d(u_stage8,128,3,1,weight_decay)
    u_stage8 = _batch_norm(u_stage8, is_training)
    u_stage8 = tf.nn.relu(u_stage8)
    u_stage8 = _conv1d(u_stage8,128,3,1,weight_decay)
    u_stage8 = _batch_norm(u_stage8, is_training)
    u_stage8 = tf.nn.relu(u_stage8)

    u_stage9 = tf.concat([tf.keras.layers.Cropping1D(51)(_deconv1d(u_stage8,64,3,3,weight_decay)), u_stage1],-1)
    u_stage9 = _conv1d(u_stage9,64,3,1,weight_decay)
    u_stage9 = _batch_norm(u_stage9, is_training)
    u_stage9 = tf.nn.relu(u_stage9)
    u_stage9 = _conv1d(u_stage9,64,3,1,weight_decay)
    u_stage9 = _batch_norm(u_stage9, is_training)
    u_stage9 = tf.nn.relu(u_stage9)
    u_stage9 = _deconv1d(u_stage9,1,9,1,weight_decay)

    dk_super = reconvolution1d(u_stage9,irf)
    dk_super = _conv1d(dk_super,1,1,1,weight_decay)
    return dk_super

_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

def unconditional_critic(inputs, unused_conditioning, weight_decay=2.5e-5, is_training=True):
    net = _conv1d(inputs,64,15,7,weight_decay)
    net = _leaky_relu(net)
    net = _conv1d(net,128,4,2,weight_decay)
    net = _leaky_relu(net)
    net = _conv1d(net,128,4,2,weight_decay)
    net = _leaky_relu(net)

    net = tf.layers.flatten(net)
    print("-------------------------")
    print(net)
    net = _dense(net,128,weight_decay)
    net = _leaky_relu(net)

    net = _dense(net,1,weight_decay)
    return net

def get_eval_metric_ops_fn(gan_model):
    real_data_logits = tf.reduce_mean(gan_model.discriminator_real_outputs)
    gen_data_logits = tf.reduce_mean(gan_model.discriminator_gen_outputs)
    real_mnist_score = eval_util.mnist_score(gan_model.real_data)
    generated_mnist_score = eval_util.mnist_score(gan_model.generated_data)
    frechet_distance = eval_util.mnist_frechet_distance(
        gan_model.real_data, gan_model.generated_data)
    return {
        'real_data_logits': tf.metrics.mean(real_data_logits),
        'gen_data_logits': tf.metrics.mean(gen_data_logits),
        'real_mnist_score': tf.metrics.mean(real_mnist_score),
        'mnist_score': tf.metrics.mean(generated_mnist_score),
        'frechet_distance': tf.metrics.mean(frechet_distance),
    }