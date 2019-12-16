"""
Operations for the 21cm GAN
"""
import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    mpl.use('Agg')
import numpy as np
import tensorflow as tf
from utils import get_shape, scale_pars, scale_pars_tf

# ########################## INPUT PIPELINE ##########################
def build_pipeline(par, PG=1):
    with tf.device('/cpu:0'):
        # Set up iterator
        filepath = os.path.join(par["folder_in"], "train.tfrecords_" + str(PG) + par["filename"])
        ds = tf.data.TFRecordDataset(filepath)
        ds = ds.map(lambda x: read_tfRecord_function(x, par["n_params"], [par["res_x"], par["res_z"]]), num_parallel_calls=4)
        if par["shuffle"]:
            ds = ds.shuffle(buffer_size=par["n_shuffle"])
        ds = ds.repeat(par["n_iter"])
        ds = ds.map(lambda im, params: parse_function_tf(im, params, par), num_parallel_calls=4)
        ds = ds.batch(par["n_batch"] * par["num_gpus"])
        ds = ds.prefetch(1)
        iterator = ds.make_one_shot_iterator()
        _data, _params = iterator.get_next()
        return _data, _params


def read_tfRecord_function(serialised, n_params, shape):
    # 1. Define a parser
    features = tf.parse_single_example(serialised,
        features={
            'params_raw': tf.VarLenFeature(tf.float32),
            'image_raw': tf.VarLenFeature(tf.float32),
        })

    # 2. Convert the data
    image = tf.sparse_tensor_to_dense(features['image_raw'], default_value=0)
    params = tf.sparse_tensor_to_dense(features['params_raw'], default_value=0)

    # 3. Reshape
    image.set_shape((shape[0] * shape[1]))
    image = tf.reshape(image, [shape[0], shape[1]])
    params.set_shape(n_params)
    return image, params


# PARSE FUNCTION FOR INPUT:
def parse_function_tf(im, params, par):
    if par["normalise_data_in"] is True:
        if par["legacy"]:
            data_trafo = lambda x: -1.0 + (x + 200.0) / 125.0
        else:
            # revert_saved_trafo = lambda x: (x - 3.0 / 5.0) * 125.0
            revert_saved_trafo = lambda x: x
            data_trafo = lambda x: revert_saved_trafo(x) / 175.0 + 2.0 / 5.0 + 0.075 * tf.asinh(0.5 * revert_saved_trafo(x))
            # data_trafo = lambda x: x / 175.0 + 2.0/5.0 + 0.075 * tf.asinh(0.5 * x)
        im = data_trafo(im)
    if par["mirror_vertical"]:
        im = tf.squeeze(tf.image.random_flip_up_down(tf.expand_dims(im, -1)), -1)

    if par["normalise_params"]:
        params = scale_pars_tf(tf.expand_dims(params, 0), mean=par["X_mean"], std=par["X_std"])[0]
    return im, tf.squeeze(params, 0)


def parse_function(im, params, par):
    if par["normalise_data_in"] is True:
        if par["legacy"]:
            data_trafo = lambda x: -1.0 + (x + 200.0) / 125.0
        else:
            # revert_saved_trafo = lambda x: (x - 3.0 / 5.0) * 125.0
            revert_saved_trafo = lambda x: x
            data_trafo = lambda x: revert_saved_trafo(x) / 175.0 + 2.0 / 5.0 + 0.075 * np.arcsinh(0.5 * revert_saved_trafo(x))
            # data_trafo = lambda x: x / 175.0 + 2.0 / 5.0 + 0.075 * np.arcsinh(0.5 * x)
        im = data_trafo(im)
    # if par["mirror_vertical"]:
    #     do_flip = np.random.uniform() > 0.5
    #     im = np.flipud(im) if do_flip else im

    if par["normalise_params"]:
        params = scale_pars(params, mean=par["X_mean"], std=par["X_std"])[0]
    return im, params

# ########################## NN operations ##########################
# WEIGHTS
# Get weights
def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None:
        fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init

    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))


# PADDING
def periodic_padding(tensor, axis, padding=1):
    """
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
    """
    if isinstance(axis, int):
        axis = (axis,)
    if isinstance(padding, int):
        padding = (padding,)

    ndim = len(tensor.shape)
    for ax, p in zip(axis, padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left

        ind_right = [slice(-p, None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = tf.concat([right, middle, left], axis=ax)

    return tensor


# CONVOLUTIONS
# 2D Convolution
def conv2d(x, output_dim, kernel, strides, gain=np.sqrt(2), use_wscale=False, padding="SAME", pad_vert=1, name="conv2d", with_w=False):
    with tf.variable_scope(name):
        w = get_weight([kernel[0], kernel[1], x.shape[-1].value, output_dim], gain=gain, use_wscale=use_wscale)
        w = tf.cast(w, x.dtype)

        if padding == 'asym':
            padding = 'VALID'
            # periodic padding in vert. dir.
            x = periodic_padding(x, axis=1, padding=pad_vert)
            # zero padding in hor. dir.
            x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [0, 0]], "CONSTANT")

        if padding == 'only_z':
            # zero padding in hor. dir.
            padding = 'VALID'
            x = tf.pad(x, [[0, 0], [0, 0], [3, 3], [0, 0]], "CONSTANT")

        elif padding == 'VALID':
            padding = 'VALID'

        conv = tf.nn.conv2d(x, w, strides=[1, strides[0], strides[1], 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        shape_out = conv._shape_as_list()
        if shape_out[0] is None: shape_out[0] = -1
        conv = tf.reshape(tf.nn.bias_add(conv, biases), shape_out)

        if with_w:
            return conv, w, biases
        else:
            return conv


# UPSCALING / DOWNSCALING
def resize_nearest_neighbor(x, new_size):
    return tf.image.resize_nearest_neighbor(x, new_size)

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def upscale(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h * scale, w * scale))

def downscale(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape


# FULLY CONNECTED LAYER
# FC layer
def fully_connect(x, output_size, gain=np.sqrt(2), use_wscale=False, name=None, with_w=False):
  shape = x.get_shape().as_list()
  with tf.variable_scope(name or "Linear"):
    w = get_weight([shape[1], output_size], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
    output = tf.matmul(x, w) + bias

    if with_w:
        return output, with_w, bias
    else:
        return output

# NORMALISATIONS
# Minibatch STD layer
def minibatch_stddev_layer(x, group_size=4, sort=[None], params=None):
    with tf.variable_scope('MinibatchStddev'):
        x = tf.transpose(x, [0, 3, 1, 2])                       # [NHWC] Input shape.
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        if sort[0] is not None:
            # Sort tensor according to parameter order as specified by sort
            priority_tensor = tf.slice(params, [0, sort[-1]], [-1, 1])
            for i_param in range(len(sort)-2, -1, -1):
                priority_tensor = tf.add(priority_tensor, tf.slice(params, [0, sort[i_param]], [-1, 1]) * 100 ** (len(sort) - 1 - i_param))  # note: for this to work, parameters must be normalised to a range much smaller than 100
            sort_ind = tf.reverse(tf.nn.top_k(priority_tensor[:, 0], k=tf.shape(params)[0], sorted=False).indices, axis=[0])
            y = tf.gather(x, sort_ind, axis=0)
        else:
            y = x
        y = tf.reshape(y, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.

        if sort[0] is not None:
            # Revert sorting
            reverse_ind = tf.reverse(tf.nn.top_k(sort_ind, k=tf.shape(sort_ind)[0], sorted=True).indices, axis=[0])
            y = tf.gather(y, reverse_ind, axis=0)

        conc = tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap
        return tf.transpose(conc, [0, 2, 3, 1])                 # [NHWC]  Output shape.

# Pixel norm
# NOTE: in original PGGAN implementation, they use NCHW! here: NHWC! 1 -> -1
def pixel_norm(x, epsilon=1e-8, axis=3):
    with tf.variable_scope('pixel_norm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True) + epsilon)

# Instance norm
def instance_norm(x, epsilon=1e-8):
    with tf.variable_scope('instance_norm'):
        x_shape = x._shape_as_list()
        if x_shape[0] is None: x_shape[0] = -1
        return x * tf.rsqrt(tf.expand_dims(tf.reduce_mean(tf.reshape(tf.square(x), [x_shape[0], x_shape[1] * x_shape[2], x_shape[3]]), axis=1, keepdims=True), 2) + epsilon)

# Group norm
def group_norm(x, G=32, eps=1e-5):
    with tf.variable_scope('{}_norm'.format("group")):
        # normalize
        # transpose: [bs, h, w, c] to [bs, c, h, w] following the paper
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        # per channel gamma and beta
        gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
        beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])

        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        # transpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])
    return output


# Batch normalisation
def batch_norm(*args, **kwargs):
    with tf.name_scope('bn'):
        # BATCH NORM
        bn = tf.layers.batch_normalization(*args, **kwargs)
    return bn


# ACTIVATIONS
# Leaky ReLU activation function
def lkrelu(x, slope=0.2):
    return tf.maximum(slope * x, x)


# Soft clipping activation function
def softclip(x, alpha=10):
    return 1. / alpha * tf.log((1. + tf.exp(alpha * x)) / (1. + tf.exp(alpha * (x - 1))))


# CONCATENATE PARAMETERS AND NOISE
# Create input tensor from given noise, consisting of one channel. The first n_param elements are the parameters.
def get_g_input(params, noise):
    return tf.expand_dims(tf.concat([params, noise[:, get_shape(params)[-1]:]], 1), -1)


# TENSOR STACKING
# Add parameter vector to the input for the discriminator, as additional channel
def stack_params(inputs, params):
    param_stretch_x, param_stretch_z = get_shape(inputs)[1], get_shape(inputs)[2]
    param_vec = tf.tile(tf.expand_dims(tf.expand_dims(params, 1), 2), [1, param_stretch_x, param_stretch_z, 1])
    return tf.concat((param_vec, inputs), axis=-1)


# MULTI-GPU:
# Average gradients
# Build the function to average the gradients
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# Assign variables to device
def assign_to_device(device, ps_device='/cpu:0'):
    PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign
