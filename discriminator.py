import tensorflow as tf
from ops import lkrelu, conv2d, downscale, minibatch_stddev_layer, fully_connect

# Discriminator class
class Discriminator(object):
    def __init__(self, inputs, par, PG, t, alpha_tra, is_training, reuse=None):
        with tf.variable_scope('D', initializer=tf.truncated_normal_initializer(stddev=par["std_init"]), reuse=reuse):
            self._par = par
            self._is_training = is_training
            self._PG = PG
            self._t = t
            self._alpha_tra = alpha_tra
            self._discriminator = self._build_discriminator(inputs, reuse=reuse)


    def _build_discriminator(self, inputs, reuse=None):
        def get_nf(stage, n_noise): return min(2 ** 3 * n_noise // 2 ** stage, n_noise) if not self._par["keep_channels_constant"] else n_noise
        if self._par["sort_minibatch"]:
            X = inputs[:, 0, 0, self._par["n_channel"]:self._par["n_channel"]+self._par["n_params"]]

        # from temperature: transition
        if self._t:
            conv_iden = downscale(inputs)
            conv_iden = lkrelu(conv2d(conv_iden, output_dim=get_nf(self._PG - 2, self._par["n_noise"]), kernel=[1, 1], strides=[1, 1], use_wscale=self._par["use_wscale"], name='dis_y_temp_conv_{}'.format(conv_iden.shape[1])))

        # from temperature
        conv = lkrelu(conv2d(inputs, output_dim=get_nf(self._PG - 1, self._par["n_noise"]), kernel=[1, 1], strides=[1, 1], use_wscale=self._par["use_wscale"], name='dis_y_temp_conv_{}'.format(inputs.shape[1])))

        for i in range(self._PG - 1):
            conv = lkrelu(conv2d(conv, output_dim=get_nf(self._PG - 1 - i, self._par["n_noise"]), kernel=[3, 3], strides=[1, 1], use_wscale=self._par["use_wscale"], padding="SAME", name='dis_n_conv_i_{}'.format(conv.shape[1])))
            conv = lkrelu(conv2d(conv, output_dim=get_nf(self._PG - 2 - i, self._par["n_noise"]), kernel=[3, 3], strides=[1, 1], use_wscale=self._par["use_wscale"], padding="SAME", name='dis_n_conv_ii_{}'.format(conv.shape[1])))
            conv = downscale(conv)
            if i == 0 and self._t:
                conv = (1 - self._alpha_tra) * conv_iden + self._alpha_tra * conv

        conv = minibatch_stddev_layer(conv, self._par["minibatch_group_size"], self._par["sort_minibatch"], X) if self._par["minibatch_std"] else tf.concat([conv, tf.zeros((tf.shape(conv)[0], 1, 8, 1))], -1)
        conv = lkrelu(conv2d(conv, output_dim=get_nf(1, self._par["n_noise"]), kernel=[1, 4], strides=[1, 1], use_wscale=self._par["use_wscale"], padding='VALID', name='dis_n_conv_i_{}'.format(conv.shape[1])))
        conv = lkrelu(conv2d(conv, output_dim=get_nf(1, self._par["n_noise"]), kernel=[1, 5], strides=[1, 1], use_wscale=self._par["use_wscale"], padding='VALID', name='dis_n_conv_ii_{}'.format(conv.shape[1])))
        conv = tf.reshape(conv, [-1, get_nf(1, self._par["n_noise"])])

        output = fully_connect(conv, output_size=1, use_wscale=self._par["use_wscale"], gain=1, name='dis_n_fully')

        if "WGAN" not in self._par["GAN_loss"]:
            output = tf.nn.sigmoid(output)

        return output

    @property
    def discriminator(self):
        return self._discriminator

    @discriminator.setter
    def discriminator(self, value):
        self._discriminator = value
