import tensorflow as tf
from ops import lkrelu, pixel_norm, group_norm, instance_norm, conv2d, upscale, fully_connect

# Generator class
class Generator(object):
    def __init__(self, inputs, par, PG, t, alpha_tra, is_training, reuse=None):
        with tf.variable_scope('G', initializer=tf.truncated_normal_initializer(stddev=par["std_init"]), reuse=reuse):
            self._par = par
            self._is_training = is_training
            self._PG = PG
            self._t = t
            self._alpha_tra = alpha_tra
            self._decoder = self._build_decoder(inputs)

    # Build decoder
    def _build_decoder(self, inputs):
        def PN(x, axis=3):
            if self._par["pixel_norm"] == 0:
                return x
            elif self._par["pixel_norm"] == 1:
                return pixel_norm(x, axis=axis)
            elif self._par["pixel_norm"] == 2:
                return group_norm(x) if len(x.shape) == 4 else group_norm(tf.expand_dims(x, -1))
            elif self._par["pixel_norm"] == 3:
                return instance_norm(x) if len(x.shape) == 4 else instance_norm(tf.expand_dims(x, -1))

        def PN_init(x, axis=3):
            if self._par["pixel_norm_init"] == 0:
                return x
            elif self._par["pixel_norm_init"] == 1:
                return pixel_norm(x, axis=axis)

        def get_nf(stage, n_noise): return min(2 ** 3 * n_noise // 2 ** stage, n_noise) if not self._par["keep_channels_constant"] else n_noise

        with tf.variable_scope('decoder'):
            # Get base spatial resolution in one step using a fully connected layer
            de = tf.reshape(PN_init(inputs, axis=1), [-1, int(get_nf(1, self._par["n_noise"]))])
            de = fully_connect(de, output_size=8 * int(get_nf(1, self._par["n_noise"])), use_wscale=self._par["use_wscale"])
            de = tf.reshape(de, [-1, 1, 8, int(get_nf(1, self._par["n_noise"]))])
            de = PN(lkrelu(de))
            de = conv2d(de, output_dim=get_nf(1, self._par["n_noise"]), kernel=[1, 3], strides=[1, 1], name="gen_n_conv_ii")
            de = PN(lkrelu(de))

            for i in range(self._PG - 1):
                if i == self._PG - 2 and self._t:
                    # to temperature: transition
                    de_iden = conv2d(de, output_dim=self._par["n_channel"], kernel=[1, 1], strides=[1, 1], use_wscale=self._par["use_wscale"], name='gen_y_temp_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                de = PN(lkrelu(conv2d(de, output_dim=get_nf(i + 1, self._par["n_noise"]), kernel=[3, 3], strides=[1, 1], use_wscale=self._par["use_wscale"], padding="SAME", name='gen_n_conv_i_{}'.format(de.shape[1]))))
                de = PN(lkrelu(conv2d(de, output_dim=get_nf(i + 1, self._par["n_noise"]), kernel=[3, 3], strides=[1, 1], use_wscale=self._par["use_wscale"], padding="SAME", name='gen_n_conv_ii_{}'.format(de.shape[1]))))

            # to temperature
            de = conv2d(de, output_dim=self._par["n_channel"], kernel=[1, 1], strides=[1, 1], use_wscale=self._par["use_wscale"], gain=1, name='gen_y_temp_conv_{}'.format(de.shape[1]))

        # Return
        if self._PG == 1:
            return de
        if self._t:
            return (1 - self._alpha_tra) * de_iden + self._alpha_tra * de
        else:
            return de

    @property
    def decoder(self):
        return self._decoder

    @decoder.setter
    def decoder(self, value):
        self._decoder = value