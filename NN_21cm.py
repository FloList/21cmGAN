import os
import time
import numpy as np
import tensorflow as tf
from discriminator import Discriminator
from generator import Generator
from ops import stack_params, get_g_input, assign_to_device, average_gradients, scale_pars
from utils import count_trainable_vars, output_plot, output_plot_fixed, output_plot_interpolate, output_plot_avg, output_plot_PDF, output_plot_hist, output_plot_power, mkdir_p, undo_data_normalisation
import random
from time import gmtime, strftime
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom

# Loss functions
########################################################################################################################
eps = 1e-10
def GAN_loss(real, fake):
    g_loss = - tf.reduce_mean(tf.log(fake + eps))
    d_loss = - tf.reduce_mean(tf.log(real + eps) + tf.log(1.0 - fake + eps))
    return g_loss, d_loss

def LSGAN_loss(real, fake):
    g_loss = tf.reduce_mean(0.5*(tf.square(fake - 1 + eps)))
    d_loss = tf.reduce_mean(0.5*(tf.square(real - 1 + eps)) + 0.5*(tf.square(fake + eps)))
    return g_loss, d_loss

def WGAN_loss(real, fake):
    g_loss = - tf.reduce_mean(fake + eps)
    d_loss = - tf.reduce_mean(real + eps) + tf.reduce_mean(fake + eps)
    return g_loss, d_loss
########################################################################################################################

# Class for the 21cm PGGAN objects
class NN_21cm(object):
    def __init__(self, par, g_params, checkpt_dir_write, checkpt_dir_read, im_dir, d_inputs=None, t=False, PG=1, only_eval=False):
        """
        par: dictionary containing hyperparameters (see main.py)
        g_params: if data comes from TFRecords: one shot iterator that returns parameter vector
        checkpt_dir_write: directory to save checkpoints
        checkpt_dir_read: directory to load previous checkpoints from
        im_dir: directory to save output images
        d_inputs: if data comes from TFRecords: one shot iterator that returns a sample
        t: True if transitional layer, else False
        PG: progressive GAN stage
        only_eval: if True, only build model parts that are needed to create samples
        """
        with tf.device('/cpu:0'):
            self._t = t
            self._PG = PG
            self._par = par
            self._is_training = tf.placeholder(tf.bool)
            self._res_x = par["res_x"]  # current resolution in x (vert. dimension)
            self._res_z = par["res_z"]  # current resolution in z (hor. dimension, corresponding to redshift)
            self._res = [self._res_x, self._res_z]
            # If data comes from a numpy / HDF5 file: parameters are a placeholder
            if par["input_type"] < 2:
                self._g_params = tf.placeholder(tf.float32, [None, par["n_params"]])  # parameters
            # If data comes from TFRecords: parameters are an iterator
            else:
                self._g_params = g_params  # parameters
            self._g_noise = tf.placeholder(tf.float32, [None, par["n_noise"]])  # noise (the first "n_param" entries will be discarded and replaced by parameters!)
            self._alpha_tra = tf.Variable(initial_value=0.0, trainable=False, name='alpha_tra')  # alpha for PGGAN transition
            self._checkpt_dir_read = checkpt_dir_read
            self._im_dir = im_dir
            if not only_eval:
                # If data comes from a numpy / HDF5 file: data are a placeholder
                if par["input_type"] < 2:
                    self._d_inputs = tf.placeholder(tf.float32, [None, self._res_x, self._res_z, par["n_channel"]])  # real sample
                # If data comes from TFRecords: data are an iterator
                else:
                    # If transitional stage: d_inputs need to be coarsened! For numpy / HDF5 data, this happens in the train method
                    if t and PG != 0:
                        alpha = self._alpha_tra
                        half_size = tf.stack([self._res_x // 2, self._res_z // 2], axis=0)
                        y_low_res = tf.image.resize_images(d_inputs, half_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
                        y_low_res = tf.image.resize_images(y_low_res, tf.shape(d_inputs)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
                        d_inputs = (1 - alpha) * y_low_res + alpha * d_inputs
                    self._d_inputs = d_inputs  # real sample
                self._gst = tf.train.get_or_create_global_step()
                self._checkpt_dir_write = checkpt_dir_write

            # Define optimisers
            if not only_eval:
                if self._par["GAN_loss"] is "WGAN":
                    self._d_opt = tf.train.RMSPropOptimizer(self._par["lr_disc"])
                else:
                    self._d_opt = tf.train.AdamOptimizer(self._par["lr_disc"], beta1=self._par["beta1"],
                                                         beta2=self._par["beta2"])

                if self._par["GAN_loss"] is "WGAN":
                    self._g_opt = tf.train.RMSPropOptimizer(self._par["lr_gen"])
                else:
                    self._g_opt = tf.train.AdamOptimizer(self._par["lr_gen"], beta1=self._par["beta1"],
                                                         beta2=self._par["beta2"])

            # Build model on each GPU
            tower_grads_g = []
            tower_grads_d = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i_gpu in range(max(par["num_gpus"], 1)):
                    ps_device = '/cpu:0'
                    if par["CPU"]:
                        device = '/cpu:0'
                    else:
                        device = assign_to_device('/gpu:{}'.format(i_gpu), ps_device=ps_device)

                    with tf.device(device):
                        # Get samples for GPU
                        n_batch_global = tf.shape(self._g_params)[0]
                        n_batch_gpu = tf.maximum(n_batch_global // par["num_gpus"], 1)  # require minimum 1 per batch (necessary when called by CPU to get one output)
                        g_params_gpu = self._g_params[i_gpu * n_batch_gpu: (i_gpu+1) * n_batch_gpu, :]
                        g_noise_gpu = self._g_noise[i_gpu * n_batch_gpu: (i_gpu+1) * n_batch_gpu, :]

                        # Get input for generator by concatenating parameters with noise and build generator
                        gen_input_with_noise = get_g_input(g_params_gpu, g_noise_gpu)  # conditional GAN
                        # gen_input_with_noise = tf.expand_dims(self._g_noise, -1)  # unconditional GAN
                        g = Generator(gen_input_with_noise, par, PG, t, self._alpha_tra, self._is_training)
                        if i_gpu == 0:
                            self._g = g

                        if only_eval:
                            # Get generator variables
                            g_vars = tf.trainable_variables(scope="G")
                            saver_eval = tf.train.Saver(g_vars)
                            if i_gpu == 0:
                                self._g_vars, self._saver_eval = g_vars, saver_eval

                        if not only_eval:
                            # Build discriminator
                            d_inputs_gpu = self._d_inputs[i_gpu * n_batch_gpu: (i_gpu + 1) * n_batch_gpu, :, :, :]
                            real_d_input = d_inputs_gpu
                            fake_d_input = g.decoder

                            real_d_input_w_param = stack_params(real_d_input, params=g_params_gpu)
                            fake_d_input_w_param = stack_params(fake_d_input, params=g_params_gpu)

                            # unconditional GAN
                            # self._real_d_input_w_param = real_d_input
                            # self._fake_d_input_w_param = fake_d_input

                            # Stack x-mean and difference towards x-mean as additional channels
                            if par["disc_add_channels"]:
                                x_mean_tiled_real = tf.tile(tf.reduce_mean(real_d_input_w_param[:, :, :, :1], axis=1, keepdims=True), [1, real_d_input_w_param._shape_tuple()[1], 1, 1])
                                x_var_real = real_d_input_w_param[:, :, :, :1] - x_mean_tiled_real
                                real_d_input_w_param = tf.concat([real_d_input_w_param, x_mean_tiled_real, x_var_real], axis=3)

                                x_mean_tiled_fake = tf.tile(tf.reduce_mean(fake_d_input_w_param[:, :, :, :1], axis=1, keepdims=True), [1, fake_d_input_w_param._shape_tuple()[1], 1, 1])
                                x_var_fake = fake_d_input_w_param[:, :, :, :1] - x_mean_tiled_fake
                                fake_d_input_w_param = tf.concat([fake_d_input_w_param, x_mean_tiled_fake, x_var_fake], axis=3)

                            # Discriminate
                            real_d = Discriminator(real_d_input_w_param, par, PG, t, self._alpha_tra, self._is_training)
                            fake_d = Discriminator(fake_d_input_w_param, par, PG, t, self._alpha_tra, self._is_training, reuse=True)
                            if i_gpu == 0:
                                self._real_d, self._fake_d = real_d, fake_d

                            # Loss
                            if par["GAN_loss"] is "GAN":
                                g_loss, d_loss = GAN_loss(real_d.discriminator, fake_d.discriminator)
                            elif par["GAN_loss"] is "LSGAN":
                                g_loss, d_loss = LSGAN_loss(real_d.discriminator, fake_d.discriminator)
                            elif par["GAN_loss"] is "WGAN":
                                g_loss, d_loss = WGAN_loss(real_d.discriminator, fake_d.discriminator)
                                _ = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')]
                            elif par["GAN_loss"] is "WGANGP":
                                g_loss, d_loss = WGAN_loss(real_d.discriminator, fake_d.discriminator)
                                # Add gradient penalty
                                alpha = tf.random_uniform(shape=tf.concat([tf.shape(d_inputs_gpu)[0:1], tf.TensorShape([1, 1, 1])], axis=0), minval=0., maxval=1.)
                                differences = g.decoder - d_inputs_gpu
                                interpolates = d_inputs_gpu + (alpha * differences)
                                interpolates_par = stack_params(interpolates, params=g_params_gpu)
                                # Stack x-mean and difference towards x-mean as additional channels
                                if par["disc_add_channels"]:
                                    interpolates_par_mean_tiled = tf.tile(tf.reduce_mean(interpolates_par[:, :, :, :1], axis=1, keepdims=True), [1, interpolates_par._shape_tuple()[1], 1, 1])
                                    interpolates_par_var = interpolates_par_mean_tiled[:, :, :, :1] - interpolates_par_mean_tiled
                                    interpolates_par = tf.concat([interpolates_par, interpolates_par_mean_tiled, interpolates_par_var], axis=3)
                                D_star = Discriminator(interpolates_par, par, PG, t, self._alpha_tra, self._is_training, reuse=True)
                                gradients = tf.gradients(D_star.discriminator, [interpolates])[0]
                                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
                                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                                d_loss += par["lambda_GP"] * gradient_penalty  # add gradient penalty
                                if i_gpu == 0:
                                    tf.summary.scalar("GP_loss", gradient_penalty)
                                drift_penalty = tf.reduce_mean(tf.square(real_d.discriminator))
                                d_loss += par["lambda_eps"] * drift_penalty  # keep discriminator output close to 0 with drift penalty
                                if i_gpu == 0:
                                    tf.summary.scalar("drift_penalty", drift_penalty)
                            else:
                                raise NotImplementedError


                            # Get trainable variables in generator / discriminator
                            g_vars = tf.trainable_variables(scope="G")
                            d_vars = tf.trainable_variables(scope="D")
                            if i_gpu == 0:
                                self._g_vars = g_vars
                                print("Trainable variables of generator:")
                                count_trainable_vars(self._g_vars)
                                self._d_vars = d_vars
                                print("Trainable variables of discriminator:")
                                count_trainable_vars(self._d_vars)

                            # All variables for inner layers
                            d_vars_n = [var for var in d_vars if 'dis_n' in var.name]
                            g_vars_n = [var for var in g_vars if 'gen_n' in var.name]

                            # Old variables for inner layers that remain
                            d_vars_n_read = [var for var in d_vars_n if '{}'.format(self._res_x) not in var.name]
                            g_vars_n_read = [var for var in g_vars_n if '{}'.format(self._res_x) not in var.name]

                            # All "To temperature" variables
                            d_vars_n_2 = [var for var in d_vars if 'dis_y_temp_conv' in var.name]
                            g_vars_n_2 = [var for var in g_vars if 'gen_y_temp_conv' in var.name]

                            # "To temperature" variables that are transitioning
                            d_vars_n_2_temp = [var for var in d_vars_n_2 if '{}'.format(self._res_x) not in var.name]
                            g_vars_n_2_temp = [var for var in g_vars_n_2 if '{}'.format(self._res_x) not in var.name]

                            # Store variables
                            if i_gpu == 0:
                                self._g_vars_n, self._g_vars_n_read, self._g_vars_n_2, self._g_vars_n_2_temp = g_vars_n, g_vars_n_read, g_vars_n_2, g_vars_n_2_temp
                                self._d_vars_n, self._d_vars_n_read, self._d_vars_n_2, self._d_vars_n_2_temp = d_vars_n, d_vars_n_read, d_vars_n_2, d_vars_n_2_temp
                                print("g_vars:", len(self._g_vars))
                                print("d_vars:", len(self._d_vars))
                                print("g_vars_n_read:", len(self._g_vars_n_read))
                                print("d_vars_n_read:", len(self._d_vars_n_read))
                                print("g_vars_n_2_temp:", len(self._g_vars_n_2_temp))
                                print("d_vars_n_2_temp:", len(self._d_vars_n_2_temp))

                                self._g_loss = g_loss
                                self._d_loss = d_loss
                                tf.summary.scalar("generator_loss", self._g_loss)
                                tf.summary.scalar("discriminator_loss", self._d_loss)

                            # Calculate gradients for this tower
                            grads_g = self._g_opt.compute_gradients(g_loss, var_list=g_vars)
                            grads_d = self._d_opt.compute_gradients(d_loss, var_list=d_vars)

                            # Keep track of the gradients across all towers
                            tower_grads_g.append(grads_g)
                            tower_grads_d.append(grads_d)

                            # Reuse variables for the next tower
                            tf.get_variable_scope().reuse_variables()

            # We must calculate the mean of each gradient. Note that this is the synchronization point across all towers
            if not only_eval:
                self._g_grad = average_gradients(tower_grads_g)
                self._d_grad = average_gradients(tower_grads_d)

                # Make summaries
                tf.summary.image("N_body_image", self._d_inputs)
                tf.summary.image("NN_image", self._g.decoder)
                self._merged = tf.summary.merge_all()

                # Apply gradients
                self._g_train_step = self._g_opt.apply_gradients(self._g_grad, global_step=self._gst)
                self._d_train_step = self._d_opt.apply_gradients(self._d_grad)

                # Save variables
                self._saver = tf.train.Saver(self._d_vars + self._g_vars + [self._gst], max_to_keep=4)
                self._saver_read = tf.train.Saver(self._d_vars_n_read + self._g_vars_n_read + [self._gst], max_to_keep=4)
                if len(self._d_vars_n_2_temp + self._g_vars_n_2_temp):
                    self._saver_temp = tf.train.Saver(self._d_vars_n_2_temp + self._g_vars_n_2_temp + [self._gst], max_to_keep=4)


    def train(self, data, params, i_0=0):
        """
        Train the PGGAN.
        data: if data comes from TFRecords: one shot iterator, else numpy array
        params: if data comes from TFRecords: one shot iterator, else numpy array
        i_0: initial iteration within this stage (>0 when resuming training)
        """
        with tf.device('/cpu:0'):
            # Determine transition alpha
            train_step = tf.placeholder(tf.float32, shape=None)
            alpha_tra_assign = self._alpha_tra.assign(train_step / self._par["n_iter"])

            # Set up logging
            tf.logging.set_verbosity(tf.logging.DEBUG)

            # Global variables initialiser
            init = tf.global_variables_initializer()
            log_device_placement = True
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device_placement)
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                train_writer = tf.summary.FileWriter(os.path.join(self._checkpt_dir_write, self._par["logdir"]), sess.graph)
                sess.run(init)

                # Reload from previous stages
                if os.path.exists(self._checkpt_dir_read):
                    if tf.train.latest_checkpoint(self._checkpt_dir_read) is not None:
                        tf.logging.info('Loading checkpoint from ' + tf.train.latest_checkpoint(self._checkpt_dir_read))
                        if self._t and not self._par["resume_training"]:
                            self._saver_read.restore(sess, tf.train.latest_checkpoint(self._checkpt_dir_read))
                            self._saver_temp.restore(sess, tf.train.latest_checkpoint(self._checkpt_dir_read))
                        else:
                            self._saver.restore(sess, tf.train.latest_checkpoint(self._checkpt_dir_read))
                    else:
                        if self._PG > 1 and not self._par["no_grow"]: raise FileNotFoundError
                        tf.logging.info('Training from scratch - no checkpoint found')
                else:
                    if self._PG > 1 and not self._par["no_grow"]: raise FileNotFoundError
                    tf.logging.info('Training from scratch - no checkpoint found')

                i = i_0
                batch_no = 0
                times = []

                # Feed dict is only needed for data from numpy / HDF5 array, except if fixed param batch shall be provided
                use_feeddict = (self._par["input_type"] < 2)

                # Initialisation of fixed param batch
                if self._par["fixed_param_batch"]:
                    i_same = 0
                    X_same_par = np.zeros(self._par["n_params"])
                    Y_same_par = np.zeros((self._par["n_batch"] * self._par["num_gpus"], self._par["res_x"], self._par["res_z"]))
                else:
                    i_same = -np.infty

                # Start training
                while i < self._par["n_iter"]:
                    start = time.time()
                    for j in range(min(self._par["n_disc"], 1)):
                        noise = np.random.normal(size=[self._par["n_batch"] * self._par["num_gpus"], self._par["n_noise"]])
                        # For numpy / HDF5 data: get current batch and coarsen if needed
                        if self._par["input_type"] < 2:
                            curr_ind = range(batch_no * self._par["n_batch"] * self._par["num_gpus"], (batch_no + 1) * self._par["n_batch"] * self._par["num_gpus"])
                            X = params.take(curr_ind, mode="wrap", axis=0)
                            Y = data.take(curr_ind, mode="wrap", axis=0)
                            # If data from numpy: resize to resolution
                            if self._par["input_type"] is 0:
                                Y = np.asarray([resize(Y[i, :, :], output_shape=[self._res_x, self._res_z], mode="constant", anti_aliasing=False) for i in range(Y.shape[0])])
                            if self._t and self._PG != 0:
                                alpha = np.float(i + 1) / self._par["n_iter"]
                                Y_low_res = zoom(Y, zoom=[1, 0.5, 0.5], mode="nearest")
                                Y_low_res = zoom(Y_low_res, zoom=[1, 2, 2], mode="nearest")
                                Y = (1 - alpha) * Y_low_res + alpha * Y
                        # For TFRecord data: get current batch
                        else:
                            Y, X = sess.run([data, params])

                        # Fixed parameter batch
                        if self._par["fixed_param_batch"]:
                            if i_same is 0:
                                X_same_par = X[0, :]
                                Y_same_par[0, :] = Y[0, :]
                                i_same += 1

                            else:
                                same_param = np.where([np.allclose(X[i, :], X_same_par) for i in range(self._par["n_batch"] * self._par["num_gpus"])])[0]
                                if len(same_param) > 0:
                                    max_ind = min(self._par["n_batch"] * self._par["num_gpus"] - i_same, len(same_param))
                                    Y_same_par[i_same:i_same+max_ind] = Y[same_param[0:max_ind]]
                                    i_same += len(same_param)

                            if i_same >= self._par["n_batch"] * self._par["num_gpus"]:
                                use_feeddict = True
                                X = np.tile(np.expand_dims(X_same_par, 0), [self._par["n_batch"] * self._par["num_gpus"], 1])
                                Y = Y_same_par[:self._par["n_batch"] * self._par["num_gpus"]]
                                i_same = 0
                                X_same_par = np.zeros(self._par["n_params"])
                                Y_same_par = np.zeros((self._par["n_batch"] * self._par["num_gpus"], self._par["res_x"], self._par["res_z"]))

                        # Define feed_dict
                        if use_feeddict:
                            feed_dict = {self._g_noise: noise, self._is_training: True, self._g_params: X, self._d_inputs: np.expand_dims(Y, -1)}
                        else:
                            feed_dict = {self._g_noise: noise, self._is_training: True}
                        use_feeddict = (self._par["input_type"] < 2)  # reset, in case data comes from TFRecord (feed_dict is only used for fixed param batch)

                        # Optimise D
                        sess.run(self._d_train_step, feed_dict=feed_dict)
                        batch_no += 1

                    # Optimise G
                    sess.run(self._g_train_step, feed_dict=feed_dict)


                    # Logging info
                    gst = sess.run(self._gst, feed_dict=feed_dict)
                    times.append(time.time() - start)
                    summary = sess.run(self._merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, gst)
                    train_writer.flush()

                    # Assign alpha for fading-in
                    sess.run(alpha_tra_assign, feed_dict={train_step: i + 1})

                    # Output image and loss
                    if gst % self._par["image_every"] == 0 or i_same is 0:
                        g_loss_curr, d_loss_curr, alpha_tra = sess.run([self._g_loss, self._d_loss, self._alpha_tra], feed_dict=feed_dict)
                        alpha_disp = str(alpha_tra) if self._t else "-"
                        tf.logging.info(" PG " + str(self._PG) + " " + str(self._t) + ", step: " + str(gst) + ", G-Loss " + str(g_loss_curr) + " | D-Loss " + str(d_loss_curr) + ", alpha = " + alpha_disp + ", time for step " + str(times[-1]))
                        if i_same is 0:
                            tf.logging.info("      Fixed parameter batch!")
                    if gst % self._par["image_every"] == 0:
                        im_filename = self._im_dir + '/iter_%d.pdf' % gst
                        im_filename_fixed = self._im_dir + '/iter_%d_fixed.pdf' % gst
                        im_filename_interp = self._im_dir + '/iter_%d_interp.pdf' % gst
                        im_filename_avg = self._im_dir + '/iter_%d_avg.pdf' % gst
                        output_plot(X, Y, sess, self, 7, self._par, im_filename, is_training=True, X_mean=self._par["X_mean"], X_std=self._par["X_std"])
                        output_plot_fixed(X, Y, sess, self, 7, self._par, im_filename_fixed, params=None, is_training=True, X_mean=self._par["X_mean"], X_std=self._par["X_std"], show_real_data=False)
                        output_plot_interpolate(X, sess, self, 4, self._par, im_filename_interp, params=None, is_training=True, X_mean=self._par["X_mean"], X_std=self._par["X_std"])
                        output_plot_avg(X, Y, sess, self, 6, self._par, im_filename_avg, p=None, is_training=True, X_mean=self._par["X_mean"], X_std=self._par["X_std"])

                    # Save the model
                    if gst % self._par["save_every"] == 0:
                        save_path = self._saver.save(sess, os.path.join(self._checkpt_dir_write, self._par["checkpt"]), global_step=gst)
                        print("Model saved in file: %s" % save_path)

                    # Increase step
                    i += 1

                train_writer.close()
            tf.reset_default_graph()


    def sample_generator(self, sess, g_params, noise, is_training=False):
        return sess.run(self._g.decoder, feed_dict={self._g_params: g_params, self._g_noise: noise, self._is_training: is_training})

    def run_analysis(self, sess, X, Y, analysis, X_mean=None, X_std=None, X_model=None, Y_model=None):
        # Output plot with random samples
        if analysis["do_output"]:
            output_dir = self._im_dir + '/Random_samples'
            mkdir_p(output_dir)
            im_filename = output_dir + '/random_samples.pdf'
            output_plot(X, Y, sess, self, len(analysis["output"]), self._par, im_filename, p=analysis["output"], is_training=False, X_mean=X_mean, X_std=X_std, norm_data=self._par["normalise_data_out"])

        # Output plots for fixed sets of parameters
        if analysis["do_fixed"]:
            fixed_dir = self._im_dir + '/Fixed_parameters'
            mkdir_p(fixed_dir)
            for i in range(analysis["fixed"].shape[0]):
                im_filename_fixed = fixed_dir + '/%g_%g_%g.pdf' % (analysis["fixed"][i, 0], analysis["fixed"][i, 1], analysis["fixed"][i, 2])
                output_plot_fixed(X, Y, sess, self, analysis["n_fixed"], self._par, im_filename_fixed, params=analysis["fixed"][i, :], is_training=False, X_mean=X_mean, X_std=X_std, norm_data=self._par["normalise_data_out"])

        # Output plot: mean
        if analysis["do_avg"]:
            avg_dir = self._im_dir + '/Averages'
            mkdir_p(avg_dir)
            im_filename_avg = avg_dir + '/average.pdf'
            output_plot_avg(X, Y, sess, self, len(analysis["avg"]), self._par, im_filename_avg, n_sample=analysis["n_avg"], p=analysis["avg"], is_training=False, X_mean=X_mean, X_std=X_std, all_in_one=True, cols=analysis["cols_avg"], norm_data=self._par["normalise_data_out"], X_model=X_model, Y_model=Y_model)

        # Interpolate in parameter space
        if analysis["do_interp"]:
            interp_dir = self._im_dir + '/Interpolation'
            mkdir_p(interp_dir)
            if analysis["interp_all"]:
                im_filename_interp = interp_dir + '/interpolation.pdf'
                output_plot_interpolate(X, sess, self, analysis["n_interp"], self._par, im_filename_interp, params=analysis["interp"], is_training=False, X_mean=X_mean, X_std=X_std)
            else:
                for i in range(analysis["interp"].shape[0]):
                    im_filename_interp = interp_dir + '/%g_%g_%g_to_%g_%g_%g.pdf' \
                                         % (analysis["interp"][i, 0, 0], analysis["interp"][i, 0, 1], analysis["interp"][i, 0, 2],
                                            analysis["interp"][i, 1, 0], analysis["interp"][i, 1, 1], analysis["interp"][i, 1, 2])

                    output_plot_interpolate(X, sess, self, analysis["n_interp"], self._par, im_filename_interp, params=analysis["interp"][i, :, :], is_training=False, X_mean=X_mean, X_std=X_std)

        # Output plot: histogram
        if analysis["do_hist"]:
            hist_dir = self._im_dir + '/Histograms'
            mkdir_p(hist_dir)
            im_filename_hist = hist_dir + '/hist.pdf'
            output_plot_hist(X, Y, sess, self, len(analysis["hist"]), self._par, im_filename_hist, redshifts=analysis["hist_z"], n_sample=analysis["n_hist"], p=analysis["hist"], is_training=False, X_mean=X_mean, X_std=X_std, cols=analysis["cols_hist"], norm_data=self._par["normalise_data_out"], X_model=X_model, Y_model=Y_model)

        # Point distribution function
        if analysis["do_PDF"]:
            PDF_dir = self._im_dir + '/PDFs'
            mkdir_p(PDF_dir)
            im_filename_PDF = PDF_dir + '/PDFs.pdf'
            output_plot_PDF(X, Y, sess, self, self._par, [self._res_x, self._res_z], im_filename_PDF, n_sample=analysis["n_PDF"], params=analysis["PDF"], is_training=False, X_mean=X_mean, X_std=X_std, norm_data=self._par["normalise_data_out"], X_model=X_model, Y_model=Y_model)

        # Power spectrum
        if analysis["do_power"]:
            power_dir = self._im_dir + '/Power'
            mkdir_p(power_dir)
            im_filename_power = power_dir + '/PS.pdf'
            output_plot_power(X, Y, sess, self, len(analysis["power"]), self._par, im_filename_power, redshifts=analysis["power_z"], n_sample=analysis["n_power"], p=analysis["power"], is_training=False, X_mean=X_mean, X_std=X_std, cols=analysis["cols_power"], norm_data=self._par["normalise_data_out"], X_model=X_model, Y_model=Y_model)

    def run_abc(self, sess, X, Y, abc, X_mean=None, X_std=None):
        # Set distance function
        dist_fun = abc["dist_fun"]

        # Get an image for inference
        if self._par["normalise_params"]:
            theta_truth = scale_pars(abc["theta_truth"], X_mean, X_std)[0]
        else:
            theta_truth = abc["theta_truth"]
        theta_in_X = np.where([np.allclose(X[j, :], theta_truth) for j in range(X.shape[0])])[0]
        which_sample = 0 if abc["take_first_sample"] else np.random.choice(len(theta_in_X))
        Y_truth = np.expand_dims(Y[theta_in_X[which_sample]], 0)

        # Define limits of uniform priors
        prior_lim = scale_pars(abc["prior_lim"], X_mean, X_std)[0] if self._par["normalise_params"] else abc["prior_lim"]

        # Define simulator
        # params is a matrix: each row corresponds to one sample and contains the 3 parameters
        def simulator(params):
            if len(np.shape(params)) is 1:
                params = np.expand_dims(params, 0)
            noise = np.random.normal(0, 1.0, [params.shape[0], self._par["n_noise"]])
            return self.sample_generator(sess, params, noise, is_training=False)[:, :, :, 0]

        # Print some distances
        theta_test = scale_pars(abc["theta_test"], X_mean, X_std)[0] if self._par["normalise_params"] else abc["theta_test"]
        model_test = undo_data_normalisation(self._par, simulator(theta_test))
        model_truth = undo_data_normalisation(self._par, simulator(theta_truth))

        if abc["dist_fun_critic"]:
            noise = np.zeros([model_test.shape[0], self._par["n_noise"]])
            test_dist = dist_fun(sess, self, noise=noise, params=theta_test, d_inputs=np.tile(np.expand_dims(Y_truth, -1), [model_test.shape[0], 1, 1, 1])).squeeze(-1)
            real_dist = dist_fun(sess, self, noise=noise[0:1], params=theta_truth, d_inputs=np.expand_dims(Y_truth, -1)).squeeze(-1)
        else:
            test_dist = dist_fun(model_test, Y_truth)
            real_dist = dist_fun(model_truth, Y_truth)
        print("Example distance for correct theta:", real_dist)
        print("Example distances for some choices of theta:", test_dist)

        # Show sample
        # plt.imshow(np.squeeze(Y_truth, 0), vmin=0, vmax=1, cmap=get_turbo())
        # plt.show()

        # Rejection sampling function
        def rejection_sampling(eps_rej=None, n_accept=128, n_samples=1024, n_batch=128, always_save=None):
            # Initialise rejection sampling
            i_sim = 0
            n_found = 0
            abc_samples = np.zeros((0, self._par["n_params"]))
            abc_dist = np.zeros((0, abc["n_metrics"]))
            abc_im = np.zeros((0, self._par["res_x"], self._par["res_z"]))

            # Sample
            while n_found < n_samples:
                # Draw parameters from prior distributions (use random toolbox instead of np because of multiprocessing)
                fx_ = [random.uniform(prior_lim[0, 0], prior_lim[1, 0]) for _ in range(n_batch)]
                rhs_ = [random.uniform(prior_lim[0, 1], prior_lim[1, 1]) for _ in range(n_batch)]
                fa_ = [random.uniform(prior_lim[0, 2], prior_lim[1, 2]) for _ in range(n_batch)]
                params_ = np.vstack([fx_, rhs_, fa_]).T

                # Calculate distances
                if abc["dist_fun_critic"]:
                    noise_ = np.zeros([n_batch, self._par["n_noise"]])
                    dist_ = dist_fun(sess, self, noise=noise_, params=params_, d_inputs=np.tile(np.expand_dims(Y_truth, -1), [n_batch, 1, 1, 1])).squeeze(-1)
                    out_ = None
                else:
                    # Run simulation
                    out_ = undo_data_normalisation(self._par, simulator(params_))
                    dist_ = np.asarray(list(dist_fun(out_, Y_truth)))
                    if len(dist_.shape) is 1:
                        dist_ = np.expand_dims(dist_, 0)

                # Determine accepted samples
                # eps-based:
                if eps_rej is not None:
                    acc_samples_ = np.where(dist_[0, :] < eps_rej)[0]
                elif n_accept is not None:
                    dist_sort_ind_ = np.argsort(dist_[0, :])  # if several metrics shall be evaluated: take the first one
                    acc_samples_ = dist_sort_ind_[:n_accept]
                else:
                    raise NotImplementedError

                n_found += len(acc_samples_)
                abc_samples = np.concatenate([abc_samples, params_[acc_samples_].tolist()], axis=0)
                abc_dist = np.concatenate([abc_dist, np.asarray(dist_[:, acc_samples_]).T], axis=0)

                if not abc["dist_fun_critic"]:
                    abc_im = np.concatenate([abc_im, out_[acc_samples_].tolist()], axis=0)

                # Output:
                print("Simulation %i:   %i / %i samples were accepted (%g). Total accepted: %i / %i." %
                      (i_sim, len(acc_samples_), n_batch, len(acc_samples_) / n_batch, n_found, n_samples))

                i_sim += 1

                if always_save is not None:
                    np.savez(os.path.join(always_save, "data_" + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + str(np.round(random.uniform(0, 1e10)))[:-2].zfill(10)),
                                          abc_samples=params_[acc_samples_], abc_dist=dist_[:, acc_samples_], abc_im=out_)
                    print("Samples saved.")

            return abc_samples, abc_dist, abc_im

        # Do rejection sampling
        abc_samples, abc_dist, abc_im = rejection_sampling(abc["eps_rej"], abc["n_accept"], abc["n_samples"], abc["n_batch"], always_save=abc["abc_path"])

        # Save
        np.savez(os.path.join(abc["abc_path"], "all_data_" + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + str(np.round(random.uniform(0, 1e10)))[:-2].zfill(10)),
                 abc_samples=abc_samples, abc_dist=abc_dist, abc_im=abc_im)

        # Return
        return abc_samples, abc_dist, abc_im

    def get_summary(self, sess, is_training=True):
        return sess.run(self._merged, feed_dict={self._is_training: is_training})

    def eval_tensor(self, tensor_name, sess, is_training=True):
        return sess.run(eval("self." + tensor_name), feed_dict={self._is_training: is_training})

    @property
    def saver(self):
        return self._saver

    @property
    def saver_eval(self):
        return self._saver_eval

    @property
    def checkpt_dir_read(self):
        return self._checkpt_dir_read

    @property
    def is_training(self):
        return self._is_training

    @property
    def real_d(self):
        return self._real_d

    @property
    def g_noise(self):
        return self._g_noise

    @property
    def g_params(self):
        return self._g_params

    @property
    def d_inputs(self):
        return self._d_inputs
