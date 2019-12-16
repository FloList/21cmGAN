"""
This file contains useful functions for the 21cm temperature prediction
"""
import os
import errno
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.legend_handler import HandlerTuple
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc('image', cmap='gnuplot')
import numpy as np
import tensorflow as tf
import re
from pynverse import inversefunc

# Normalisation for colour map:
# NOTE: This normalisation is different from the one used to normalised the images for training!
class PlotNormalise(colors.Normalize):
    def __init__(self, par, vmin=-200, vmax=50, vcenter=0, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)
        self._data_mean = par["data_mean"]
        self._data_std = par["data_std"]
        self._scale_par = par["scale_par"]

    def __call__(self, value, clip=None):
        data_trafo = lambda im: 2.0 / np.pi * np.arctan((im - self._data_mean) / self._data_std * self._scale_par)
        x, y = [data_trafo(self.vmin), data_trafo(self.vcenter), data_trafo(self.vmax)], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(data_trafo(value), x, y))


# Make directory
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# DATA SCALING
# Undo scaling to arcsinh domain
def undo_data_normalisation(par, Y):
    if par["legacy"]:
        return (Y - 3.0 / 5.0) * 125.0
    else:
        data_trafo = lambda x: x / 175.0 + 2.0 / 5.0 + 0.075 * np.arcsinh(0.5 * x)
        return inversefunc(data_trafo, Y)


# PARAMETER SCALING
def scale_pars_tf(X, mean=None, std=None):
    X_ = tf.concat([tf.log(X[:, 0:1]), X[:, 1:2], tf.log(X[:, 2:])], axis=1)
    X_mean = tf.reduce_mean(X_, axis=0) if mean is None else mean
    X_std = tf.math.reduce_std(X_, axis=0) if std is None else std
    X_ = (X_ - X_mean) / X_std
    return X_, X_mean, X_std


# Parameter scaling
def scale_pars(X, mean=None, std=None):
    X_ = X.copy()  # need to copy array in order not to affect original X
    X_[:, 0] = np.log(X_[:, 0])
    X_[:, 2] = np.log(X_[:, 2])
    X_mean = np.mean(X_, axis=0) if mean is None else mean
    X_std = np.std(X_, axis=0) if std is None else std
    X_ = (X_ - X_mean) / X_std
    return X_, X_mean, X_std


# Undo parameter scaling
def undo_par_scaling(X, mean, std):
    X_ = X.copy()  # make a copy in order not to overwrite the original parameters
    X_ = (X_ * std) + mean
    X_[:, 0] = np.exp(X_[:, 0])
    X_[:, 2] = np.exp(X_[:, 2])
    return X_

# Check arguments
def check_input(X, Y):
    assert X.shape[0] == Y.shape[0], "X and Y contain a different number of images!"
    return 0

# Get shape of a tensor
def get_shape(tensor):
    return tensor.get_shape().as_list()

# Count trainable variables
def count_trainable_vars(vars=None):
    total_parameters = 0
    for variable in vars or tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Trainable variables:", total_parameters)
    return total_parameters


# Find most recent PG stage folder and latest iteration
# (only supports one-digit PG stages)
def get_latest_stage(path):
    subdirs = os.listdir(path)
    max_PG = 0
    max_has_non_t = False
    if len(subdirs) > 0:
        max_PG = np.max([[i * (str(i) in subdirs[j]) for i in range(9)] for j in range(len(subdirs))])  # max. found PG
        max_has_non_t = os.path.exists(os.path.join(path, str(max_PG) + "_False"))  # True if non-trans. PG is found, else False
        latest_folder = os.path.join(path, str(max_PG) + "_False") if max_has_non_t else os.path.join(path, str(max_PG) + "_True")
        content = os.listdir(latest_folder)
        max_it = 0
        for f in content:
            no_curr = [int(s) for s in re.findall(r'\d+', f)]
            no_curr = 0 if no_curr == [] else no_curr[0]
            max_it = max(max_it, no_curr)
    else:
        max_it = 0

    return max_PG, max_has_non_t, max_it


# ########################## WASSERSTEIN DISTANCE (taken from scipy.stats) ##########################
def validate_distribution(values, weights):
    # Validate the value array.
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Distribution can't be empty.")

    # Validate the weight array, if specified.
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(values):
            raise ValueError('Value and weight array-likes for the same '
                             'empirical distribution must be of the same size.')
        if np.any(weights < 0):
            raise ValueError('All weights must be non-negative.')
        if not 0 < np.sum(weights) < np.inf:
            raise ValueError('Weight array-like sum must be positive and '
                             'finite. Set as None for an equal distribution of '
                             'weight.')

        return values, weights
    return values, None


def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
    return cdf_distance(1, u_values, v_values, u_weights, v_weights)


def cdf_distance(p, u_values, v_values, u_weights=None, v_weights=None):
    u_values, u_weights = validate_distribution(u_values, u_weights)
    v_values, v_weights = validate_distribution(v_values, v_weights)

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
    return np.power(np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p),
                                       deltas)), 1/p)
# #######################################################################

# ########################## DISTANCE MEASURES ##########################
# input: either n_batch x n_samples x H x W or
#               n_batch             x H x W.
# NOTE: when comparing generated samples with n_batch > 1 to truth sample with n_batch = 1, truth needs to be y2
# Helper function: calculate T histogram across the slit for each redshift, input shape: n_batch x H x W, output shape: n_batch x W x n_bins
def calc_T_hist(im, bins=np.linspace(-200, 50, 251)):
    n_out = 1.0 * np.asarray([[np.histogram(im[k, :, j], bins=bins)[0] for j in range(im.shape[-1])] for k in range(im.shape[0])])
    n_out /= im.shape[1]
    return n_out

# Just compare means at each redshift and take L2 norm
def dist_mean_L2(y1, y2):
    if len(y1.shape) is 4:
        y1 = np.reshape(y1, [y1.shape[0], y1.shape[1] * y1.shape[2], y1.shape[3]], order="F")
    if len(y2.shape) is 4:
        y2 = np.reshape(y2, [y2.shape[0], y2.shape[1] * y2.shape[2], y2.shape[3]], order="F")
    return np.sqrt(((np.abs(np.mean(y1, axis=1, keepdims=True) - np.mean(y2, axis=1, keepdims=True))) ** 2).mean(axis=2)).flatten()

# Function that calculates the norm over a (= (1+z)^(-1)) of the Wasserstein distance between two temperature distribution
# The calculation can be done binned or unbinned (default, takes longer!)
def dist_wasserstein(y1, y2, bins=None, p=2):
    if len(y1.shape) is 4:
        y1 = np.reshape(y1, [y1.shape[0], y1.shape[1] * y1.shape[2], y1.shape[3]], order="F")
    if len(y2.shape) is 4:
        y2 = np.reshape(y2, [y2.shape[0], y2.shape[1] * y2.shape[2], y2.shape[3]], order="F")
    # Binned calculation
    if bins is not None:
        bins_c = (bins[1:] + bins[:-1]) / 2.0
        y1_hist = calc_T_hist(y1, bins)
        y2_hist = calc_T_hist(y2, bins)
        # if y2 is ground truth (n_batch == 1), tile to shape of y1
        if y2_hist.shape[0] == 1 and y1_hist.shape[0] > 1:
            y2_hist = np.tile(y2_hist, [y1_hist.shape[0], 1, 1])
        # L2 distance w.r.t. scale factor a
        if p is 2:
            d_wasserstein = np.sqrt(np.mean(np.asarray(
                [[wasserstein_distance(u_values=bins_c, v_values=bins_c, u_weights=y1_hist[k, j, :], v_weights=y2_hist[k, j, :])
                for j in range(y1_hist.shape[1])] for k in range(y1_hist.shape[0])]) ** 2, axis=1))
        # L1 distance w.r.t. scale factor a
        elif p is 1:
            d_wasserstein = np.mean(np.asarray(np.abs(
                [[wasserstein_distance(u_values=bins_c, v_values=bins_c, u_weights=y1_hist[k, j, :],
                                       v_weights=y2_hist[k, j, :])
                  for j in range(y1_hist.shape[1])] for k in range(y1_hist.shape[0])])), axis=1)

    # Unbinned calculation
    else:
        # if y2 is ground truth (n_batch == 1), tile to shape of y1
        if y2.shape[0] == 1 and y1.shape[0] > 1:
            y2 = np.tile(y2, [y1.shape[0], 1, 1])

        # L2 distance w.r.t. scale factor a
        if p is 2:
            d_wasserstein = np.sqrt(np.mean(np.asarray(
            [[wasserstein_distance(u_values=y1[k, :, j], v_values=y2[k, :, j])
              for j in range(y1.shape[2])] for k in range(y1.shape[0])]) ** 2, axis=1))
        # L1 distance w.r.t. scale factor a
        elif p is 1:
            d_wasserstein = np.mean(np.asarray(np.abs(
                [[wasserstein_distance(u_values=y1[k, :, j], v_values=y2[k, :, j])
                  for j in range(y1.shape[2])] for k in range(y1.shape[0])])), axis=1)

    return np.atleast_1d(d_wasserstein)

# Function that calculates the L2-norm over z of the difference of the logarithmic T distributions
def dist_PDF_L2(y1, y2, eps=1e-16):
    if len(y1.shape) is 4:
        y1 = np.reshape(y1, [y1.shape[0], y1.shape[1] * y1.shape[2], y1.shape[3]], order="F")
    if len(y2.shape) is 4:
        y2 = np.reshape(y2, [y2.shape[0], y2.shape[1] * y2.shape[2], y2.shape[3]], order="F")
    y1_hist = calc_T_hist(y1)
    y2_hist = calc_T_hist(y2)
    # set minimum value before taking the log
    y1_hist[np.where(y1_hist < eps)] = eps
    y2_hist[np.where(y2_hist < eps)] = eps
    d_PDF_L2 = (np.sqrt(np.mean(np.mean((np.log(y1_hist) - np.log(y2_hist)) ** 2, axis=-1), axis=-1))).flatten()
    return d_PDF_L2

# Wrapper that returns all the distances above
def dist_all(y1, y2, eps=1e-16, bins=None):
    return dist_mean_L2(y1, y2), dist_wasserstein(y1, y2, bins=bins, p=1), dist_wasserstein(y1, y2, bins=bins, p=2), dist_PDF_L2(y1, y2, eps=eps)

# Function that calculates the loss as determined by the critic of the GAN
def dist_critic_loss(sess, model, noise, params, d_inputs):
    critic_score = model.real_d.discriminator.eval(session=sess, feed_dict={model.is_training: False, model.g_noise: noise, model.g_params: params, model.d_inputs: d_inputs})
    return - critic_score

##########################

# Create output plot
def output_plot(X, Y, sess, model, n_image, par, filename, is_training=True, p=None, X_mean=None, X_std=None, vmin=-200, vmax=50, norm_data=True):
    # if p is given, it overrides n_image
    fig = plt.figure()
    midnorm = PlotNormalise(par, vmin=vmin, vcenter=0, vmax=vmax)
    fig.set_size_inches(10, 10)
    fig.subplots_adjust(left=0.05, bottom=0.05,
                           right=0.95, top=0.95, wspace=0.1, hspace=0.1)
    n_image = min(n_image, X.shape[0])
    if p is None:
        p = np.random.permutation(Y.shape[0])
    for i in range(0, 2 * n_image, 2):
        # Plot 2 images: First is the ground truth, second is the generator output
        params = np.expand_dims(np.squeeze(X[p[i // 2]]), 0)
        if X_mean is not None and X_std is not None:
            params_plot = np.round(undo_par_scaling(params, X_mean, X_std), 1)
        else:
            params_plot = np.round(params, 1)

        noise = np.random.normal(size=[1, par["n_noise"]])
        y = Y[p[i // 2]]
        model_out = np.squeeze(model.sample_generator(sess, params, noise, is_training=is_training)[0], 2)
        model_out = undo_data_normalisation(par, model_out)

        fig.add_subplot(n_image, 2, i + 1)
        y_out = undo_data_normalisation(par, y) if norm_data else y
        im = plt.imshow(y_out, norm=midnorm, cmap=get_cmap21())
        ax = plt.gca()
        if i is 0:
            ax.set_title("21SSD samples", fontsize=20)
        ax.text(0.0125, 0.9, r"$f_X=%0.1f$" % params_plot[0, 0], verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes, color='white', fontsize=7)
        ax.text(0.0125, 0.5, r"$r_{h/s}=%0.1f$" % params_plot[0, 1], verticalalignment='center', horizontalalignment='left',
                transform=ax.transAxes, color='white', fontsize=7)
        ax.text(0.0125, 0.1, r"$f_\alpha=%0.1f$" % params_plot[0, 2], verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes, color='white', fontsize=7)
        plt.axis('off')

        if i >= 2 * n_image - 2:
            labels_nu = np.arange(6, 16)[::-1]
            labels_pos = np.cumsum(np.asarray([1 / z - 1 / (z+1) for z in labels_nu[:-1]]))
            labels_pos *= (model_out.shape[-1] - 1) / labels_pos[-1]
            labels_pos = np.hstack([0, labels_pos])
            ax.set_xticks(labels_pos[:-1])
            ax.set_xticklabels(labels_nu[:-1], fontsize=10)
            ax.axis('on')
            ax.get_yaxis().set_visible(False)
            ax.tick_params(axis=u'both', which=u'both', length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.add_subplot(n_image, 2, i + 2)
        plt.imshow(model_out, norm=midnorm, cmap=get_cmap21())
        ax = plt.gca()
        if i is 0:
            ax.set_title("NN samples", fontsize=20)
        plt.axis('off')
        if i >= 2 * n_image - 2:
            ax.set_xticks(labels_pos[1:])
            ax.set_xticklabels(labels_nu[1:], fontsize=10)
            ax.axis('on')
            ax.get_yaxis().set_visible(False)
            ax.tick_params(axis=u'both', which=u'both', length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.subplots_adjust(left=0.05, bottom=0.18, right=0.95, top=0.95, wspace=0.02, hspace=-0.86)
    cbar_ax = fig.add_axes([0.05, 0.30, 0.9, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"$\delta T_b \ [\mathrm{mK}]$")
    pretty_plots()
    plt.savefig(filename)
    plt.close("all")


# Plot samples for a specific set of parameters
def output_plot_fixed(X, Y, sess, model, n_image, par, filename, params=None, is_training=True, X_mean=None, X_std=None, vmin=-200, vmax=50, norm_data=True, show_real_data=True):
    midnorm = PlotNormalise(par, vmin=vmin, vcenter=0, vmax=vmax)
    if params is None:
        params = X[np.random.choice(X.shape[0]), :]
    fig = plt.figure()
    fig.set_size_inches(18, 9)
    fig.subplots_adjust(left=0.05, bottom=0.05,
                           right=0.95, top=0.95, wspace=0.1, hspace=0.1)
    n_image = min(n_image, X.shape[0])

    # Check if / where parameters exist in X
    par_in_X = np.where([np.all(X[i, :] == params) for i in range(X.shape[0])])[0]
    if len(par_in_X) > 0 and show_real_data:
        n_image = min(n_image, len(par_in_X))
        rand_perm = np.random.permutation(len(par_in_X))
        Y_par = Y[par_in_X[rand_perm]]

    # If parameters are contained in X: plot 2 columns, else 1 column
    no_plots = 2 * n_image if len(par_in_X) > 0 and show_real_data else n_image

    if X_mean is not None and X_std is not None:
        params_plot = np.round(undo_par_scaling(np.expand_dims(params, 0), X_mean, X_std), 1)
    else:
        params_plot = np.round(np.expand_dims(params, 0), 1)

    for i in range(0, n_image):
        ii = 2 * i if len(par_in_X) > 0 and show_real_data else i

        # Plot 2 images: First is the ground truth (if it exists), second is the generator output
        if len(par_in_X) > 0 and show_real_data:
            y = Y_par[i]
            fig.add_subplot(n_image, no_plots / n_image, ii + 1)
            y_out = undo_data_normalisation(par, y) if norm_data else y
            plt.imshow(y_out, norm=midnorm, cmap=get_cmap21())
            ax = plt.gca()
            if i is 0:
                title_str = "21SSD samples\n"
                title_str += r"$f_X=%.1f$, $r_{h/s}=%.1f$, $f_\alpha=%.1f$" % (params_plot[0, 0], params_plot[0, 1], params_plot[0, 2])
                ax.set_title(title_str, fontsize=20)
            plt.axis('off')

            if i >= n_image - 1:
                labels_nu = np.arange(6, 16)[::-1]
                labels_pos = np.cumsum(np.asarray([1 / z - 1 / (z + 1) for z in labels_nu[:-1]]))
                labels_pos *= (y.shape[-1] - 1) / labels_pos[-1]
                labels_pos = np.hstack([0, labels_pos])
                ax.set_xticks(labels_pos[:-1])
                ax.set_xticklabels(labels_nu[:-1], fontsize=16)
                ax.axis('on')
                ax.get_yaxis().set_visible(False)
                ax.tick_params(axis=u'both', which=u'both', length=0)
                for spine in ax.spines.values():
                    spine.set_visible(False)

        noise = np.random.normal(size=[1, par["n_noise"]])
        model_out = np.squeeze(model.sample_generator(sess, np.expand_dims(params, 0), noise, is_training=is_training)[0], 2)
        model_out = undo_data_normalisation(par, model_out)

        ii_NN = ii + 2 if len(par_in_X) > 0 and show_real_data else ii + 1
        fig.add_subplot(n_image, no_plots / n_image, ii_NN)
        im = plt.imshow(model_out, norm=midnorm, cmap=get_cmap21())
        ax = plt.gca()
        if i is 0:
            title_str = "NN samples\n"
            title_str += r"$f_X=%.1f$, $r_{h/s}=%.1f$, $f_\alpha=%.1f$" % (params_plot[0, 0], params_plot[0, 1], params_plot[0, 2])
            ax.set_title(title_str, fontsize=20)
        plt.axis('off')
        if i >= n_image - 1:
            labels_nu = np.arange(6, 16)[::-1]
            labels_pos = np.cumsum(np.asarray([1 / z - 1 / (z + 1) for z in labels_nu[:-1]]))
            labels_pos *= (model_out.shape[-1] - 1) / labels_pos[-1]
            labels_pos = np.hstack([0, labels_pos])
            ax.set_xticks(labels_pos[1:])
            ax.set_xticklabels(labels_nu[1:], fontsize=16)
            ax.axis('on')
            ax.get_yaxis().set_visible(False)
            ax.tick_params(axis=u'both', which=u'both', length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.14, right=0.95, top=0.9, wspace=0.02, hspace=0.02)
    cbar_ax = fig.add_axes([0.05, 0.07, 0.9, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"$\delta T_b \ [\mathrm{mK}]$", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    pretty_plots()
    plt.savefig(filename)
    plt.close("all")


# Plot samples interpolating linearly between two sets of parameters (params: 2 x 3 or n_interp x 2 x 3)
def output_plot_interpolate(X, sess, model, n_image, par, filename, params=None, is_training=True, X_mean=None, X_std=None, vmin=-200, vmax=50):
    midnorm = PlotNormalise(par, vmin=vmin, vcenter=0, vmax=vmax)
    var_names = np.asarray([r"$f_X$", r"$r_{h/s}$", r"$f_\alpha$"])
    if params is None:
        params = X[np.random.choice(X.shape[0], 2), :]

    # Take the same noise for all samples
    noise = np.random.normal(size=[1, par["n_noise"]])

    # Get 3D tensor
    if len(params.shape) is 2:
        params = np.expand_dims(params, 0)
    n_interp = params.shape[0]

    fig = plt.figure()
    fig.set_size_inches(10 * max(1, 2 * n_interp / 3), 10)
    fig.subplots_adjust(left=0.05, bottom=0.05,
                        right=0.95, top=0.95, wspace=0.1, hspace=0.1)

    for i in range(0, n_image):
        for i_interp in range(n_interp):
            def interp_fun(alpha):
                return (1 - alpha) * params[i_interp, 0, :] + alpha * params[i_interp, 1, :]
            all_params = np.asarray([interp_fun(i / (n_image - 1)) for i in range(n_image)])

            # Plot 1 image: generator output
            model_out = np.squeeze(model.sample_generator(sess, np.expand_dims(all_params[i, :], 0), noise, is_training=is_training)[0], 2)
            model_out = undo_data_normalisation(par, model_out)

            fig.add_subplot(n_image, n_interp, n_interp * i + i_interp + 1)
            im = plt.imshow(model_out, norm=midnorm, cmap=get_cmap21())
            ax = plt.gca()
            if i is 0:
                if n_interp is 3:
                    ax.set_title("Varying " + var_names[i_interp], fontsize=20)
            plt.axis('off')
            if X_mean is not None and X_std is not None:
                params_plot = np.round(undo_par_scaling(np.expand_dims(all_params[i], 0), X_mean, X_std), 1)
            else:
                params_plot = np.round(np.expand_dims(all_params[i], 0), 1)
            ax.text(0.0125, 0.9, r"$f_X=%0.1f$" % params_plot[0, 0], verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='white', fontsize=10)
            ax.text(0.0125, 0.5, r"$r_{h/s}=%0.1f$" % params_plot[0, 1], verticalalignment='center', horizontalalignment='left', transform=ax.transAxes, color='white', fontsize=10)
            ax.text(0.0125, 0.1, r"$f_\alpha=%0.1f$" % params_plot[0, 2], verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, color='white', fontsize=10)
            plt.axis('off')

            if i >= n_image - 1:
                labels_nu = np.arange(6, 16)[::-1]
                labels_pos = np.cumsum(np.asarray([1 / z - 1 / (z+1) for z in labels_nu[:-1]]))
                labels_pos *= (model_out.shape[-1] - 1) / labels_pos[-1]
                labels_pos = np.hstack([0, labels_pos])
                if i_interp > 0:
                    labels_nu = labels_nu[1:]
                    labels_pos = labels_pos[1:]
                if i_interp < n_interp - 1:
                    labels_nu = labels_nu[:-1]
                    labels_pos = labels_pos[:-1]
                ax.set_xticks(labels_pos)
                ax.set_xticklabels(labels_nu, fontsize=12)
                ax.axis('on')
                ax.get_yaxis().set_visible(False)
                ax.tick_params(axis=u'both', which=u'both', length=0)
                for spine in ax.spines.values():
                    spine.set_visible(False)

    plt.tight_layout()
    hspace = -np.log2(model_out.shape[0] + 1) * 1.0 / 6
    plt.subplots_adjust(left=0.05, bottom=0.18, right=0.95, top=0.95, wspace=0.02, hspace=hspace)
    bottom = 0.36 * np.log2(model_out.shape[0] + 1) / 6
    cbar_ax = fig.add_axes([0.05, bottom, 0.9, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"$\delta T_b \ [\mathrm{mK}]$", fontdict={"size": 16})
    cbar.ax.tick_params(labelsize=16)
    pretty_plots()
    plt.savefig(filename)
    plt.close("all")


# Create output plot showing the mean temperature at each redshift
def output_plot_avg(X, Y, sess, model, n_image, par, filename, n_sample=30, p=None, is_training=True, X_mean=None, X_std=None, all_in_one=False, cols=None, norm_data=True, X_model=None, Y_model=None):
    # if p is given, it overrides n_image
    if all_in_one:
        fig, axs = plt.subplots(2, 1, sharex="all", sharey=False, gridspec_kw=dict(width_ratios=[1], height_ratios=[3, 1]))
        fig.set_size_inches(18.86, 10)
    else:
        fig = plt.figure()
        fig.set_size_inches(10, 10)

    fig.subplots_adjust(left=0.05, bottom=0.05,
                           right=0.95, top=0.95, wspace=0.1, hspace=0.1)

    if p is None:
        n_image = min(n_image, X.shape[0])

    if all_in_one:
        if cols is None:
            cols = np.asarray(['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#ca82d6', '#6a3d9a', '#cccc99', '#b15928', '#c9d9d9', '#000000'])
        cols_1 = cols[::2]
        cols_2 = cols[1::2]
    else:
        cols_1 = cols_2 = [None]

    if p is None:
        p = np.random.permutation(Y.shape[0])
    for i in range(0, n_image):
        # Plot 1 image for ground truth and generator output
        if X_model is None:
            params = np.tile(np.expand_dims(np.squeeze(X[p[i]]), 0), [n_sample, 1])
            noise = np.random.normal(size=[n_sample, par["n_noise"]])
            model_out = np.squeeze(model.sample_generator(sess, params, noise, is_training=is_training), -1)
            model_out = undo_data_normalisation(par, model_out)
        else:
            # Check if / where parameters exist in X_model
            if X_mean is not None and X_std is not None:
                pars_orig = undo_par_scaling(np.expand_dims(X[p[i]], 0), X_mean, X_std)
            else:
                pars_orig = X[p[i]]
            pars_in_X_model = np.where([np.all(np.isclose(X_model[j, :], pars_orig)) for j in range(X_model.shape[0])])[0]
            assert (len(pars_in_X_model) >= n_sample), "Not enough samples found!"
            rand_choice = np.random.choice(len(pars_in_X_model), size=n_sample)
            model_out = Y_model[pars_in_X_model[rand_choice]]
        model_out_z_avg_mean = np.mean(np.mean(model_out, 1), 0)
        model_out_z_avg_std = np.std(np.mean(model_out, 1), 0)

        labels_nu = np.arange(6, 16)[::-1]
        labels_pos = np.cumsum(np.asarray([1 / z - 1 / (z + 1) for z in labels_nu[:-1]]))
        labels_pos *= (model_out.shape[-1] - 1) / labels_pos[-1]
        labels_pos = np.hstack([0, labels_pos])

        y_par = X[p[i]]
        y_par_all = np.where([np.all(X[i, :] == y_par) for i in range(X.shape[0])])[0]
        y = Y[y_par_all]
        y = undo_data_normalisation(par, y) if norm_data else y
        y_z_avg_mean = np.mean(np.mean(y, 1), 0)
        y_z_avg_std = np.std(np.mean(y, 1), 0)

        if not all_in_one:
            fig.add_subplot(n_image, 1, i + 1)
            plt.plot(np.arange(len(model_out_z_avg_mean)), model_out_z_avg_mean)
            plt.fill_between(x=np.arange(len(model_out_z_avg_mean)), y1=model_out_z_avg_mean + model_out_z_avg_std, y2=model_out_z_avg_mean - model_out_z_avg_std, alpha=0.3)
            plt.plot(np.arange(len(y_z_avg_mean)), y_z_avg_mean)
            plt.fill_between(x=np.arange(len(y_z_avg_mean)), y1=y_z_avg_mean + y_z_avg_std, y2 = y_z_avg_mean - y_z_avg_std, alpha = 0.3)
            ax = plt.gca()
            if X_mean is not None and X_std is not None:
                params_plot = np.round(undo_par_scaling(np.expand_dims(params[0], 0), X_mean, X_std), 1)
            else:
                params_plot = np.round(np.expand_dims(params[0], 0), 1)
            fig.text(0.5, ax.get_position().corners()[0, 1] - 0.025, r"$f_X=%.1f, r_{h/s}=%.1f, f_\alpha=%.1f$" % (params_plot[0, 0], params_plot[0, 1], params_plot[0, 2]), ha="center")
            ax.axis('off')
        else:
            color_real = None if not all_in_one else np.take(cols_1, i, mode="wrap")
            color_fake = None if not all_in_one else np.take(cols_2, i, mode="wrap")
            axs[0].plot(np.arange(len(y_z_avg_mean)), y_z_avg_mean - y_z_avg_std, lw=1.5, color="k", ls=":")
            axs[0].plot(np.arange(len(y_z_avg_mean)), y_z_avg_mean + y_z_avg_std, lw=1.5, color="k", ls=":")
            # axs[0].fill_between(x=np.arange(len(y_z_avg_mean)), y1=y_z_avg_mean + y_z_avg_std, y2=y_z_avg_mean - y_z_avg_std, alpha=0.1, facecolor=color_real, lw=4, edgecolor="k", ls=":")
            axs[0].fill_between(x=np.arange(len(model_out_z_avg_mean)), y1=model_out_z_avg_mean + model_out_z_avg_std, y2=model_out_z_avg_mean - model_out_z_avg_std, alpha=0.1, facecolor=color_fake)
            axs[0].plot(np.arange(len(y_z_avg_mean)), y_z_avg_mean, color=color_real, label='real' + str(i), lw=2.5)
            axs[0].plot(np.arange(len(model_out_z_avg_mean)), model_out_z_avg_mean, color=color_fake, label='fake' + str(i), lw=2.5)
            axs[1].plot(np.arange(len(y_z_avg_mean)), model_out_z_avg_mean-y_z_avg_mean, color=color_fake, label='delta' + str(i), lw=2.5)
            axs[1].set_xticks(labels_pos)
            axs[1].set_xticklabels(labels_nu, fontsize=20)
            axs[0].set_yticks(np.arange(-150, 75, 25))
            axs[0].set_yticklabels(np.arange(-150, 75, 25), fontsize=20)
            axs[0].set_ylim([-160, 50])
            axs[0].set_yticklabels(axs[0].get_yticklabels(), fontsize=20)
            axs[0].tick_params(axis=u'both', which=u'both', length=0, pad=10)
            axs[0].set_ylabel(r"$\langle\delta T_b\rangle \ [\mathrm{mK}]$", fontsize=20)
            axs[1].set_ylabel(r"$Difference \ [\mathrm{mK}]$", fontsize=20)
            axs[1].set_ylim([-20, 20])
            axs[1].set_yticks(np.arange(-10, 20, 10))
            axs[1].set_yticklabels(np.arange(-10, 20, 10), fontsize=20)
            axs[1].set_xlabel(r"$z$", fontsize=20)
            axs[1].tick_params(axis=u'both', which=u'both', length=0, pad=10)

        if i is 0:
            # ax.set_title("Global 21cm signal", fontsize=20)
            if all_in_one:
                axs[0].legend(["NN", "21SSD"])

    if all_in_one:
        if X_mean is not None and X_std is not None:
            params_plot = np.round(undo_par_scaling(X[p], X_mean, X_std), 1)
        else:
            params_plot = np.round(X[p], 1)
        leg_str = [""] * params_plot.shape[0]
        var_names = np.asarray([r"$f_X$", r"$r_{h/s}$", r"$f_\alpha$"])
        for i_set in range(params_plot.shape[0]):
            for i_var in range(params_plot.shape[1]):
                just_width = 7 if i_set > 1 and i_var is 0 else 6
                leg_str[i_set] += var_names[i_var] + " = " + str(np.round(params_plot[i_set, i_var], 1)).ljust(just_width)

        handles, _ = axs[0].get_legend_handles_labels()
        handles_tuple = [tuple([handles[i], handles[i + 1]]) for i in range(0, 2*params_plot.shape[0], 2)]
        axs[0].legend(handles_tuple, np.asarray(leg_str), fontsize=20, handler_map={tuple: HandlerTuple(ndivide=None)}, ncol=1)

        ax_top = axs[0].twiny()
        new_tick_loc = np.round(np.linspace(0, model_out.shape[-1], 7)).astype(int)
        ax_top.set_xlim(axs[0].get_xlim())
        ax_top.set_xticks(new_tick_loc)
        c_light = 2.998e10
        lambda_0 = 21.106
        a_ = 1.0 / 16.0
        b_ = (1.0 / 7.0 - 1.0 / 16.0) / (y.shape[-1] - 1)
        new_redshifts = 1.0 / (a_ + new_tick_loc * b_) - 1
        ax_top.set_xticklabels(
            np.round((1.0 / (1.0 + new_redshifts) * c_light / lambda_0 / 1e6)).astype(int).astype(str), fontsize=20)
        ax_top.set_xlabel(r"$\nu \ [\mathrm{MHz}]$" + "\n", fontsize=20)
        ax_top.tick_params(axis=u'both', which=u'both', length=0)
        ax_top.set_ylabel(axs[0].get_ylabel())
        ax_top.set_yticks(axs[0].get_yticks())
        ax_top.set_yticklabels(axs[0].get_yticklabels())

    pretty_plots()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close("all")


# Create output plot showing the pixel distribution function (PDF) of the mean temperature at each redshift
def output_plot_PDF(X, Y, sess, model, par, shape, filename, n_sample=100, params=None, is_training=True, X_mean=None, X_std=None, norm_data=True, X_model=None, Y_model=None):
    if params is None:
        params = X[np.random.choice(X.shape[0]), :]
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    fig.subplots_adjust(left=0.05, bottom=0.05,
                        right=0.95, top=0.95, wspace=0.1, hspace=0.1)
    bins = np.linspace(-200, 50, 251)
    aspect = 0.001 * Y.shape[2]

    for i in range(0, params.shape[0]):
        if X_mean is not None and X_std is not None:
            params_plot = np.round(undo_par_scaling(np.expand_dims(params[i], 0), X_mean, X_std), 1)
        else:
            params_plot = np.round(np.expand_dims(params[i], 0), 1)

        # Check if / where parameters exist in X
        par_in_X = np.where([np.all(X[j, :] == params[i, :]) for j in range(X.shape[0])])[0]
        if len(par_in_X) > 0:
            n_image = len(par_in_X)
            rand_perm = np.random.permutation(n_image)
            Y_par = Y[par_in_X[rand_perm]]

        # Plot 3 images: First is the ground truth (if it exists), second is the generator output, third is the difference (if ground truth exists)
        if len(par_in_X) > 0:
            y = np.reshape(Y_par, [-1, Y_par.shape[-1]])
            y = undo_data_normalisation(par, y) if norm_data else y
            n_y = 1.0 * np.asarray([np.histogram(y[:, j], bins=bins)[0] for j in range(y.shape[1])])
            n_y /= (1.0 * n_y.max())
            fig.add_subplot(params.shape[0], 3, 3 * i + 1)
            plt.imshow(np.log(1.0 + n_y.T), vmin=0, vmax=1, cmap=get_turbo(), interpolation="spline16", aspect=aspect)
            ax = plt.gca()
            title_str = r"$f_X=%.1f, r_{h/s}=%.1f, f_\alpha=%.1f$" % (params_plot[0, 0], params_plot[0, 1], params_plot[0, 2])
            title_str = "21SSD\n" + title_str if i == 0 else title_str
            ax.set_title(title_str, fontsize=20)
            plt.axis('off')
        else:
            fig.add_subplot(params.shape[0], 3, 3 * i + 1)
            plt.imshow(np.ones((len(bins), shape[1])), aspect=aspect)
            ax = plt.gca()
            plt.axis('off')

        # NN plot
        if X_model is None:
            params_tiled = np.tile(np.asarray(params[i, :]), [n_sample, 1])
            noise = np.random.normal(size=[n_sample, par["n_noise"]])
            model_out = np.squeeze(model.sample_generator(sess, params_tiled, noise, is_training=is_training), -1)
            model_out = undo_data_normalisation(par, model_out)
        else:
            # Check if / where parameters exist in X_model
            if X_mean is not None and X_std is not None:
                pars_orig = undo_par_scaling(np.expand_dims(params[i, :], 0), X_mean, X_std)
            else:
                pars_orig = params[i, :]
            pars_in_X_model = np.where([np.all(np.isclose(X_model[j, :], pars_orig)) for j in range(X_model.shape[0])])[0]
            assert (len(pars_in_X_model) >= n_sample), "Not enough samples found!"
            rand_choice = np.random.choice(len(pars_in_X_model), size=n_sample)
            model_out = Y_model[pars_in_X_model[rand_choice]]
        model_reshape = np.reshape(model_out, [-1, model_out.shape[-1]])
        n_out = 1.0 * np.asarray([np.histogram(model_reshape[:, j], bins=bins)[0] for j in range(model_reshape.shape[1])])
        n_out /= (1.0 * n_out.max())
        fig.add_subplot(params.shape[0], 3, 3 * i + 2)
        plt.imshow(np.log(1.0 + n_out.T), vmin=0, vmax=1, cmap=get_turbo(), interpolation="spline16", aspect=aspect)
        ax = plt.gca()
        title_str = r"$f_X=%.1f, r_{h/s}=%.1f, f_\alpha=%.1f$" % (params_plot[0, 0], params_plot[0, 1], params_plot[0, 2])
        title_str = "NN\n" + title_str if i == 0 else title_str
        ax.set_title(title_str, fontsize=20)
        plt.axis('off')

        # Difference plot
        if len(par_in_X) > 0:
            d_EM = [wasserstein_distance(range(len(bins)-1), range(len(bins)-1), n_y[i], n_out[i]) for i in range(n_out.shape[0])]
            d_mean_T = np.abs(y.mean(0) - model_reshape.mean(0))
            std_y = y.std(0)
            std_out = model_reshape.std(0)
            fig.add_subplot(params.shape[0], 3, 3 * i + 3)
            plt.plot(range(n_out.shape[0]), d_EM)
            plt.plot(range(n_out.shape[0]), d_mean_T)
            plt.plot(range(n_out.shape[0]), np.abs(std_y - std_out), 'k--')
            ax = plt.gca()
            title_str = r"$f_X=%.1f, r_{h/s}=%.1f, f_\alpha=%.1f$" % (params_plot[0, 0], params_plot[0, 1], params_plot[0, 2])
            title_str = "Wasserstein distance\n" + title_str if i == 0 else title_str
            ax.set_title(title_str, fontsize=20)
            ax.legend(["EMD", r"$\Delta$ means", r"$\Delta$ STDs", r"$\sqrt{\sigma_{\mathrm{NN}}}$"])


    pretty_plots()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close("all")

# Create output plot showing the pixel histogram at several redshifts
def output_plot_hist(X, Y, sess, model, n_image, par, filename, redshifts, n_sample=30, p=None, is_training=True, X_mean=None, X_std=None, nbins=20, cols=None, norm_data=True, X_model=None, Y_model=None):
    if cols is None:
        cols = np.asarray(['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#ca82d6', '#6a3d9a', '#cccc99', '#b15928', '#c9d9d9', '#000000'])
    cols_1 = cols[::2]
    cols_2 = cols[1::2]

    # if p is given, it overrides n_image
    n_redshift = len(np.asarray(redshifts).flatten())
    if not np.sqrt(n_redshift) % 1:
        fig, axs = plt.subplots(n_redshift // 2, n_redshift // 2, sharex="none", sharey="none", squeeze=False)
    else:
        fig, axs = plt.subplots(1, n_redshift, sharex="none", sharey="row", squeeze=False)
    fig.set_size_inches(12, 8)
    ax_big = fig.add_subplot(111, frameon=False)
    ax_big.set_ylabel(""), ax_big.set_xticks([]), ax_big.set_yticks([])
    fig.subplots_adjust(left=0.1, bottom=0.1,
                           right=0.9, top=0.9, wspace=0.1, hspace=0.1)

    if p is None:
        n_image = min(n_image, X.shape[0])

    if p is None:
        p = np.random.permutation(Y.shape[0])
    for i in range(0, n_image):
        # Plot 1 image for ground truth and generator output
        if X_model is None:
            params = np.tile(np.expand_dims(np.squeeze(X[p[i]]), 0), [n_sample, 1])
            noise = np.random.normal(size=[n_sample, par["n_noise"]])
            model_out = np.squeeze(model.sample_generator(sess, params, noise, is_training=is_training), -1)
            model_out = undo_data_normalisation(par, model_out)
        else:
            # Check if / where parameters exist in X_model
            if X_mean is not None and X_std is not None:
                pars_orig = undo_par_scaling(np.expand_dims(X[p[i]], 0), X_mean, X_std)
            else:
                pars_orig = X[p[i]]
            pars_in_X_model = np.where([np.all(np.isclose(X_model[j, :], pars_orig)) for j in range(X_model.shape[0])])[0]
            assert (len(pars_in_X_model) >= n_sample), "Not enough samples found!"
            rand_choice = np.random.choice(len(pars_in_X_model), size=n_sample)
            model_out = Y_model[pars_in_X_model[rand_choice]]
        model_reshape = np.reshape(model_out, [-1, model_out.shape[-1]])
        y_par = X[p[i]]
        y_par_all = np.where([np.all(X[i, :] == y_par) for i in range(X.shape[0])])[0]
        y = Y[y_par_all]
        y = undo_data_normalisation(par, y) if norm_data else y
        y_reshape = np.reshape(y, [-1, y.shape[-1]])
        color_real = np.take(cols_1, i, mode="wrap")
        color_fake = np.take(cols_2, i, mode="wrap")

        a_ = 1.0 / 16.0
        b_ = (1.0 / 7.0 - 1.0 / 16.0) / (y.shape[-1] - 1)
        all_pixels = np.arange(Y.shape[-1])
        all_redshifts = 1.0 / (a_ + all_pixels * b_) - 1
        pixels_z = np.asarray([np.argmin(np.abs(all_redshifts - z)) for z in redshifts])
        redshifts_ = 1.0 / (a_ + pixels_z * b_) - 1  # redshift range: 15 - 6

        for j_ax, j in enumerate(pixels_z):
            n_out, bins = np.histogram(model_reshape[:, j], bins=nbins, density=True)
            n_y = np.histogram(y_reshape[:, j], bins=bins, density=True)[0]
            bins_c = (bins[1:] + bins[:-1]) / 2.0
            ax = axs[0, j_ax] if not np.sqrt(n_redshift) % 1 else axs[j_ax // 2, j_ax % 2]
            ax.plot(bins_c, n_y, color=color_real, lw=2.5, label="real" + str(j_ax))
            ax.plot(bins_c, n_out, color=color_fake, lw=2.5, label="fake" + str(j_ax))
            if j_ax is 0:
                ax.set_ylabel("Probability density", fontsize=16)
            ax.set_title(r"$z = %g$" % np.round(redshifts_[j_ax], 1), fontsize=20)
            # xticklabels = np.round(np.asarray(ax.get_xticks())).astype(int)
            xlim = [-155, 30] if j_ax is 0 else [-210, 55]
            ax.set_xlim(xlim)
            xticks = np.arange(-150, 50, 50) if j_ax is 0 else np.arange(-200, 100, 100)
            xticklabels = [r"$" + str(l) + r"$" for l in xticks]
            yticklabels = np.round(np.asarray(ax.get_yticks()), 2)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, fontdict={"size": 16})
            ax.set_yticklabels(yticklabels, fontdict={"size": 16})
            ax.tick_params(axis=u'both', which=u'both', length=0, pad=10)

    # if X_mean is not None and X_std is not None:
    #     params_plot = np.round(undo_par_scaling(X[p], X_mean, X_std), 1)
    # else:
    #     params_plot = np.round(X[p], 1)
    # leg_str = [""] * params_plot.shape[0]
    # var_names = np.asarray([r"$f_X$", r"$r_{h/s}$", r"$f_\alpha$"])
    # for i_set in range(params_plot.shape[0]):
    #     for i_var in range(params_plot.shape[1]):
    #         just_width = 7 if i_set > 0 and i_var is 0 else 6
    #         leg_str[i_set] += var_names[i_var] + " = " + str(np.round(params_plot[i_set, i_var], 1)).ljust(just_width)
    #
    # handles, _ = axs[0, 0].get_legend_handles_labels()
    # handles_tuple = [tuple([handles[i], handles[i + 1]]) for i in range(0, 2 * params_plot.shape[0], 2)]
    # axs[0, 0].legend(handles_tuple, np.asarray(leg_str), fontsize=20, handler_map={tuple: HandlerTuple(ndivide=None)}, ncol=1)
    ax_big.set_xlabel("\n\n" + r"$\delta T_b$", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    pretty_plots()
    plt.savefig(filename)
    plt.close("all")


# Create output plot showing the power spectrum at several redshifts
def output_plot_power(X, Y, sess, model, n_image, par, filename, redshifts, n_sample=30, p=None, is_training=True, X_mean=None, X_std=None, cols=None, norm_data=True, X_model=None, Y_model=None):
    if cols is None:
        cols = np.asarray(['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#ca82d6', '#6a3d9a', '#cccc99', '#b15928', '#c9d9d9', '#000000'])
    cols_1 = cols[::2]
    cols_2 = cols[1::2]

    # if p is given, it overrides n_image
    n_redshift = len(np.asarray(redshifts).flatten())
    if not np.sqrt(n_redshift) % 1:
        fig, axs = plt.subplots(n_redshift // 2, n_redshift // 2, sharex="none", sharey="none", squeeze=False)
    else:
        fig, axs = plt.subplots(1, n_redshift, sharex="none", sharey="row", squeeze=False)
    fig.set_size_inches(12, 8)
    ax_big = fig.add_subplot(111, frameon=False)
    ax_big.set_ylabel(""), ax_big.set_xticks([]), ax_big.set_yticks([])
    ax_big.tick_params(axis=u'both', which=u'minor', length=0, pad=10)
    # fig.subplots_adjust(left=0.05, bottom=0.05,
    #                     right=0.95, top=0.95, wspace=0.1, hspace=0.1)

    if p is None:
        n_image = min(n_image, X.shape[0])

    if p is None:
        p = np.random.permutation(Y.shape[0])
    for i in range(0, n_image):
        # Plot 1 image for ground truth and generator output
        if X_model is None:
            params = np.tile(np.expand_dims(np.squeeze(X[p[i]]), 0), [n_sample, 1])
            noise = np.random.normal(size=[n_sample, par["n_noise"]])
            model_out = np.squeeze(model.sample_generator(sess, params, noise, is_training=is_training), -1)
            model_out = undo_data_normalisation(par, model_out)
        else:
            # Check if / where parameters exist in X_model
            if X_mean is not None and X_std is not None:
                pars_orig = undo_par_scaling(np.expand_dims(X[p[i]], 0), X_mean, X_std)
            else:
                pars_orig = X[p[i]]
            pars_in_X_model = np.where([np.all(np.isclose(X_model[j, :], pars_orig)) for j in range(X_model.shape[0])])[0]
            assert (len(pars_in_X_model) >= n_sample), "Not enough samples found!"
            rand_choice = np.random.choice(len(pars_in_X_model), size=n_sample)
            model_out = Y_model[pars_in_X_model[rand_choice]]

        y_par = X[p[i]]
        y_par_all = np.where([np.all(X[i, :] == y_par) for i in range(X.shape[0])])[0]
        y = Y[y_par_all]
        y = undo_data_normalisation(par, y) if norm_data else y
        color_real = np.take(cols_1, i, mode="wrap")
        color_fake = np.take(cols_2, i, mode="wrap")

        a_ = 1.0 / 16.0
        b_ = (1.0 / 7.0 - 1.0 / 16.0) / (y.shape[-1] - 1)
        all_pixels = np.arange(Y.shape[-1])
        all_redshifts = 1.0 / (a_ + all_pixels * b_) - 1
        pixels_z = np.asarray([np.argmin(np.abs(all_redshifts - z)) for z in redshifts])
        redshifts_ = 1.0 / (a_ + pixels_z * b_) - 1  # redshift range: 15 - 6

        for j_ax, j in enumerate(pixels_z):
            ps_y = np.abs(np.asarray([np.fft.rfft(y[i_sample, :, j]) for i_sample in range(y.shape[0])])) ** 2
            ps_y_mean = ps_y.mean(0)
            ps_y_std = ps_y.std(0)
            ps_out = np.abs(np.asarray([np.fft.rfft(model_out[i_sample, :, j]) for i_sample in range(model_out.shape[0])])) ** 2
            ps_out_mean = ps_out.mean(0)
            ps_out_std = ps_out.std(0)

            ax = axs[0, j_ax] if not np.sqrt(n_redshift) % 1 else axs[j_ax // 2, j_ax % 2]
            box_width = 200.0  # box width in Mpc / h
            k_vec = 2.0 * np.pi / box_width * np.arange(0, Y.shape[1]//2+1)
            ax.plot(k_vec[1:], k_vec[1:] / (2.0 * np.pi) * ps_y_mean[1:Y.shape[1]], color=color_real, label="real"+str(j_ax), lw=2.5)
            ax.plot(k_vec[1:], k_vec[1:] / (2.0 * np.pi) * ps_out_mean[1:Y.shape[1]], color=color_fake, label="fake" + str(j_ax), lw=2.5)  # 1 px is 6.25 Mpc/h high -> 32 px = 200 Mpc/h, -> k_nyquist = 1.005 h / Mpc / 2
            # ax.set_xlabel(r"$k \ [h \ \mathrm{Mpc}^{-1}]$", fontsize=20)
            if j_ax is 0:
                ax.set_ylabel(r"$\frac{k}{2 \pi} \, P(k) \ [\mathrm{mK^2}]$", fontsize=20)
            ax.set_title(r"$z = %g$" % np.round(redshifts_[j_ax], 1), fontsize=20)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.tick_params(labelsize=20)
            ax.tick_params(axis=u'both', which=u'major', length=15, pad=10)
            ax.tick_params(axis=u'both', which=u'minor', length=5, pad=10)

    if X_mean is not None and X_std is not None:
        params_plot = np.round(undo_par_scaling(X[p], X_mean, X_std), 1)
    else:
        params_plot = np.round(X[p], 1)
    leg_str = [""] * params_plot.shape[0]
    var_names = np.asarray([r"$f_X$", r"$r_{h/s}$", r"$f_\alpha$"])
    for i_set in range(params_plot.shape[0]):
        for i_var in range(params_plot.shape[1]):
            just_width = 7 if i_set > 0 and i_var is 0 else 6
            leg_str[i_set] += var_names[i_var] + " = " + str(np.round(params_plot[i_set, i_var], 1)).ljust(just_width)

    handles, _ = axs[0, 0].get_legend_handles_labels()
    handles_tuple = [tuple([handles[i], handles[i + 1]]) for i in range(0, 2 * params_plot.shape[0], 2)]
    axs[0, 0].legend(handles_tuple, np.asarray(leg_str), fontsize=18, handler_map={tuple: HandlerTuple(ndivide=None)}, ncol=1)
    ax_big.set_xlabel("\n\n" + r"$k \ [h \ \mathrm{Mpc}^{-1}]$", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    pretty_plots()
    plt.savefig(filename)
    plt.close("all")


# Pretty plots
def pretty_plots():
    # Get all figures
    # Plot settings
    mpl.rcParams['text.usetex'] = False
    mpl.rc('font', family='serif')
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \mathrm command
    mpl.rcParams['mathtext.default']='regular'
    mpl.rcParams['mathtext.fontset'] = 'dejavuserif'

    ########################################################################################################################
    ########################################################################################################################
    # Define settings
    font_family = 'serif'  # 'sans-serif', 'serif'
    for i in plt.get_fignums():
        gcf = plt.figure(i)

        # Get all axes
        for i_ax in range(len(gcf.axes)):

            # All axis labels in correct font
            gcf.axes[i_ax].xaxis.get_label().set_family(font_family)
            gcf.axes[i_ax].yaxis.get_label().set_family(font_family)

            # All ticklabels in correct font
            xticklabs = gcf.axes[i_ax].xaxis.get_ticklabels()
            for i_lab in range(len(xticklabs)):
                xticklabs[i_lab].set_family(font_family)

            yticklabs = gcf.axes[i_ax].yaxis.get_ticklabels()
            for i_lab in range(len(gcf.axes[i_ax].yaxis.get_ticklabels())):
                yticklabs[i_lab].set_family(font_family)

            # Everything for axes in correct font
            all_text = gcf.axes[i_ax].findobj(match=mpl.text.Text)
            for j in range(len(all_text)):
                all_text[j].set_family(font_family)

        # Everything for figure in correct font
        all_text_fig = gcf.findobj(match=mpl.text.Text)
        for k in range(len(all_text_fig)):
            all_text_fig[k].set_family(font_family)

    # plt.show()


# t-SNE visualisation of the ground truth images
# t-SNE
# from sklearn.manifold import TSNE
# dim = 2
# if dim is 3:
#     from mpl_toolkits.mplot3d import Axes3D
# tsne = TSNE(n_components=dim, init="random", perplexity=30)
# Y_embed = tsne.fit_transform(np.reshape(Y, [par["n_slices"], -1]))
# colours = np.zeros((Y_embed.shape[0], 3))
# colours[:, 0] = np.sqrt(X[:, 0]) / np.sqrt(10.0)
# colours[:, 1] = X[:, 1] ** 2 / 1.0
# colours[:, 2] = X[:, 2] ** 2 / 2.0 ** 2
# fig = plt.figure()
# col_dim = None
# if col_dim is not None:
#     if dim is 3:
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter3D(xs=Y_embed[:, 0], ys=Y_embed[:, 1], zs=Y_embed[:, 2], c=colours[:, col_dim])
#     else:
#         ax = fig.add_subplot(111)
#         ax.scatter(Y_embed[:, 0], Y_embed[:, 1], c=colours[:, col_dim])
# else:
#     if dim is 3:
#         ax = fig.add_subplot(111, projection='3d')
#         for j in range(Y_embed.shape[0]):
#             ax.scatter3D(xs=Y_embed[j, 0], ys=Y_embed[j, 1], zs=Y_embed[j, 2], c=colours[j, :].ravel())
#     else:
#         ax = fig.add_subplot(111)
#         for j in range(Y_embed.shape[0]):
#             ax.scatter(Y_embed[j, 0], Y_embed[j, 1], c=np.expand_dims(colours[j, :], 0))


# 21cm colourmap
def get_cmap21():
    from matplotlib.colors import ListedColormap
    c = np.asarray([[0.8, 1, 0.8],
                    [0, 1, 0.6],
                    [0, 0.2, 1],
                    [0, 0, 0],
                    [0.9, 0.17, 0],
                    [1, 0.8, 0],
                    [1, 1, 0.8]])
    pt = np.asarray([0, 0.1, 0.4, 0.5, 0.65, 0.8, 1.0])
    c_r, c_g, c_b = c.T
    out_pt = np.linspace(0, 1.0, 255)
    out_r = np.interp(out_pt, pt, c_r)
    out_g = np.interp(out_pt, pt, c_g)
    out_b = np.interp(out_pt, pt, c_b)
    colormap_data = np.vstack([out_r, out_g, out_b]).T
    return ListedColormap(colormap_data)

# Get default dictionary for generating samples
def get_default_dict():
    par = dict()
    par["input_type"] = 1
    par["name"] = ""
    par["filename"] = ""
    par["num_gpus"] = 1
    par["n_params"] = 3
    par["folder_in"] = ""
    par["n_slices"] = 0
    par["CPU"] = True
    par["fl_read"] = 6
    par["n_noise"] = 512
    par["n_channel"] = 1
    par["n_iter"] = None
    par["normalise_params"] = True
    par["normalise_data_in"] = False
    par["normalise_data_out"] = False
    par["keep_channels_constant"] = True
    par["legacy"] = False
    par["pixel_norm_init"] = 1
    par["pixel_norm"] = 3
    par["use_wscale"] = True
    par["std_init"] = 0.02
    par["out_dir"] = "output"
    par["checkpt_dir"] = "checkpoints"
    par["mirror_vertical"] = False
    par["lr_gen"] = par["lr_disc"] = par["beta1"] = par["beta2"] = 0.0
    par["n_batch"] = 0
    par["shuffle"] = False
    par["checkpt"] = "trained.ckpt"
    par["X_mean"] = [-0.0210721, 0.5, 0.0]  # for log_e(f_X), r_{h/s}, log_e(f_alpha)
    par["X_std"] = [1.62837806, 0.40824829, 0.5659523]  # for log_e(f_X), r_{h/s}, log_e(f_alpha)
    par["res_x"] = 1 * pow(2, par["fl_read"] - 1)
    par["res_z"] = 8 * pow(2, par["fl_read"] - 1)
    par["scale_par"] = 0.5
    par["data_mean"] = 0.0
    par["data_std"] = 30.0
    return par



