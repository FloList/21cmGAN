import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend')
    mpl.use('Agg')
else:
    mpl.use('TkAgg')
mpl.use('Agg')
import sys
sys.path.append('/home/flo/PycharmProjects/21cm_new')
from NN_21cm import NN_21cm
from utils import *
from ops import build_pipeline, parse_function
import h5py as h5
import random
from time import gmtime, strftime

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TASKS
TASK = 1    # 0: do_analysis, 1: do_ABC, 2: do_sample
do_analysis = True if TASK is 0 else False
do_ABC = True if TASK is 1 else False
do_sample = True if TASK is 2 else False

# SET HYPERPARAMETERS
# seed = 2019201920
# np.random.seed(seed)
par = dict()
par["input_type"] = 1       # if 0: load all data into memory from a single numpy file, downscale as needed on the go
                            # if 1: one HDF5 file is expected for each stage of the PGGAN
                            # if 2: one TFRecord file is expected for each stage of the PGGAN
par["name"] = "HDF5_nonlin"  # Name of the experiment
par["filename"] = "_shuffled"  # "data_medium_res_all.npy"  # if data from numpy: file name of numpy file
                                                            # if data from HDF5: file name for each stage PG is "fl" + str(PG) + par["filename"] + ".h5"
                                                            # if data from TFRecords: file name for each stage PG is: "train.tfrecords_" + str(PG) + par["filename"]
par["num_gpus"] = 1  # For evaluation: 1 GPU / CPU
par["n_params"] = 3  # Number of parameters
par["folder_in"] = "/home/flo/PycharmProjects/21cm/Data/high_res/Numpy/Downscaled"  # Name of folder with data (/home/flo/PycharmProjects/21cm/Data/high_res/Numpy/Downscaled)
par["n_slices"] = 276480  # 17280  # 276480 # number of slices available (only TFRecord)
par["CPU"] = True  # evaluation on CPU
par["fl_read"] = 6  # architecture to read (for evaluation: always transition == False)
par["n_noise"] = 512  # size of latent vector
par["n_channel"] = 1  # number of channels
par["n_iter"] = None  # set to None when not training
par["normalise_params"] = True  # should the parameters be normalised?
par["normalise_data_in"] = False  # should the data be normalised when reading? -> set to True for training, False for evaluation
par["normalise_data_out"] = False  # should the normalisation be undone when plotting?
par["keep_channels_constant"] = True  # do not decrease channels as network grows
par["legacy"] = False  # if legacy: linear scaling of data is assumed: x_orig = (x_train - 3/5) * 125
par["pixel_norm_init"] = 1  # pixel norm for initial layer? 0: False, 1: True
par["pixel_norm"] = 3  # pixel norm for other layers? 0: False, 1: True, 2: replace by group norm, 3: replace by instance norm
par["use_wscale"] = True  # weight scaling (see PGGAN paper)
par["std_init"] = 0.02  # STD for random initializers
par["out_dir"] = "output"  # directory for output files
par["checkpt_dir"] = "checkpoints"  # checkpoint directory
par["mirror_vertical"] = False  # not needed for evaluation
par["lr_gen"] = par["lr_disc"] = par["beta1"] = par["beta2"] = 0.0  # learning rate for generator
par["n_batch"] = par["n_slices"]  # get all available slices for inference
par["shuffle"] = False  # shuffle data randomly
par["checkpt"] = "trained.ckpt"  # name of trained model
dist_fun = dist_all  # see distances in utils
# NOTE on dist_critic_loss: in case of a minibatch std layer in the discriminator, the composition and size of
# the minibatches at inference time heavily influences the output! Therefore, don't use a fixed epsilon for the ABC, but
# rather draw samples quantile-based!

# The following hyper parameters are only needed if dist_fun == dist_critic_loss
par["disc_add_channels"] = True  # if True: provide data - mean_x(data) and mean_x(data) as additional channels for discriminator
par["sort_minibatch"] = [None]  # either array of None (no sorting) or parameter order by which to sort input for minibatch layer ([0, 2, 1]: fx, fa, rhs)
par["lambda_GP"] = 0.0  # lambda for gradient penalty
par["lambda_eps"] = 0.0  # weight for extra term in order to keep discriminator output close to 0
par["GAN_loss"] = "WGANGP"  # "GAN", "LSGAN", "WGAN", "WGANGP"
par["minibatch_std"] = True  # if False: replace minibatch std layer by zeroes
par["minibatch_group_size"] = 4  # group size for minibatch STD layer

# For parameter scaling (see main.py):
X_mean = par["X_mean"] = [-0.0210721, 0.5, 0.0]  # for log_e(f_X), r_{h/s}, log_e(f_alpha)
X_std = par["X_std"] = [1.62837806, 0.40824829, 0.5659523]  # for log_e(f_X), r_{h/s}, log_e(f_alpha)

# Data scaling for color bars in plots:
par["scale_par"] = 0.5  # centre the data around T_b = 0
par["data_mean"] = 0.0
par["data_std"] = 30.0

# Set parameters for this stage
par["res_x"] = 1 * pow(2, par["fl_read"] - 1)  # current resolution in x, base resolution is 1
par["res_z"] = 8 * pow(2, par["fl_read"] - 1)  # current resolution in z, base resolution is 8

# If task is not sampling: load data
if not do_sample or par["input_type"] is 2:
    # If all data is saved in a numpy file: load
    if par["input_type"] is 0:
        data_all = np.load(os.path.join(par["folder_in"], par["filename"]), allow_pickle=True)
        X_all = data_all[()]["params"]
        Y_all = data_all[()]["data"]
        check_input(X_all, Y_all)

    # If data is saved in a HDF5 file: load
    elif par["input_type"] is 1:
        this_file = os.path.join(par["folder_in"], "fl" + str(par["fl_read"]) + par["filename"] + ".h5")
        with h5.File(this_file, 'r') as hf:
            Y_all = np.asarray(hf["data"])
            X_all = np.asarray(hf["params"])
            check_input(X_all, Y_all)

    # Shuffle and normalise
    if par["input_type"] < 2:
        if par["shuffle"]:
            perm = np.random.choice(par["n_slices"], par["n_slices"], replace=False)
            X_all = X_all[perm, :]
            Y_all = Y_all[perm, :, :]

        # Normalise data and parameters if needed
        Y_all, X_all = parse_function(Y_all, X_all, par)

    else:
        # Build pipeline
        data, params = build_pipeline(par, PG=par["fl_read"])

# Save resolution
par["res"] = [par["res_x"], par["res_z"]]

# Make output folder
im_dir = os.path.join(par["out_dir"], par["name"], str(par["fl_read"]) + "_False")
mkdir_p(im_dir)

# Build model
checkpt_dir_read = os.path.join(par["checkpt_dir"], par["name"], str(par["fl_read"]) + "_False")
only_eval = False if dist_fun is dist_critic_loss else True
if par["input_type"] < 2:
    model_21cm = NN_21cm(par, None, "", checkpt_dir_read, im_dir, t=False, PG=par["fl_read"], only_eval=only_eval)
else:
    model_21cm = NN_21cm(par, params, "", checkpt_dir_read, im_dir, t=False, PG=par["fl_read"], only_eval=only_eval, d_inputs=tf.expand_dims(data, -1))

# Start session
init = tf.global_variables_initializer()
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
sess.run(init)

# Load saved model
saver = model_21cm.saver if dist_fun is dist_critic_loss else model_21cm.saver_eval
if os.path.exists(model_21cm.checkpt_dir_read):
    if tf.train.latest_checkpoint(model_21cm.checkpt_dir_read) is not None:
        tf.logging.info('Loading checkpoint from ' + tf.train.latest_checkpoint(model_21cm.checkpt_dir_read))
        saver.restore(sess, tf.train.latest_checkpoint(model_21cm.checkpt_dir_read))
    else:
        raise FileNotFoundError
else:
    raise FileNotFoundError

# # # # # # # # # # Define sampling settings and sample # # # # # # # # # #
if do_sample:
    sampling = dict()
    sampling["pars"] = np.asarray([[10, 0, 2], [10, 1, 0.5], [3, 0.5, 1], [1, 0.5, 1], [0.3, 0.5, 1], [0.1, 0, 0.5], [0.1, 1, 2]])  # parameter sets
    if par["normalise_params"]:
        sampling["pars"] = scale_pars(sampling["pars"], mean=X_mean, std=X_std)[0]
    sampling["n_samples"] = 128  # no. of samples for each parameter
    sampling["n_batch"] = 64  # how many samples shall be created in one go?
    sampling["rand_name"] = True  # random file name?
    model_out = np.zeros((0, par["res_x"], par["res_z"]))
    params_out = np.zeros((0, par["n_params"]))

    for pars in sampling["pars"]:
        for i_draw in range(sampling["n_samples"] // sampling["n_batch"]):
            noise = np.random.normal(0, 1, [sampling["n_batch"], par["n_noise"]])
            params_feed = np.tile(np.expand_dims(pars, 0), [sampling["n_batch"], 1])
            samples_tmp = undo_data_normalisation(par, np.squeeze(model_21cm.sample_generator(sess, params_feed, noise, is_training=False), -1))
            params_tmp = undo_par_scaling(params_feed, X_mean, X_std)
            model_out = np.concatenate([model_out, samples_tmp], axis=0)
            params_out = np.concatenate([params_out, params_tmp], axis=0)
            current_perc = np.round(100 * params_out.shape[0] / (sampling["pars"].shape[0] * sampling["n_samples"]), 2)
            print(str(params_out.shape[0]) + " out of " + str((sampling["pars"].shape[0] * sampling["n_samples"])) + " samples generated (" + str(current_perc) + " per cent).")

    sample_folder = os.path.join(par["out_dir"], par["name"], str(par["fl_read"]) + "_False", "samples")
    mkdir_p(sample_folder)
    if sampling["rand_name"]:
        out_file = os.path.join(sample_folder, strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + str(np.round(random.uniform(0, 1e10)))[:-2].zfill(10) + "_samples.h5")
    else:
        out_file = os.path.join(sample_folder, "samples.h5")
    hf = h5.File(out_file, 'w')
    hf.create_dataset('data', data=model_out, compression="gzip")
    hf.create_dataset('params', data=params_out, compression="gzip")
    exit(0)


# Get data
if par["input_type"] < 2:
    Y, X_feed = Y_all, X_all
else:
    Y, X_feed = sess.run([data, params])

# # # # # # # # # # Define analysis settings # # # # # # # # # #
if do_analysis:
    # Colors
    cols = np.asarray(['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#ca82d6', '#6a3d9a', '#cccc99', '#b15928', '#c9d9d9', '#000000'])

    analysis = dict()
    analysis["cols"] = cols
    analysis["load_from_file"] = True  # if True: load samples from os.path.join(par["out_dir"], par["name"], str(par["fl_read"]) + "_False", "samples", "samples.h5")
    if analysis["load_from_file"]:
        sample_file = os.path.join(par["out_dir"], par["name"], str(par["fl_read"]) + "_False", "samples", "samples.h5")
        with h5.File(sample_file, 'r') as hf:
            Y_model = np.asarray(hf["data"])
            X_model = np.asarray(hf["params"])
            check_input(X_model, Y_model)
    else:
        Y_model = X_model = None

    # Random plot
    analysis["do_output"] = False  # plot random samples
    output_sets = np.asarray([[10, 0, 2], [10, 1, 0.5], [3, 0.5, 1], [1, 0.5, 1], [0.3, 0.5, 1], [0.1, 0, 0.5], [0.1, 1, 2]])  # parameter sets
    if par["normalise_params"]:
        output_sets = scale_pars(output_sets, mean=X_mean, std=X_std)[0]
    p_for_out = [None] * output_sets.shape[0]
    for i_par, out_par in enumerate(output_sets):
        p_curr = np.where([np.allclose(X_feed[i, :], out_par) for i in range(par["n_batch"])])[0]
        p_for_out[i_par] = p_curr[np.random.choice(len(p_curr))]

    analysis["output"] = p_for_out

    # Multiple samples for fixed sets of parameters
    analysis["do_fixed"] = False  # plot samples for fixed parameters
    analysis["n_fixed"] = 6  # number of samples to plot
    analysis["fixed"] = np.asarray([[10, 0, 2], [10, 1, 0.5], [3, 0.5, 1], [1, 0.5, 1], [0.3, 0.5, 1], [0.1, 0, 0.5], [0.1, 1, 2]])  # parameter sets
    if par["normalise_params"]:
        analysis["fixed"] = scale_pars(analysis["fixed"], mean=X_mean, std=X_std)[0]

    # Average Delta T as a function of redshift
    analysis["do_avg"] = False  # plot mean
    analysis["n_avg"] = 24  # number of samples to average over
    avg_par_sets = analysis["fixed"]
    p_for_avg = [None] * avg_par_sets.shape[0]
    for i_par, avg_par in enumerate(avg_par_sets):
        p_curr = np.where([np.allclose(X_feed[i, :], avg_par) for i in range(par["n_batch"])])[0]
        p_for_avg[i_par] = p_curr[np.random.choice(len(p_curr))]
    analysis["avg"] = p_for_avg
    analysis["cols_avg"] = cols

    # Interpolate in parameter space
    analysis["do_interp"] = False  # plot interpolated samples
    analysis["n_interp"] = 4
    analysis["interp"] = np.asarray([[[0.1, 0.5, 1], [3, 0.5, 1]], [[1, 0, 1], [1, 1, 1]], [[1, 0.5, 0.5], [1, 0.5, 2]]])
    analysis["interp_all"] = True  # all interpolations in one plot
    if par["normalise_params"]:
        analysis["interp"] = np.transpose(np.asarray([scale_pars(analysis["interp"][:, i, :], mean=X_mean, std=X_std)[0] for i in range(2)]), [1, 0, 2])

    # Plot PDFs
    analysis["do_PDF"] = False  # plot point distribution function (PDF)
    analysis["n_PDF"] = 24  # number of samples to consider
    analysis["PDF"] = analysis["fixed"]  # parameter sets

    # Plot histogram of pixel distribution
    analysis["do_hist"] = True  # plot point distribution function (PDF)
    analysis["n_hist"] = 24  # number of samples to consider
    analysis["hist_z"] = [9.5, 8.0]  # number of redshifts to plot
    for_hist = [0, 3, 6]
    analysis["hist"] = np.asarray(p_for_avg)[for_hist]  # parameter sets
    col_vec_hist = np.asarray([[cols[2*for_hist[k]]] + [cols[2*for_hist[k]+1]] for k in range(len(for_hist))]).flatten()
    analysis["cols_hist"] = col_vec_hist

    # Plot power spectrum
    analysis["do_power"] = True  # plot power spectrum
    analysis["n_power"] = 24  # number of samples to consider
    analysis["power_z"] = [9.5, 8.0]  # number of redshifts to plot
    for_power = [0, 3, 6]
    analysis["power"] = np.asarray(p_for_avg)[for_power]  # parameter sets
    col_vec_power = np.asarray([[cols[2*for_power[k]]] + [cols[2*for_power[k]+1]] for k in range(len(for_power))]).flatten()
    analysis["cols_power"] = col_vec_power

    # Run analysis
    model_21cm.run_analysis(sess, X_feed, Y, analysis, X_mean=X_mean, X_std=X_std, X_model=X_model, Y_model=Y_model)

# # # # # # # # # # ABC model  # # # # # # # # # #
if do_ABC:
    abc = dict()

    # Set true theta (a random sample corresponding to these parameters will be chosen)
    abc["theta_truth"] = np.asarray([[1.0, 0.5, 1]])
    abc["take_first_sample"] = True  # if True, the first found image corresponding to the above parameters is chosen, otherwise: random sample

    # Define prior limits (uniform priors)
    abc["prior_lim"] = np.asarray([[0.1, 0, 0.5], [10, 1, 2]])

    # Define a set of example parameters to output distance
    abc["theta_test"] = np.asarray([[10, 0, 2], [10, 1, 0.5], [3, 0.5, 1], [1, 0.0, 1],  [1, 1.0, 1], [0.3, 0.5, 1], [0.1, 0, 0.5], [0.1, 1, 2]])  # parameter sets

    # Define distance function
    abc["dist_fun"] = dist_fun
    abc["dist_fun_critic"] = True if dist_fun is dist_critic_loss else False
    abc["n_metrics"] = 4 if dist_fun is dist_all else 1

    # Set path for saving samples
    abc["abc_path"] = os.path.join(par["out_dir"], par["name"], str(par["fl_read"]) + "_False", "abc")
    mkdir_p(abc["abc_path"])

    # Set settings for rejection sampling: SET EITHER abc["eps_rej"] OR abc["n_accept"]!
    abc["eps_rej"] = None  # Rejection tolerance
    abc["n_accept"] = 16  # Always accept the fittest n_accept samples
    abc["n_samples"] = 48  # no. of total samples for posterior distribution
    abc["n_batch"] = 16  # no. of samples to draw at the same time, POWER OF 2 in case minibatch STD layer is used!

    # Run rejection sampling
    abc_samples, abc_dist, abc_im = model_21cm.run_abc(sess, X_feed, Y, abc, X_mean=X_mean, X_std=X_std)

    # Extract smaller population?
    dist_column = 1
    abc_ind_sorted = np.argsort(abc_dist[:, dist_column], axis=0)
    abc_samples_sorted = np.asarray(abc_samples)[abc_ind_sorted.flatten()]
    extract_samples = np.arange(25)

    # Corner plot
    import corner
    fig_corner = corner.corner(undo_par_scaling(abc_samples_sorted[extract_samples], X_mean, X_std), labels=[r"$f_x$", r"$r_{h/s}$", r"$f_\alpha$"],
                               quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12}, truths=abc["theta_truth"].flatten())
    fig_corner.savefig(os.path.join(abc["abc_path"], "corner.pdf"))
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
