import os
import matplotlib as mpl
import sys
from NN_21cm import NN_21cm
from utils import *
from ops import build_pipeline, parse_function
import h5py as h5
print('Using non-interactive Agg backend')
mpl.use('Agg')
sys.path.append('/home/flo/PycharmProjects/21cm_new')
# os.environ["CUDA_VISIBLE_DEVICES"] ="0"  # set CUDA visible devices if needed

# SET HYPERPARAMETERS
par = dict()
par["input_type"] = 1       # if 0: load all data into memory from a single numpy file, downscale as needed on the go
                            # if 1: one HDF5 file is expected for each stage of the PGGAN
                            # if 2: one TFRecord file is expected for each stage of the PGGAN
par["name"] = "Test"  # Name of the experiment
par["n_params"] = 3  # Number of parameters
par["folder_in"] = "/home/flo/PycharmProjects/21cm/Data/high_res/Numpy/Downscaled"  # Name of folder with data
par["n_slices"] = 276480  # 17280  # number of slices available (only TFRecord)
par["num_gpus"] = 4  # Number of GPUs (if training on CPU, set to 1 and set set par["CPU"] = True)
par["n_iter"] = 40000  # number of iterations per architecture (e.g. 40000)
par["n_batch"] = 16  # batch size per GPU
par["filename"] = "_shuffled"  # "data_medium_res_all.npy"  # if data from numpy: file name of numpy file
                                                            # if data from HDF5: file name for each stage PG is "fl" + str(PG) + par["filename"] + ".h5"
                                                            # if data from TFRecords: file name for each stage PG is: "train.tfrecords_" + str(PG) + par["filename"]
par["CPU"] = False  # run on CPU instead of GPU(s)
par["no_grow"] = False  # deactivate growing architectures and train at fixed resolution
par["keep_channels_constant"] = True  # do not decrease channels as network grows
par["legacy"] = False  # if True: linear scaling of the data is assumed: x_orig = (x_train - 3/5) * 125
# par["fl"] = [6]  # no growing!
# par["fl_read"] = [6]  # no growing!
par["fl"] = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]  # flags for each GAN architecture
par["fl_read"] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]  # flag to read from for each GAN architecture
par["start_new_fl"] = False  # if True: continue with next stage of architecture without checking whether training on the latest stage has finished, if False: continue training at current stage
par["lr_gen"] = 1.0e-3  # learning rate for generator
par["lr_disc"] = 1.0e-3  # learning rate for discriminator
par["lr_fac"] = 0.0  # learning rate decay factor: LR = LR_0 * exp(-par["lr_fac"] * (PG-1))
par["n_noise"] = 512  # length of latent vector (including parameters)
par["n_disc"] = 1  # discriminator iterations per generator iterations
par["disc_add_channels"] = True  # if True: provide data - mean_vert(data) and mean_vert(data) as additional channels for discriminator
par["GAN_loss"] = "WGANGP"  # "GAN", "LSGAN", "WGAN", "WGANGP"
par["lambda_GP"] = 10.0  # lambda for gradient penalty for WGANGP
par["lambda_eps"] = 0.001  # weight for extra penalty term that keeps discriminator output close to 0 (see PGGAN paper)
par["n_channel"] = 1  # number of channels
par["normalise_params"] = True  # should the parameters normalised?
par["normalise_data_in"] = True  # should the data be normalised when reading? -> set to True for training (if data is not already saved scaled to [-1, 1]), False for evaluation
par["normalise_data_out"] = True  # should the normalisation be undone when plotting?
par["mirror_vertical"] = False  # if true: flip vertically with a probability of 50% (only TFRecords)
par["beta1"] = 0.0  # decay rate of first moment for Adam optimizer
par["beta2"] = 0.99  # decay rate of second moment for Adam optimizer
par["pixel_norm_init"] = 1  # pixel norm for initial layer? 0: False, 1: True
par["pixel_norm"] = 3  # pixel norm for other layers? 0: False, 1: True, 2: replace by group norm, 3: replace by instance norm
par["minibatch_std"] = True  # minibatch STD layer?
par["minibatch_group_size"] = 4  # group size for minibatch STD layer
par["sort_minibatch"] = [None]  # either array of None (no sorting) or parameter order by which to sort input for minibatch layer ([0, 2, 1]: fx, fa, rhs) (note: this feature has not been tested yet!)
par["fixed_param_batch"] = False  # from time to time: show a layer with the same parameters (note: this feature has not been tested yet!)
par["use_wscale"] = True  # weight scaling (see PGGAN paper)
par["std_init"] = 0.02  # STD for random initializers
par["shuffle"] = False  # shuffle data after reading (not necessary if slices are already saved shuffled)
par["n_shuffle"] = 17280  # size of shuffle buffer for TFRecords pipeline (does not need to be huge if data is saved in a shuffled way anyway)
par["logdir"] = "logs"  # directory for log files
par["checkpt_dir"] = "checkpoints"  # checkpoint directory
par["image_dir"] = "images"  # image folder
par["checkpt"] = "trained.ckpt"  # name of trained model
par["image_every"] = 2000  # output images every ... steps
par["save_every"] = 2000  # save model every ... steps

# For parameter scaling:
par["X_mean"] = [-0.0210721, 0.5, 0.0]  # for log_e(f_X), r_{h/s}, log_e(f_alpha)
par["X_std"] = [1.62837806, 0.40824829, 0.5659523]  # for log_e(f_X), r_{h/s}, log_e(f_alpha)

# Data scaling for color map in plots:
par["scale_par"] = 0.5  # centre the data around T_b = 0
par["data_mean"] = 0.0
par["data_std"] = 30.0

# Make folder
mkdir_p(os.path.join(par["checkpt_dir"], par["name"]))

# Find latest stage to continue training
check_path = os.path.join(par["checkpt_dir"], par["name"])
PG_0, has_non_t_0, max_it_0 = get_latest_stage(check_path)
par["resume_training"] = False

# If there are already saved checkpoints for this training
if PG_0 > 0:
    PG_start_all = np.where(par["fl"] == PG_0)[0]
    if par["no_grow"] and PG_0 != par["fl"][0]:
        i_0 = 0
    else:
        i_0 = PG_start_all.max() if has_non_t_0 else PG_start_all.min()
        par["resume_training"] = True
        # if previous stage has finished or if par["start_new_fl"], start next stage, otherwise continue stage
        if max_it_0 % par["n_iter"] is 0 or par["start_new_fl"]:
            i_0 += 1
            par["resume_training"] = False
else:
    i_0 = 0

# Store LR_0
LR_0_gen = par["lr_gen"]
LR_0_disc = par["lr_disc"]

# If all data is saved in a numpy file: load
if par["input_type"] is 0:
    data_all = np.load(os.path.join(par["folder_in"], par["filename"]), allow_pickle=True)
    X_all = data_all[()]["params"]
    Y_all = data_all[()]["data"]
    check_input(X_all, Y_all)
    print("Numpy file'" + os.path.join(par["folder_in"], par["filename"]) + "' loaded. Size of image array in memory: " + str(Y_all.nbytes // 1e6) + " MB.")

    if par["shuffle"]:
        perm = np.random.choice(par["n_slices"], par["n_slices"], replace=False)
        X_all = X_all[perm, :]
        Y_all = Y_all[perm, :, :]

    # Normalise data and parameters if needed
    Y_all, X_all = parse_function(Y_all, X_all, par)

# Progressive GAN
for i in range(i_0, len(par["fl"])):
    # If data is saved in a HDF5 file: load
    if par["input_type"] is 1:
        this_file = os.path.join(par["folder_in"], "fl" + str(par["fl"][i]) + par["filename"] + ".h5")
        with h5.File(this_file, 'r') as hf:
            Y_all = np.asarray(hf["data"])
            X_all = np.asarray(hf["params"])
            check_input(X_all, Y_all)
            print("HDF5 file '" + this_file + "' loaded. Size of image array in memory: " + str(Y_all.nbytes // 1e6) + " MB.")

        if par["shuffle"]:
            perm = np.random.choice(par["n_slices"], par["n_slices"], replace=False)
            X_all = X_all[perm, :]
            Y_all = Y_all[perm, :, :]

        # Normalise data and parameters if needed
        Y_all, X_all = parse_function(Y_all, X_all, par)

    # Set parameters for this stage
    par["res_x"] = 1 * pow(2, par["fl"][i] - 1)  # current resolution in x, base resolution is 1
    par["res_z"] = 8 * pow(2, par["fl"][i] - 1)  # current resolution in z, base resolution is 8

    # Update learning rates
    par["lr_gen"] = LR_0_gen * np.exp(-(par["fl"][i]-1) * par["lr_fac"])
    par["lr_disc"] = LR_0_disc * np.exp(-(par["fl"][i]-1) * par["lr_fac"])

    # Transitional stage?
    t = (i % 2 is not 0)

    # Define current directories
    checkpt_dir_write = os.path.join(par["checkpt_dir"], par["name"], str(par["fl"][i]) + "_" + str(t))
    if par["resume_training"]:
        checkpt_dir_read = checkpt_dir_write
        start_it = max_it_0 % par["n_iter"]
    else:
        checkpt_dir_read = os.path.join(par["checkpt_dir"], par["name"], str(par["fl_read"][i]) + "_" + str(not t))
        start_it = 0
    im_dir = os.path.join(par["image_dir"], par["name"], str(par["fl"][i]) + "_" + str(t))
    mkdir_p(checkpt_dir_write)
    if par["fl"][i] > 1 and par["no_grow"] is False:
        mkdir_p(checkpt_dir_read)
    mkdir_p(im_dir)

    # If data comes from TFRecord files
    if par["input_type"] is 2:
        # Build input pipeline
        data, params = build_pipeline(par, par["fl"][i])
        # Build model
        model = NN_21cm(par, params, checkpt_dir_write, checkpt_dir_read, im_dir, t=t, PG=par["fl"][i], d_inputs=tf.expand_dims(data, -1))
        # Train model
        model.train(data, params, start_it)

    # If data comes from a numpy / HDF5 file
    else:
        # Build model
        model = NN_21cm(par, None, checkpt_dir_write, checkpt_dir_read, im_dir, t=t, PG=par["fl"][i])
        # Train model
        model.train(Y_all, X_all, start_it)

    # Switch off resume_training
    par["resume_training"] = False