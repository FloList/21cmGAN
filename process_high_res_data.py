import numpy as np
import wget
# import shlex
# import subprocess
# import ssl
import struct
import os
from skimage.transform import resize

# NOTE: THE FILES ARE STORED WITHOUT RESCALING!!!
# This script takes the .dat files as an input and output slices of a given resolution, one file per .dat file.

# Functions
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        # >>> # linear interpolation of NaNs
        # >>> nans, x = nan_helper(y)
        # >>> y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interp_nans(slice, dim_x, dim_z):
    """ Interpolate NaNs"""
    nans, slice_new = nan_helper(np.reshape(slice, [-1]))
    interp_values = np.interp(slice_new(nans), slice_new(~nans), np.reshape(slice, [-1])[~nans])
    slice = np.reshape(slice, [-1])
    slice[nans] = interp_values
    return np.reshape(slice, [dim_x, dim_z])


def floats_from_file(folder, filename, chunksize):
    with open(os.path.join(folder, filename), "rb") as f:
        return struct.unpack('f'*chunksize, f.read(4*chunksize))


# def bar_custom(current, total, width=80):
#     print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total))

folder_url = "https://21ssd.obspm.fr/browse/21ssd/lightcones/"
folder_save = "/home/flo/PycharmProjects/21cm/Data/high_res"
fx_values = ('0.1', '0.3', '1', '3', '10')
rhs_values = ('0', '0.5', '1')
fa_values = ('0.5', '1', '2')
coord_values = ('x', 'y', 'z')

remove_nans = True
# dim_x, dim_y, dim_z = 16, 16, 128  # low res for DEBUG
dim_x, dim_y, dim_z = 1024, 1024, 8096  # high res
out_res = 64, 512
delta_slice = 1
delete_dat = False
data_augmentation = False

# Initialise
slice_ind_x = np.arange(0, dim_x, delta_slice)
slice_ind_y = np.arange(0, dim_y, delta_slice)
pix = np.zeros((len(slice_ind_x) + len(slice_ind_y), out_res[0], out_res[1]))
params = np.zeros((len(slice_ind_x) + len(slice_ind_y), 3))
i_param = 0

# ssl._create_default_https_context = ssl._create_unverified_context

for i_fx, fx in enumerate(fx_values):
    for i_rhs, rhs in enumerate(rhs_values):
        for i_fa, fa in enumerate(fa_values):
            for i_coord, coord in enumerate(coord_values):
                filename = "fx=" + fx + "_RHS=" + rhs + "_fa=" + fa + "_" + coord + "_lightcone_dtb_fullres.dat"
                if not os.path.exists(os.path.join(folder_save, filename)):
                    filename_url = folder_url + filename
                    print("Downloading file", filename_url)
                    wget.download(filename_url, folder_save)
                #     subprocess.call(['./download_file.sh', filename_url])

                # Process file
                lightcone = floats_from_file(folder_save, filename, dim_x * dim_y * dim_z)
                lightcone = np.transpose(np.reshape(np.asarray(lightcone), (dim_z, dim_y, dim_x)), (2, 1, 0))
                i_slice = 0
                while i_slice < len(slice_ind_x):
                    slice = lightcone[slice_ind_x[i_slice], :, :]
                    if remove_nans and np.any(np.isnan(slice)):
                        slice = interp_nans(slice, dim_x, dim_z)
                    slice = resize(slice, output_shape=out_res, mode="constant", anti_aliasing=False)
                    pix[i_slice, :, :] = slice
                    params[i_param, :] = [float(fx), float(rhs), float(fa)]
                    i_slice += 1
                    i_param += 1
                i_slice = 0
                while i_slice < len(slice_ind_y):
                    slice = lightcone[slice_ind_y[i_slice], :, :]
                    if remove_nans and np.any(np.isnan(slice)):
                        slice = interp_nans(slice, dim_x, dim_z)
                    slice = resize(slice, output_shape=out_res, mode="constant", anti_aliasing=False)
                    pix[i_slice + len(slice_ind_x), :, :] = slice
                    params[i_param, :] = [float(fx), float(rhs), float(fa)]
                    i_slice += 1
                    i_param += 1
                i_slice = 0

                # Mirror
                if data_augmentation:
                    pix_mirror = np.asarray([np.flipud(pix[i, :, :]) for i in range(pix.shape[0])])
                    pix_out = np.concatenate([pix, pix_mirror], 0)
                    params_out = np.tile(params, [2, 1])
                else:
                    pix_out = pix
                    params_out = params

                # Save
                dict_out = dict()
                dict_out["data"] = pix_out
                dict_out["params"] = params_out
                np.save(os.path.join(folder_save, "Numpy", filename), dict_out)

                # Delete .dat file
                print("Data file ", os.path.join(folder_save, filename), "processed.")
                if delete_dat:
                    os.remove(os.path.join(folder_save, filename))
                    print("Data file ", os.path.join(folder_save, filename), "deleted.")

                i_param = 0

# Debug
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(params.shape[0], 1)
# for i in range(8):
#     axs[i].imshow(pix_out[i])
