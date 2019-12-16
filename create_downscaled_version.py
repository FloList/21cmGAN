import numpy as np
import os
from skimage.transform import resize

# Set folders
folder_in = "/short/u95/fl9575/Output/21cm/Data/high_res/Numpy"
folder_out = os.path.join(folder_in, "Downscaled")
format = 2  # 0: Numpy, 1: pickle, 2: HDF5
if format is 1:
    import pickle
elif format is 2:
    import h5py

data_full = np.zeros([0, 0, 0])
params_full = np.zeros([0, 0])

all_files = np.asarray(os.listdir(folder_in))
all_files = all_files[np.where([files.endswith(".npy") for files in all_files])[0]]

this_im = np.load(os.path.join(folder_in, all_files[0]))
data_full = this_im[()]["data"]
params_full = this_im[()]["params"]


for file in all_files[1:].tolist():
    this_im = np.load(os.path.join(folder_in, file))
    data_full = np.concatenate([data_full, this_im[()]["data"]], axis=0)
    params_full = np.concatenate([params_full, this_im[()]["params"]], axis=0)


dict_out = dict()
res_array = range(7)
res_x = lambda i: 1 * pow(2, i)
res_z = lambda i: 8 * pow(2, i)

for i_res in res_array:
    data_downscaled = np.asarray([resize(data_full[j], output_shape=[res_x(i_res), res_z(i_res)], mode="constant", anti_aliasing=False) for j in range(data_full.shape[0])])
    dict_out["data"] = data_downscaled
    dict_out["params"] = params_full

    if format is 0:
        np.save(os.path.join(folder_out, "fl" + str(i_res + 1)), dict_out)
    elif format is 1:
        with open(os.path.join(folder_out, "fl" + str(i_res + 1) + ".dat"), 'wb') as outfile:
            pickle.dump(dict_out, outfile, pickle.HIGHEST_PROTOCOL)
    elif format is 2:
        hf = h5py.File(os.path.join(folder_out, "fl" + str(i_res + 1) + ".h5"), 'w')
        hf.create_dataset('data', data=data_downscaled, compression="gzip")
        hf.create_dataset('params', data=params_full, compression="gzip")


# TEST
# f1 = np.load("/home/flo/PycharmProjects/21cm_new/fx=0.1_RHS=0_fa=0.5_x_lightcone_dtb_fullres.dat.npy")
# data = f1[()]["data"]
# params = f1[()]["params"]
