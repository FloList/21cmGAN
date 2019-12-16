import numpy as np
import h5py as h5

folder_in = "./high_res/Numpy/Downscaled/"
res = range(4)
np.random.seed(20192019)
rand_perm = np.random.permutation(276480)

for i_res in res:
    ff = h5.File(folder_in + "fl" + str(i_res+1) + ".h5", "r")
    data = ff["data"]
    params = ff["params"]
    data_new = np.asarray(data)[rand_perm]
    params_new = np.asarray(params)[rand_perm]
    ff_out = h5.File(folder_in + "fl" + str(i_res+1) + "_shuffled.h5", "w")
    ff_out.create_dataset('data', data=data_new, compression="gzip")
    ff_out.create_dataset('params', data=params_new, compression="gzip")
