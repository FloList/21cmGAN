import os
import numpy as np
import h5py as h5
import tensorflow as tf

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# Input data
folder_in = "/home/flo/PycharmProjects/21cm/Data/high_res/Numpy/Downscaled"
stages = range(1, 8)

for stage in stages:
    this_file = os.path.join(folder_in, "fl" + str(stage) + "_shuffled.h5")
    with h5.File(this_file, 'r') as hf:
        Y = np.asarray(hf["data"])
        X = np.asarray(hf["params"])
        print("File '" + this_file + "' loaded. Size of image array in memory: " + str(Y.nbytes // 1e6) + " MB.")

        name = "train.tfrecords_" + str(stage)
        filename = os.path.join(folder_in, name)
        tfrecord_writer = tf.python_io.TFRecordWriter(filename)

        n_samples = X.shape[0]
        rows = Y.shape[1]
        cols = Y.shape[2]

        for index in range(n_samples):
            # 1. Convert data into tf.train.Feature
            Y_raw = Y[index].flatten()  #.tostring()
            X_raw = X[index].flatten()  #.tostring()

            feature = {
                'params_raw': _floats_feature(X_raw),
                'image_raw': _floats_feature(Y_raw)
            }

            # 2. Create a tf.train.Features
            features = tf.train.Features(feature=feature)
            # 3. Createan example protocol
            example = tf.train.Example(features=features)
            # 4. Serialize the Example to string
            example_to_string = example.SerializeToString()
            # 5. Write to TFRecord
            tfrecord_writer.write(example_to_string)

# Test
# filename = '/home/flo/PycharmProjects/21cm/Data/high_res/Numpy/Downscaled/train.tfrecords_1'
# def decode(serialized_example):
#     # 1. define a parser
#     features = tf.parse_single_example(
#         serialized_example,
#         # Defaults are not specified since both keys are required.
#         features={
#             'params_raw': tf.VarLenFeature(tf.float32),
#             'image_raw': tf.VarLenFeature(tf.float32),
#         })
#
#     # 2. Convert the data
#     image = tf.sparse_tensor_to_dense(features['image_raw'], default_value=0)
#     params = tf.sparse_tensor_to_dense(features['params_raw'], default_value=0)
#
#     # 3. Reshape
#     image.set_shape((8))
#     image = tf.reshape(image, [1, 8])
#     params.set_shape(3)
#     return image, params
#
# dataset = tf.data.TFRecordDataset(filename)
# dataset = dataset.map(decode)
