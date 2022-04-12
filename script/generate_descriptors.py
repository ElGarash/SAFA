import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from spatial_net import *
from input_data_cvusa import InputData
import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import pickle

tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser(description="TensorFlow implementation.")

parser.add_argument("--network_type", type=str, help="network type", default="SAFA_8")
parser.add_argument("--polar", type=int, help="polar", default=1)

args = parser.parse_args()

# --------------  configuration parameters  -------------- #
network_type = args.network_type
polar = args.polar

batch_size = 32
is_training = False

DESCRIPTORS_DIRECTORY = "/kaggle/working/descriptors/SAFA"

# -------------------------------------------------------- #

if __name__ == "__main__":
    if os.path.exists(f"{DESCRIPTORS_DIRECTORY}/satellite_descriptors.pkl"):
        print("Satellite descriptor already exists on the file system.")
        exit(0)

    tf.reset_default_graph()
    # import data
    input_data = InputData(polar)

    # define placeholders
    if polar:
        sat_x = tf.placeholder(tf.float32, [None, 112, 616, 3], name="sat_x")
    else:
        sat_x = tf.placeholder(tf.float32, [None, 256, 256, 3], name="sat_x")

    grd_x = tf.placeholder(tf.float32, [None, 112, 616, 3], name="grd_x")

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # build model
    dimension = int(network_type[-1])
    sat_global, grd_global = SAFA(sat_x, grd_x, keep_prob, dimension, is_training)

    out_channel = sat_global.get_shape().as_list()[-1]
    sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])
    grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])

    print("setting saver...")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    print("setting saver done...")

    # run model
    print("run model...")
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    print("open session ...")
    with tf.Session(config=config) as sess:
        print("initialize...")
        sess.run(tf.global_variables_initializer())

        print("loading model...")

        load_model_path = "/kaggle/working/models/SAFA/CVUSA/Trained/model.ckpt"
        saver.restore(sess, load_model_path)

        print("validating...")
        print("computing global descriptors...")
        input_data.reset_scan()

        val_i = 0
        while True:
            print("progress %d" % val_i)
            batch_sat, batch_grd = input_data.next_batch_scan(batch_size)
            if batch_sat is None:
                break
            feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
            sat_global_val, grd_global_val = sess.run(
                [sat_global, grd_global], feed_dict=feed_dict
            )

            sat_global_descriptor[
                val_i : val_i + sat_global_val.shape[0], :
            ] = sat_global_val

            grd_global_descriptor[
                val_i : val_i + grd_global_val.shape[0], :
            ] = grd_global_val

            val_i += sat_global_val.shape[0]

        dist_array = 2 - 2 * np.matmul(
            sat_global_descriptor, np.transpose(grd_global_descriptor)
        )

        if not os.path.exists(DESCRIPTORS_DIRECTORY):
            os.makedirs(DESCRIPTORS_DIRECTORY, exist_ok=True)

        # store descriptors and distance matrices
        with open(f'{DESCRIPTORS_DIRECTORY}/dist_array_total.pkl', 'wb') as f:
            pickle.dump(dist_array, f)
        
        with open(f'{DESCRIPTORS_DIRECTORY}/ground_descriptors.pkl', 'wb') as f:
            pickle.dump(grd_global_descriptor, f)
        
        with open(f'{DESCRIPTORS_DIRECTORY}/satellite_descriptors.pkl', 'wb') as f:
            pickle.dump(sat_global_descriptor, f)