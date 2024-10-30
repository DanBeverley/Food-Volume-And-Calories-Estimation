import time
import os, shutil
import numpy as np
import tensorflow as tf
import sys
slim = tf.contrib.slim
sys.path.append(os.getcwd())
from nets import model as model
import matplotlib.pyplot as plt
from utils.pascal_voc import pascal_segmentation_lut
from utils.visualization import visualize_segmentation_adaptive
from utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors

tf.app.flags.DEFINE_string("test_data_path", "data/pascal_augmented_train.tfrecords","")
tf.app.flags.DEFINE_string("gpu_list", "0","")
tf.app.flags.DEFINE_integer("num_classes", 21, "")
tf.app.flags.DEFINE_string("checkpoint_path", "checkpoints/","")
tf.app.flags.DEFINE_string("result_path", "result/", "")
tf.app.flags.DEFINE_integer("test_size", 384, "")

FLAGS = tf.app.flags.FLAGS

def main(argv = None):
    os.environ("CUDA_VISIBLE_DEVICES") = FLAGS.gpu_list
    pascal_voc_lut = pascal_segmentation_lut()

    filename_queue = tf.train.string_input_producer([FLAGS.test_data_path], num_epochs = 1)
    image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

    image_batch_tensor = tf.expand_dims(image, axis = 0)
    annotation_batch_tensor = tf.expand_dims(annotation, axis = 0)

    input_image_shape = tf.shape(image_batch_tensor)
    image_height_width = input_image_shape[1:3]