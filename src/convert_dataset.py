"""This file will convert food-101 into TFRecord files which can be easily ingested by Tensorflow
for batch processing.
This will make the pipeline for multi-process model training much more efficient"""

import random
import os
import math
import tensorflow as tf
from dataset_utils import _dataset_exists, _get_filenames_and_classes, \
     write_label_file, _convert_dataset

"""==| Arguments Definition |=="""
flags = tf.app.flags

# Dataset directory
flags.DEFINE_string('dataset_dir', None, "String: Your dataset directory")
# Number of images in validation set. Knowing the total number of examples in advance is necessary
flags.DEFINE_float("validation_set", .3, "Float: the proportion of examples in the dataset to be used for validation")
# Number of shards per dataset split
flags.DEFINE_integer("num_shards", 2, "Int: Number of shards to split the TFRecord files")
# Seed fir repeatability
flags.DEFINE_integer("random_seed", 0, "Int: Random seed to use for repeatability")
# Output filename for the naming the TFRecord file
flags.DEFINE_string("tfrecord_filename", None, "String: The output filename to name your TFRecord file")
FLAGS = flags.FLAGS

def main():
     """==| Checks |=="""
     # Check if there is a tfrecord_filename entered
     if not FLAGS.tfrecord_filename:
          raise ValueError("tfrecord_filename is empty. Please state a tfrecord_filename argument.")
     # Check if there is a dataset directory entered
     if not FLAGS.dataset_dir:
          raise ValueError('dataset_dir is empty. Please state a dataset_dir argument')
     # If the TFRecord files already exist in the directory, then exit without creating the files again
     if _dataset_exists(dataset_dir = FLAGS.dataset_dir,
                        _NUM_SHARDS = FLAGS.num_shards,
                        output_filename = FLAGS.tfrecord_filename):
          print("Dataset files already exist. Exiting without re-creating them")
          return None
     """==| END OF CHECKS |=="""

     # Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
     photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_dir,
                                                               output_filename=FLAGS.tfrecord_filename)

     # Some datasets have a "categories" file with actual names those photo_names correspond to
     # such as UECFOOD256 dataset. Let's map it out.
     # if os.path.exists(os.path.join(FLAGS.dataset_dir, 'category.txt')):
     #     with open(os.path.join(FLAGS.dataset_dir, 'category.txt')) as cat_file:
     #         replacement_dict = [cat_name.split('\t') for cat_name in cat_file]
     #         class_names = [replacement_dict[int(class_name)][1].replace('\n','') for class_name in class_names]
     #         import pdb; pdb.set_trace()

     # Refer each of the class name to a specific integer number for predictions later
     class_names_to_ids = dict(zip(class_names, range(len(class_names))))

     # Find the number of validation examples we need
     num_validation = int(FLAGS.validation_size * len(photo_filenames))

     # Divide the training datasets into train and test:
     random.seed(FLAGS.random_seed)
     random.shuffle(photo_filenames)
     training_filenames = photo_filenames[num_validation:]
     validation_filenames = photo_filenames[:num_validation]

     # First, convert the training and validation sets.
     _convert_dataset('train', training_filenames, class_names_to_ids,
                      dataset_dir=FLAGS.dataset_dir,
                      tfrecord_filename=FLAGS.tfrecord_filename,
                      _NUM_SHARDS=FLAGS.num_shards)
     _convert_dataset('validation', validation_filenames, class_names_to_ids,
                      dataset_dir=FLAGS.dataset_dir,
                      tfrecord_filename=FLAGS.tfrecord_filename,
                      _NUM_SHARDS=FLAGS.num_shards)

     # Finally, write the labels file:
     labels_to_class_names = dict(zip(range(len(class_names)), class_names))
     write_label_file(labels_to_class_names, FLAGS.dataset_dir)

     print('\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename))


if __name__ == "__main__":
     main()

