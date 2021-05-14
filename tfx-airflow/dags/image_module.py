from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import absl
import tensorflow as tf
# import tensorflow_transform as tft # Step 4

from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor

from tfx_bsl.tfxio import dataset_options

import os
import tensorflow as tf

###############################
##Feature engineering functions

def make_input_fn(dir_uri, mode, vnum_epochs = None, batch_size = 512):
    def decode_tfr(serialized_example):
      # 1. define a parser
      features = tf.io.parse_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            "image": tf.io.FixedLenFeature([], tf.string),
            "path": tf.io.FixedLenFeature([], tf.string),
            "area": tf.io.FixedLenFeature([], tf.float32),
            "bbox": tf.io.VarLenFeature(tf.float32),
            "category_id": tf.io.FixedLenFeature([], tf.int64),
            "id": tf.io.FixedLenFeature([], tf.int64),
            "image_id": tf.io.FixedLenFeature([], tf.int64),
            "caption_1": tf.io.FixedLenFeature([],dtype=tf.string),
            "caption_2": tf.io.FixedLenFeature([],dtype=tf.string),
        })

      return features, features['caption_1']

    def _input_fn(v_test=False):
      # Get the list of files in this directory (all compressed TFRecord files)
      tfrecord_filenames = tf.io.gfile.glob(dir_uri)

      # Create a `TFRecordDataset` to read these files
      dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
      for tfrecord in dataset.take(4):
          serialized_example = tfrecord.numpy()
          example = tf.train.Example()
          example.ParseFromString(serialized_example)
         
      if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = vnum_epochs # indefinitely
      else:
        num_epochs = 1 # end-of-input after this

      dataset = dataset.batch(batch_size).prefetch(buffer_size = batch_size)

      #Convert TFRecord data to dict
      dataset = dataset.map(decode_tfr)

      if mode == tf.estimator.ModeKeys.TRAIN:
          num_epochs = vnum_epochs # indefinitely
          dataset = dataset.shuffle(buffer_size = batch_size)
      else:
          num_epochs = 1 # end-of-input after this

      dataset = dataset.repeat(num_epochs)       
      
      #Begins - Uncomment for testing only -----------------------------------------------------<
      if v_test == True:
        print(next(dataset.__iter__()))
        
      #End - Uncomment for testing only -----------------------------------------------------<
      return dataset
    return _input_fn