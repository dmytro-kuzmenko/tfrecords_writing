import os
import sys
import random

import cv2
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from settings import app


def _load_image(path):
    raw_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(raw_bytes, channels=3)
    image = tf.image.resize(image, (224,224), method='nearest')
    image = tf.io.encode_jpeg(image, optimize_size=True)
    return image

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _build_examples_list(input_folder, seed):
    examples = []
    images = tf.io.gfile.glob(input_folder + "/*.jpg")
    for filepath in images:
        example = {
            'classname': 'unlabeled', 
            'path': filepath
        }
        examples.append(example)

    random.seed(seed)
    random.shuffle(examples)
    examples = np.random.choice(examples, 100)
    return examples

def _split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

def _get_examples_share(examples, training_split):
    examples_size = len(examples)
    len_training_examples = int(examples_size * training_split)

    return np.split(examples, [len_training_examples])

def _write_tfrecord(examples, output_filename):
    writer = tf.io.TFRecordWriter(output_filename)
    for example in tqdm(examples):
        try:
            image = _load_image(example['path'])
            if image is not None:
                feature = {
                    'label': _bytes_feature(tf.compat.as_bytes(example['classname'])),
                    'image': _bytes_feature(image)
                }
                tf_example = tf.train.Example(features = tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())
        except Exception as inst:
            print(inst)
            pass
    writer.close()

def _write_sharded_tfrecord(examples, number_of_shards, base_output_filename, is_training = True):
    sharded_examples = _split_list(examples, number_of_shards)
    for count, shard in tqdm(enumerate(sharded_examples, start = 1)):
        output_filename = '{0}_{1}_{2:02d}of{3:02d}.tfrecord'.format(
            base_output_filename,
            'training' if is_training else 'test',
            count,
            number_of_shards 
        )
        _write_tfrecord(shard, output_filename)


examples = _build_examples_list(app['IMAGES_INPUT_FOLDER'], app['SEED'])
training_examples, test_examples = _get_examples_share(examples, app['TRAINING_EXAMPLES_SPLIT']) # pylint: disable=unbalanced-tuple-unpacking

print("Creating training shards", flush = True)
_write_sharded_tfrecord(training_examples, app['NUMBER_OF_SHARDS'], app['OUTPUT_FILENAME'])
print("\nCreating test shards", flush = True)
_write_sharded_tfrecord(test_examples, app['NUMBER_OF_SHARDS'], app['OUTPUT_FILENAME'], False)
print("\n", flush = True)