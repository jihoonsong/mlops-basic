# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import numpy as np
import os
import sys
import tensorflow as tf
from google.cloud import bigquery


_train_file = 'train.tfrecord'
_test_file = 'test.tfrecord'


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _raw_to_feature(image_raw):
    # Fashion-MNIST is a dataset of 28x28 grayscale images.
    return np.asarray(image_raw.split(','), 'float').reshape([28, 28])


def load_data_from_bigquery(project):
    dataset = 'fashion_mnist'
    train = 'train'
    test = 'test'
    dataset_path = (os.path.dirname(os.path.realpath(sys.argv[0])) +
                    '/../fashion_mnist_classifier/datasets/')
    train_path = dataset_path + _train_file
    test_path = dataset_path + _test_file

    # Get train and test tables.
    client = bigquery.Client(project=project)
    train_table = client.get_table(f'{project}.{dataset}.{train}')
    test_table = client.get_table(f'{project}.{dataset}.{test}')

    # # Load train data from BigQuery to train.tfrecord if train.tfrecord does not exist.
    if not os.path.exists(train_path):
        # Load train data from BigQuery.
        rows_iter = client.list_rows(train_table)
        arrows = rows_iter.to_arrow()
        arrows_dict = arrows.to_pydict()

        # Revert x_train and y_train to ndarray.
        x_train, y_train = arrows_dict['image_raw'], arrows_dict['label']
        x_train = np.array([_raw_to_feature(image_raw) for image_raw in x_train], 'float')
        y_train = np.array(y_train)

        # Write to train.tfrecord.
        with tf.io.TFRecordWriter(train_path) as writer:
            output_rows = 0
            for output_rows, (image, label) in enumerate(zip(x_train, y_train)):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image_raw': _bytes_feature(image.tostring()),
                            'label': _int64_feature(label)
                        }))
                writer.write(example.SerializeToString())

            print(f'Downloaded {output_rows + 1} rows into {_train_file}')

    # Load test data from BigQuery to test.tfrecord if test.tfrecord does not exist.
    if not os.path.exists(test_path):
        # Load test data from BigQuery.
        rows_iter = client.list_rows(test_table)
        arrows = rows_iter.to_arrow()
        arrows_dict = arrows.to_pydict()

        # Revert x_test and y_test to ndarray.
        x_test, y_test = arrows_dict['image_raw'], arrows_dict['label']
        x_test = np.array([_raw_to_feature(image_raw) for image_raw in x_test], 'float')
        y_test = np.array(y_test)

        # Write to test.tfrecord.
        with tf.io.TFRecordWriter(test_path) as writer:
            output_rows = 0
            for output_rows, (image, label) in enumerate(zip(x_test, y_test)):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image_raw': _bytes_feature(image.tostring()),
                            'label': _int64_feature(label)
                        }))
                writer.write(example.SerializeToString())

            print(f'Downloaded {output_rows + 1} rows into {_test_file}')


if __name__ == "__main__":
    description = """
    This script generates fashion_mnist data and upload it as a
    BigQuery dataset under [PROJECT ID].fashion_mnist.{train,test}.

    To run this script successfully, you need to complete the following steps:
    1. Select or create a Cloud Platform project.
    2. Enable billing for your project.
    3. Enable the Google Cloud BigQuery API.
    4. Setup Authentication.

    Please see (https://googleapis.dev/python/bigquery/latest/index.html) for more information."""

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', type=str, default='mlops-basic', help='Your GCP project id')

    args = parser.parse_args()

    load_data_from_bigquery(args.project)
