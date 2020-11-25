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
import tensorflow as tf
from fashion_mnist_classifier.models.dnn import DNN
from google.cloud import bigquery


_dataset_path = 'fashion_mnist_classifier/datasets/'
_x_train_file = _dataset_path + 'x_train.npy'
_y_train_file = _dataset_path + 'y_train.npy'
_x_test_file = _dataset_path + 'x_test.npy'
_y_test_file = _dataset_path + 'y_test.npy'


def load_data_from_bigquery(project):
    def raw_to_feature(image_raw):
        return np.asarray(image_raw.split(','), 'float')

    dataset = 'fashion_mnist'
    train = 'train'
    test = 'test'

    # Get train and test tables.
    client = bigquery.Client(project=project)
    train_table = client.get_table(f'{project}.{dataset}.{train}')
    test_table = client.get_table(f'{project}.{dataset}.{test}')

    # # Load train data from BigQuery to x_train.npy and y_train.npy.
    if (not os.path.exists(_x_train_file) or
        not os.path.exists(_y_train_file)):
        # Load train data from BigQuery.
        rows_iter = client.list_rows(train_table)
        arrows = rows_iter.to_arrow()
        arrows_dict = arrows.to_pydict()

        # Revert x_train and y_train to ndarray.
        x_train, y_train = arrows_dict['image_raw'], arrows_dict['label']
        x_train = np.array([raw_to_feature(image_raw) for image_raw in x_train], 'float')
        y_train = np.array(y_train)

        # Save x_train as x_train.npy and y_train as y_train.npy.
        np.save(_x_train_file, x_train)
        np.save(_y_train_file, y_train)

        print(f'Downloaded {x_train.shape[0]} rows into {_x_train_file}')
        print(f'Downloaded {y_train.shape[0]} rows into {_y_train_file}')

    # Load test data from BigQuery to x_test.npy and y_test.npy.
    if (not os.path.exists(_x_test_file) or
        not os.path.exists(_y_test_file)):
        # Load test data from BigQuery.
        rows_iter = client.list_rows(test_table)
        arrows = rows_iter.to_arrow()
        arrows_dict = arrows.to_pydict()

        # Revert x_test and y_test to ndarray.
        x_test, y_test = arrows_dict['image_raw'], arrows_dict['label']
        x_test = np.array([raw_to_feature(image_raw) for image_raw in x_test], 'float')
        y_test = np.array(y_test)

        # Save x_test as x_test.npy and y_test as y_test.npy.
        np.save(_x_test_file, x_test)
        np.save(_y_test_file, y_test)

        print(f'Downloaded {x_test.shape[0]} rows into {_x_test_file}')
        print(f'Downloaded {y_test.shape[0]} rows into {_y_test_file}')


def generator(images, labels):
    def _generator():
        for image, label in zip(images, labels):
            yield image, label

    return _generator


def preprocess_data(image, label):
    image = image / 255.0
    image = tf.reshape(image, [28, 28, 1])

    # The image data are mapped to 'image' feature column.
    return {'image': image}, label


def train_input_fn():
    x_train = np.load(_x_train_file)
    y_train = np.load(_y_train_file)

    train_dataset = tf.data.Dataset.from_generator(
        generator(x_train, y_train),
        (tf.float32, tf.int32),
        (tf.TensorShape([28 * 28 * 1]), tf.TensorShape([]))
    )
    train_dataset = train_dataset.map(preprocess_data).batch(1024)

    return train_dataset


def predict_input_fn():
    x_test = np.load(_x_test_file)
    y_test = np.load(_y_test_file)

    test_dataset = tf.data.Dataset.from_generator(
        generator(x_test, y_test),
        (tf.float32, tf.int32),
        (tf.TensorShape([28 * 28 * 1]), tf.TensorShape([]))
    )
    test_dataset = test_dataset.map(preprocess_data).batch(1024)

    return test_dataset


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

    # Prepare data.
    load_data_from_bigquery(args.project)

    # Prepare model.
    dnn = DNN()
    estimator = dnn.get_estimator()

    # Train and evaluate model.
    estimator.train(input_fn=train_input_fn)
    evaluation = estimator.evaluate(input_fn=predict_input_fn)
    print(evaluation)
