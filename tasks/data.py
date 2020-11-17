import argparse
import os
import tensorflow as tf
from google.cloud import bigquery


_train_file = 'train.csv'
_test_file = 'test.csv'


def clear_data():
    if os.path.exists(_train_file):
        os.remove(_train_file)
        print(f'Removed {_train_file}')

    if os.path.exists(_test_file):
        os.remove(_test_file)
        print(f'Removed {_test_file}')


def generate_data():
    # We use Fashion-MNIST here.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize to [0.0, 1.0]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channel dimension.
    x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]

    return x_train, y_train, x_test, y_test


def save_data_as_csv(x_train, y_train, x_test, y_test):
    # Save train data as train.csv if train.csv does not exist.
    if not os.path.exists(_train_file):
        with open(_train_file, 'w') as output:
            output_rows = 0
            for output_rows, (image, label) in enumerate(zip(x_train, y_train)):
                image_raw = ','.join(map(str, image.flatten().tolist()))
                output.write(f'{output_rows}|{image_raw}|{label}')
                output.write('\n')

            print(f'Generated {output_rows + 1} rows into {_train_file}')

    # Save test data as test.csv if test.csv does not exist.
    if not os.path.exists(_test_file):
        with open(_test_file, 'w') as output:
            output_rows = 0
            for output_rows, (image, label) in enumerate(zip(x_test, y_test)):
                image_raw = ','.join(map(str, image.flatten().tolist()))
                output.write(f'{output_rows}|{image_raw}|{label}')
                output.write('\n')

            print(f'Generated {output_rows + 1} rows into {_test_file}')


def load_data_to_bigquery(project):
    dataset = 'fashion_mnist'
    train = 'train'
    test = 'test'

    # Create dataset, train and test tables.
    client = bigquery.Client(project=project)
    client.create_dataset(dataset)
    train_table = client.create_table(f'{project}.{dataset}.{train}')
    test_table = client.create_table(f'{project}.{dataset}.{test}')

    # Set job configuration.
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.field_delimiter = "|"
    job_config.schema = [
        bigquery.schema.SchemaField("image_raw", "STRING"),
        bigquery.schema.SchemaField("label", "INTEGER"),
    ]
    job_config.autodetect = True

    # Load train data to BigQuery.
    with open(_train_file, "rb") as input:
        train_load = client.load_table_from_file(
            input, train_table, job_config=job_config
        )

    # Load test data to BigQuery.
    with open(_test_file, "rb") as input:
        test_load = client.load_table_from_file(
            input, test_table, job_config=job_config
        )

    # The loading is an async operation. Wait for it to finish.
    train_load.result()
    test_load.result()

    print(f'Loaded {train_load.output_rows} rows from {_train_file} into {project}.{dataset}.{train}')
    print(f'Loaded {test_load.output_rows} rows from {_test_file} into {project}.{dataset}.{test}')


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

    save_data_as_csv(*generate_data())
    load_data_to_bigquery(args.project)
    clear_data()
