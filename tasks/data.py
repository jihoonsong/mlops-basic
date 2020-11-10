import os
import tensorflow as tf


_train_file = 'train.csv'
_test_file = 'test.csv'


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


if __name__ == "__main__":
    save_data_as_csv(*generate_data())
