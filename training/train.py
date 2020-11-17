import argparse


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
