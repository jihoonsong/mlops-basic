# MLOps Basic

An example of MLOps that loads data to/from BigQuery, trains and serves model via Cloud Functions.

## Prerequisite

This project uses BigQuery and Cloud Functions, so you need to setup for those environments.

### For BigQuery
1. [Select or create a Cloud Platform project.](https://console.cloud.google.com/project)
2. [Enable billing for your project.](https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project)
3. [Enable the Google Cloud BigQuery API.](https://cloud.google.com/bigquery)
4. [Setup Authentication.](https://cloud.google.com/docs/authentication/getting-started)

### For Cloud Functions
1. [Setup Cloud SDK.](https://cloud.google.com/sdk/docs/quickstart)

## Steps to Follow

1. Set the environment variable GOOGLE_APPLICATION_CREDENTIALS and PYTHONPATH as follows. Please note that you need to replace [PATH] with the file path of the JSON file that contains your service account key.

```shell
export GOOGLE_APPLICATION_CREDENTIALS="[PATH]"
export PYTHONPATH='.'
```

2. Generate data and load it to BigQuery. In this project, the data is fasion-MNIST. Though I didn't do any preprocessing here, whether preprocess data before load data to BigQuery or after load data from BigQuery is up to you.

```shell
python tasks/data.py
```

3. Load data from BigQuery, train and evaluate model.

```shell
python fashion_mnist_classifier/training/train_and_evaluate.py
```

4. (Optional) Check saved MetaGraphDefs and SignatureDefs as follows. You can see there is a signature_def['predict'], which will be used later for serving.

```shell
saved_model_cli show --dir fashion_mnist_classifier/weights/1606328955/ --all
```

5. Deploy Cloud Functions. If you encoutner the memory limit exceeded error, please see [this.](https://stackoverflow.com/questions/43313251/cloud-functions-for-firebase-killed-due-to-memory-limit-exceeded) In my case, increasing memory to 1G was enough.

```shell
gcloud beta functions deploy predict_fashion_mnist --runtime python37 --trigger-http --project mlops-basic --region asia-northeast1
```

6. Test API. Please note that you need to replace [PROJECT] with the name of your Cloud Platform project.  
e.g. https://asia-northeast1-mlops-basic.cloudfunctions.net/predict_fashion_mnist


```shell
curl -X POST https://asia-northeast1-[PROJECT].cloudfunctions.net/predict_fashion_mnist -H 'Content-Type: application/json' -d '{"url":"https://raw.githubusercontent.com/ryfeus/gcf-packs/master/tensorflow2.0/example/test.png"}'
```

&emsp; Or, equivalently, you can do this.

```shell
./fashion_mnist_classifier/tests/test.sh
```

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

