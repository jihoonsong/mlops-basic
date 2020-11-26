curl -X POST https://asia-northeast1-mlops-basic.cloudfunctions.net/predict_fashion_mnist \
-H 'Content-Type: application/json' \
-d '{"url":"https://raw.githubusercontent.com/ryfeus/gcf-packs/master/tensorflow2.0/example/test.png"}'