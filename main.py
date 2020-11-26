import numpy
import requests
import tensorflow as tf
from io import BytesIO
from werkzeug.exceptions import BadRequest
from flask import jsonify
from PIL import Image


_weights_path = 'fashion_mnist_classifier/weights/'
_weights_tag = '1606328955'

# Use model as a global variable to keep it warm.
model = None


def predict_fashion_mnist(request):
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    global model
    if not model:
        model = tf.saved_model.load(_weights_path + _weights_tag).signatures['predict']

    if request.get_json():
        request_json = request.get_json()
        url = request_json.get('url')
        if not url:
            raise BadRequest('Invalid URL')
    else:
        raise BadRequest('Invalid argument')

    image_raw = requests.get(url)
    image = (numpy.array(Image.open(BytesIO(image_raw.content))) / 255.0)[numpy.newaxis, :, :, numpy.newaxis]

    prediction = model(tf.constant(image, dtype=tf.float32))
    class_id = class_names[prediction['class_ids'][0][0]]

    response = {
        'status': 200,
        'data': {
            'class': class_id,
        },
    }

    return jsonify(response)
