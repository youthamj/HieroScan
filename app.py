import base64

from flask import Flask, request, jsonify
from detect import detect
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from collections import Counter, defaultdict
from classify import image_to_gardiner
from preprocess import preprocess
from segmentation import forwardwordsegmentation
from translation import translate
from skimage import io

app = Flask(__name__)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/hieroscan', methods=["POST"])
def translate_image():
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        image = data['image']
        with open('input.jpg', 'wb') as fh:
            fh.write(base64.decodebytes(bytes(image, 'utf-8')))
        input_image = io.imread('input.jpg')
        preprocessed = preprocess(input_image)
        detections = detect(preprocessed)
        gardiners, classified_image = image_to_gardiner(preprocessed, detections)
        io.imsave('output.jpg', classified_image)
        transliteration = forwardwordsegmentation(gardiners)
        translated = translate(transliteration)
        output_file = open("output.jpg", "rb")
        encoded_output = base64.encodebytes(output_file.read()).decode('utf-8')
        res = {'translation': translated, 'transliteration': transliteration,
               'image': encoded_output}
        return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
