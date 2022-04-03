from flask import Flask, render_template, request, redirect, url_for
from io import BytesIO
import numpy as np
import cv2
import inference
app = Flask(__name__)

inference_model = inference.load_model_onnx(path='models/fashion_mnist_model_1e-3LR.onnx')


@app.route('/upload', methods = ['POST'])
def upload_file():
    filestr = request.files['file'].read()
    prediction = inference.predict_onnx(inference_model, pic_data=filestr)
    return render_template('img.html', pic='', label=prediction, prob='prob1')


@app.route('/upload', methods = ['GET'])
def upload_file_get():
    return render_template('upload.html')    


@app.route('/')
def hello():
    return redirect(url_for('upload_file'))


if __name__ == "__main__":
     app.run(host='127.0.0.1', port=5678)

