import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from keras.datasets import fashion_mnist
import tf2onnx
import onnx
import cv2

X_train, y_train, X_test, y_test = None, None,None, None
labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'}


def build_model():
    '''
    first, lets import the data
    '''
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    print("Train shapes:", X_train.shape)
    print("Test shapes:", X_test.shape)

    '''
    now lets pre-process the data 
    first verify the images are B/W 
    then, reshape the image such that it will have a proper dimension with the B/W value  
    '''
    # Normalize the images.
    X_train = (X_train / 255) - 0.5
    X_test = (X_test / 255) - 0.5

    # Reshape the images.
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    print("Train shapes:", X_train.shape)
    print("Test shapes:", X_test.shape)

    '''
    build the first model: model1.onnx
    '''
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    save_model_onnx(model, "model1")
    '''
    build second model: fashion_mnist_model_1e-3LR.onnx
    '''
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu', padding='same', kernel_initializer='he_uniform'),
        MaxPooling2D(pool_size=2),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')])

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    save_model_onnx(model, "fashion_mnist_model_1e-3LR")

    return model


def fit_model(model):
    '''
    fit the model
    '''
    history = model.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32, validation_data=(X_test,  to_categorical(y_test)), verbose=1)


def save_model_onnx(model, name):
    '''
    save as onnx file
    '''
    input_signature = [tf.TensorSpec([None, 28, 28, 1], tf.int8, name=name)]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save(onnx_model, f"models/{name}.onnx")


def load_model_onnx(path=''):
    import onnxruntime as ort
    onnx_model = ort.InferenceSession(path)
    print(onnx_model.get_outputs()[1].name)
    return onnx_model


def predict_onnx(onnx_model, pic_data):
    npimg = np.fromstring(pic_data, np.uint8)
    pic = cv2.imdecode(npimg, -1)
    # if len(pic.shape) >= 3:
    #     pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.resize(pic, (28, 28), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('pic', pic)
    pic = pic.reshape((1, 28, 28, 1)).astype(np.float32)
    print("pic shapes:", pic.shape)
    prediction = labels[np.argmax(onnx_model.run(None, {"x": pic}))]
    probabilities = onnx_model.get_outputs()[1].name
    print(f"prediction: {prediction}, probabilities: {probabilities}")

    return prediction


def save_images_for_testing():
    for i in range(9):
        cv2.imwrite(f'test_images/p{i}-{labels[y_test[i]]}.png', X_test[i])


def main():
    onnx_model = load_model_onnx(path='models/fashion_mnist_model_1e-3LR.onnx')
    directory = 'test_images'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f"img: {f} ; prediction: {labels[predict_onnx(onnx_model, pic_data=f)]}")
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()