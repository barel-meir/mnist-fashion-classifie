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
import base64
import wandb
from wandb.keras import WandbCallback
# Image Libraries
from PIL import Image, ImageFilter, ImageStat

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


def build_model(model_type=1):
    '''
    first, lets import the data
    '''
    global X_train
    global y_train
    global X_test
    global y_test
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    print("Train shapes:", X_train.shape)
    print("Test shapes:", X_test.shape)

    # Reshape the images.
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    print("Train shapes:", X_train.shape)
    print("Test shapes:", X_test.shape)

    # Initilize a new wandb run
    wandb.init(entity="wandb", project="keras-intro", name=f"model{model_type}")

    # Default values for hyper-parameters
    config = wandb.config  # Config is a variable that holds and saves hyper parameters and inputs
    config.learning_rate = 0.001
    config.epochs = 10
    config.img_width = 28
    config.img_height = 28
    config.num_classes = 10
    config.batch_size = 128
    config.validation_size = 10000
    config.weight_decay = 0.0005
    config.activation = 'relu'
    config.optimizer = 'adam'
    config.seed = 42


    if model_type == 1:
        '''
        build the first model: model1.onnx
        '''
        model = Sequential(name='model1')
        model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
    else:
        '''
        build second model: fashion_mnist_model_1e-3LR.onnx
        '''
        model = Sequential([
            Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu', padding='same',
                   kernel_initializer='he_uniform'),
            MaxPooling2D(pool_size=2),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')],
            name='model2')

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, to_categorical(y_train),
              epochs=config.epochs,
              batch_size=config.batch_size,
              validation_data=(X_test, to_categorical(y_test)),
              verbose=1,
              callbacks=[
                  WandbCallback(data_type="image", validation_data=(X_test, to_categorical(y_test)), labels=labels),
                  tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
              )

    wandb.finish()
    return model, model.evaluate(x=X_test, y=to_categorical(y_test))


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
    return onnx_model


def predict_onnx(onnx_model, pic_data):
    npimg = np.fromstring(pic_data, np.uint8)
    pic = cv2.imdecode(npimg, -1)
    # if len(pic.shape) >= 3:
    #     pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.resize(pic, (28, 28), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('pic', pic)
    
    displayPic = cv2.resize(pic, (128, 128), interpolation=cv2.INTER_CUBIC)
    ret, png = cv2.imencode('.png',displayPic)
    png_as_text = base64.b64encode(png).decode('utf-8')
        
    pic = pic.reshape((1, 28, 28, 1)).astype(np.float32)
    print("pic shapes:", pic.shape)
    pred = np.squeeze(onnx_model.run(None, {"x": pic}))
    index = np.argmax(pred)
    prediction = labels[index]
    proba = pred[index]
    print(f"prediction: {prediction}")

    return (png_as_text,prediction,proba)


def save_images_for_testing():
    for i in range(9):
        cv2.imwrite(f'test_images/p{i}-{labels[y_test[i]]}.png', X_test[i])


def main():
    model_name = "fashion_mnist_model_1e-3LR"
    model1, acc1 = build_model(1)
    model2, acc2 = build_model(2)
    if acc1[1] > acc2[1]:
        # save the higher accuracy model
        save_model_onnx(model1, model_name)
    else:
        save_model_onnx(model2, model_name)

    if False:
        # this is for testing!
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