import tensorflow as tf
from tensorflow import keras

# # Helper libraries
import numpy as np
import math
import pathlib as path
from PIL import Image
import random

print(tf.__version__)

# Helper function to display digit images

MODEL_FILE = "mnist.h5"


def pre_process():
    def _process(p):
        return np.asfarray(Image.open(
            p).resize((28, 28), Image.BICUBIC).convert('L').point(lambda x: 255 - x)) / 255

    root_dir = path.Path("English")
    train_dir = [i for i in root_dir.iterdir()
                 if i.name.startswith("Sample")]
    test_dir = [i for i in root_dir.iterdir()
                if i.name.startswith("Test")]

    train_dir.sort()
    test_dir.sort()

    test_data = [[_process(str(j)) for j in i.iterdir()] for i in test_dir]
    train_data = [[_process(str(j)) for j in i.iterdir()] for i in train_dir]

    return train_data, test_data


def build():
    train_data, test_data = pre_process()

    train_labels = [i for i in range(len(train_data))
                    for j in train_data[i]]
    test_labels = [i for i in range(len(test_data))
                   for j in test_data[i]]

    train_data = [j for i in train_data for j in i]
    test_data = [j for i in test_data for j in i]

    test_dl = list(zip(test_data, test_labels))
    train_dl = list(zip(train_data, train_labels))

    random.shuffle(test_dl)
    random.shuffle(train_dl)

    test_data, test_labels = zip(*test_dl)
    train_data, train_labels = zip(*train_dl)

    test_data = np.asarray(test_data)
    test_labels = np.asarray(test_labels)
    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)

    # Define the model architecture

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        # keras.layers.Dense(128, activation=tf.nn.relu),

        # Optional: You can replace the dense layer above with the convolution layers below to get higher accuracy.
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                            activation=tf.nn.relu),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                            activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the digit classification model
    model.fit(train_data, train_labels, epochs=5)

    # Evaluate the model using test dataset.
    test_loss, test_acc = model.evaluate(test_data, test_labels)

    print('Test accuracy:', test_acc)

    # Predict the labels of digit images in our test dataset.
    # predictions = model.predict(test_data)

    # Then plot the first 25 test images and their predicted labels.
    # show_sample(test_images,
    #             ['Predicted: %d' % np.argmax(result) for result in predictions])

    model.save(MODEL_FILE)


def recognize(images) -> int:
    if not path.Path(MODEL_FILE).exists():
        build()
    model = keras.models.load_model(MODEL_FILE)
    predictions = model.predict(images)
    return [np.argmax(result) for result in predictions]


class Recognizer:
    def __init__(self):
        if not path.Path(MODEL_FILE).exists():
            build()
        self.model = keras.models.load_model(MODEL_FILE)

    def predict(self, images):
        predictions = self.model.predict(images)
        return [np.argmax(result) for result in predictions]


if __name__ == "__main__":
    build()
