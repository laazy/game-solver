import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import math
import pathlib as path

print(tf.__version__)

# Helper function to display digit images

MODEL_FILE = "mnist.h5"


def build():
    # Download MNIST dataset.
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # If you can't download the MNIST dataset from Keras, please try again with an alternative method below
    # path = keras.utils.get_file('mnist.npz',
    #                             origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
    #                             file_hash='8a61469f7ea1b51cbae51d4f78837e45')
    # with np.load(path, allow_pickle=True) as f:
    #   train_images, train_labels = f['x_train'], f['y_train']
    #   test_images, test_labels = f['x_test'], f['y_test']

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Show the first 25 images in the training dataset.
    # show_sample(train_images,
    #             ['Label: %s' % label for label in train_labels])

    # Define the model architecture
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),

        # Optional: You can replace the dense layer above with the convolution layers below to get higher accuracy.
        # keras.layers.Reshape(target_shape=(28, 28, 1)),
        # keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
        # keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
        # keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # keras.layers.Dropout(0.25),
        # keras.layers.Flatten(input_shape=(28, 28)),
        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dropout(0.5),

        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the digit classification model
    model.fit(train_images, train_labels, epochs=5)

    # Evaluate the model using test dataset.
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

    # Predict the labels of digit images in our test dataset.
    predictions = model.predict(test_images)

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


if __name__ == "__main__":
    build()
