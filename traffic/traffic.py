import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

EPOCHS = 30
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.2


def main():

    # Check command-line arguments
    # if len(sys.argv) not in [2, 3]:
    #     sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(r"C:\Users\limji\Documents\Comp sci\Python\CS50_AI_course\traffic\gtsrb")

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    early_stopping_callback = EarlyStopping(
    monitor='val_loss',      # Metric to monitor
    patience=5,              # Number of epochs with no improvement after which training will be stopped
    verbose=1,               # Print message when stopping
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored metric)
    )
    
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=[early_stopping_callback])

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    # if len(sys.argv) == 3:
    filename = r"C:\Users\limji\Documents\Comp sci\Python\CS50_AI_course\traffic\model.keras"
    model.save(filename)
    print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    for foldername in os.listdir(data_dir):
        print(f"\nProcessing images in {foldername}")
        folderpath = os.path.join(data_dir, foldername)
        if not os.path.isdir(folderpath):
            continue
        count = 0
        for img in os.listdir(folderpath):
            imgpath = os.path.join(folderpath, img)
            image = cv2.imread(imgpath)
            if image is None:
                continue
            rimage = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            rimage = rimage.astype('float32') / 255.0
            images.append(rimage)
            labels.append(int(foldername))
            count += 1
            # Update line in terminal
            sys.stdout.write(f"\rProcessing folder {foldername}: {count} images processed")
            sys.stdout.flush()
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
    # Input layer - specify input shape including color channels (3 for RGB)
    tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)), # Explicit Input layer is good practice

    # Optional: Add data augmentation here for better generalization
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomRotation(0.1), # Rotate slightly
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1), # Shift slightly
    tf.keras.layers.RandomZoom(0.1), # Keep your zoom
    tf.keras.layers.RandomContrast(0.1), # Adjust contrast

    # Convolutional Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Convolutional Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Convolutional Block 3 (Optional, can add more complexity)
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Flatten and Dense Layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5), # Regularization

    # Output Layer for Binary Classification
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
    )
    return model



if __name__ == "__main__":
    main()
