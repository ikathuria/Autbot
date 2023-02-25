"""
This module contains functions for creating and optimizing the model.
"""

import numpy as np

from sklearn.model_selection import train_test_split

from keras import Input, Model
from keras.models import Sequential
from keras.layers import Reshape, Permute, MultiHeadAttention, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from hyperas.distributions import uniform, choice
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim


# MODELING #################################################################################################
def create_model(input_shape, num_classes, v=False):
    """
    Create the CNN + Transformer model.

    Args:
        input_shape: shape of the input data.
        num_classes: number of classes in the dataset.
        v: verbose mode.

    Returns:
        model: CNN + Transformer model.
    """
    # Input layer
    inputs = Input(shape=input_shape)

    # CNN layers
    x = Conv2D(
        64, kernel_size=(3, 3), activation="relu", padding="valid"
    )(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(
        64, kernel_size=(3, 3), activation="relu", padding="valid"
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(
        32, kernel_size=(3, 3), activation="relu", padding="valid",
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(
        32, kernel_size=(3, 3), activation="relu", padding="valid",
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Reshape and Permute for Transformer
    x = Reshape((-1, 64))(x)
    x = Permute((2, 1))(x)

    # Transformer layers
    x = LayerNormalization()(x)
    x = MultiHeadAttention(
        num_heads=10, key_dim=64
    )(x, x)

    x = LayerNormalization()(x)
    # x = Add()([x, Reshape((64, -1))(x)])

    x = Flatten()(x)

    x = Dense(32, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    # Output layer
    outputs = Dense(num_classes, activation="softmax")(x)

    # Compile with categorical cross-entropy loss
    # and Adam optimizer
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0001),
        metrics=["accuracy"]
    )

    if v:
        model.summary()

    return model


def build_model(shape=(128, 300, 3), classes=4):
    """
    Build the CNN model architecture.

    Args:
        shape: shape of the input data.
        classes: number of classes in the dataset.

    Returns:
        model: Keras CNN model
    """

    model = Sequential([
        # conv1
        Conv2D(
            64, kernel_size=(3, 3), activation="relu",
            padding="valid", input_shape=shape
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),

        # conv2
        Conv2D(
            32, kernel_size=(3, 3), activation="relu",
            padding="valid",
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),
        Dropout(0.2),

        # conv3
        Conv2D(
            64, kernel_size=(3, 3), activation="relu",
            padding="valid",
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),
        Dropout(0.5),

        # flattened
        Flatten(),

        Dense(128, activation="relu"),
        Dense(64, activation="relu"),

        Dense(classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    return model


# OPTIMIZATION #################################################################################################
def data():
    x = np.load("data/melspectrogram/RAVDESS_features.npy")
    y = np.load("data/melspectrogram/RAVDESS_labels.npy")

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2,
        stratify=y,
        random_state=0
    )

    return X_train, Y_train, X_test, Y_test


def model(X_train, Y_train, X_test, Y_test):
    """
    Build the model architecture.

    Returns:
        model: Keras LSTM model

    """
    shape = X_train[0].shape

    model = Sequential([
        Conv2D(
            16, kernel_size=(3, 3), activation="relu",
            padding="same", input_shape=shape
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),

        Conv2D(
            32, kernel_size=(3, 3), activation="relu",
            padding="valid",
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),

        Conv2D(
            64, kernel_size=(3, 3), activation="relu",
            padding="valid",
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),
        Dropout({{uniform(0, 1)}}),

        Conv2D(
            64, kernel_size=(3, 3), activation="relu",
            padding="valid",
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),
        Dropout({{uniform(0, 1)}}),

        Flatten(),

        Dense(128, activation="relu"),
        Dense(64, activation="relu"),

        Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stopping_monitor = EarlyStopping(
        monitor="val_loss", patience=3
    )
    model.fit(
        X_train, Y_train,
        batch_size=16,
        epochs=50,
        verbose=0,
        callbacks=[early_stopping_monitor],
        validation_split=0.2
    )

    score, acc = model.evaluate(X_train, Y_train, verbose=0)
    print("\nTrain accuracy: %.4f\nTrain loss: %.4f" % (acc, score))

    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print("Test accuracy: %.4f\nTest loss: %.4f" % (acc, score))

    return {"loss": -acc, "status": STATUS_OK, "model": model}


if __name__ == "__main__":
    # pass
    model = create_model(
        (128, 300, 1), 4
    )
