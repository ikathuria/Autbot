"""
This module contains functions for evaluating the model.

Functions:
    - evaluate_dataset: evaluates the model on the dataset.
    - evaluate_custom: evaluates the model on custom audio files.
"""

import numpy as np
import pandas as pd
import gc
from speech_emotion.features import extract_melspectrogram, split


def evaluate_dataset(model, x, y):
    """
    Evaluates the model on the dataset.

    Args:
        model: the model to evaluate.
        x: the features.
        y: the labels.

    Returns:
        a1: the test accuracy.
        l1: the test loss.
        a2: the train accuracy.
        l2: the train loss.
        a3: the full dataset accuracy.
        l3: the full dataset loss.
    """
    x_train, x_test, y_train, y_test = split(x, y)

    l1, a1 = model.evaluate(x_test, y_test, verbose=0)
    del x_test, y_test
    gc.collect()

    l2, a2 = model.evaluate(x_train, y_train, verbose=0)
    del x_train, y_train
    gc.collect()

    l3, a3 = model.evaluate(x, y, verbose=0)
    del x, y
    gc.collect()

    return a1, l1, a2, l2, a3, l3


def evaluate_custom(model):
    """
    Evaluates the model on custom audio files.

    Args:
        model: the model to evaluate.

    Returns:
        preds: the predictions.
        true: the true labels.
    """
    ex = np.array([
        extract_melspectrogram("examples/sad.wav"),
        extract_melspectrogram("examples/angry1.wav"),
    ])
    true = "sad, angry"

    labels = pd.read_json(
        "data/labels.json", orient="index"
    ).to_dict()[0]

    preds = model.predict(ex)
    del ex
    gc.collect()

    preds = ", ".join([
        labels[np.argmax(i)] for i in preds
    ])

    return preds, true
