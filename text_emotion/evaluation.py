"""
This module contains functions for evaluating the model.

Functions:
    - evaluate_dataset: evaluates the model on the dataset.
"""

import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def encode_labels(y):
    labels = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
    return [labels[i] for i in y]


def split(x, y):
    """
    Split the dataset into train and test sets.

    Args:
        x: features.
        y: labels.

    Returns:
        x_train: train features.
        x_test: test features.
        y_train: train labels.
        y_test: test labels.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2,
        stratify=y,
        random_state=0
    )

    print("Train X shape:", x_train.shape)
    print("Test X shape:", x_test.shape)

    print("Train Y shape:", y_train.shape)
    print("Test Y shape:", y_test.shape)

    return x_train, x_test, y_train, y_test


def get_prediction(model, x):
    # happy = [
    #     'admiration', 'optimism', 'pride', 'realization',
    #     'relief', 'amusement', 'approval', 'caring',
    #     'desire', 'gratitude', 'joy', 'excitement',
    #     'curiosity', 'love', 'surprise'
    # ]
    # angry = ['anger', 'annoyance']
    # sad = [
    #     'confusion', 'remorse', 'sadness', 'disappointment',
    #     'nervousness', 'disapproval', 'disgust', 'embarrassment',
    #     'grief', 'fear',
    # ]
    happy = ['joy', 'surprise']
    angry = ['anger']
    sad = ['fear', 'sadness', 'disgust']

    pred = model(x)[0]['label']

    if pred in happy:
        return 'happy'

    elif pred in sad:
        return 'sad'

    elif pred in angry:
        return 'angry'
    
    else:
        return 'neutral'



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
    y = encode_labels(y)

    preds = []
    for test in tqdm(x, desc="Evaluating full dataset"):
        preds.append(get_prediction(model, test))

    preds = encode_labels(preds)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')

    del x
    gc.collect()

    return acc, f1, preds, y
