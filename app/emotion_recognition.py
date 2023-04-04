"""
This module contains functions for the emotion prediction.

Functions:
    - speech_emotion: Predict the emotion based on speech.
    - text_emotion: Predict the emotion based on text.
    - predict_emotion: Predict the emotion based on speech and text.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'Autbot')))
sys.path.append(os.path.abspath(os.path.join('..', 'Autbot/speech_emotion')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import gc
import pandas as pd
import numpy as np

# text emotion
from transformers import pipeline, logging
logging.set_verbosity_error()

# speech emotion
from speech_emotion.features import extract_melspectrogram
from keras.models import load_model


print("----- Loading models -----")
LABELS = pd.read_json(
    "data/labels.json", orient="index"
).to_dict()[0]

DATASET = "TESS"
SPEECH_MODEL = load_model(f"model/{DATASET}_model_4.h5")

TEXT_MODEL = pipeline(
    'text-classification',
    model='j-hartmann/emotion-english-distilroberta-base',
    device=0
)
TEXT_LABELS = [
    'admiration', 'optimism', 'pride', 'realization',
    'relief', 'amusement', 'approval', 'caring', 'desire',
    'gratitude', 'joy', 'desire', 'excitement', 'curiosity',
    'love', 'surprise', 'anger', 'annoyance', 'confusion',
    'remorse', 'sadness', 'disappointment', 'nervousness',
    'disapproval', 'disgust', 'embarrassment', 'grief', 'fear',
]


def speech_emotion(file="user.wav"):
    """
    Predicts the emotion of the audio file.

    Args:
        file (str): Path to the audio file.

    Returns:
        preds (np.array): Predictions of the audio file.
    """
    feature = np.array([extract_melspectrogram(file)])
    preds = SPEECH_MODEL.predict(feature)[0]

    del feature
    gc.collect()

    return preds


def text_emotion(text):
    """
    Predicts the emotion of the text.

    Args:
        text (str): Text to predict the emotion.

    Returns:
        preds (np.array): Predictions of the text.
    """
    happy = ['joy', 'surprise']
    angry = ['anger']
    sad = ['fear', 'sadness', 'disgust']

    preds = [0, 0, 0, 0]
    predictions = TEXT_MODEL(text, return_all_scores=True)[0]

    for prediction in predictions:
        key, val = list(prediction.values())
        if key in angry:
            preds[0] += val

        elif key in happy:
            preds[1] += val

        elif key in sad:
            preds[3] += val

        else:
            preds[2] += val

    return np.array(preds)


def predict_emotion(text="hi", file="user.wav"):
    """
    Predicts the emotion of the text and audio file.

    Args:
        text (str): Text to predict the emotion.
        file (str): Path to the audio file.

    Returns:
        emotion (str): Predicted emotion.
        confidence (float): Confidence of the prediction.
    """
    s = speech_emotion(file)
    t = text_emotion(text)

    if np.argmax(s) == np.argmax(t):
        return LABELS[np.argmax(s)], s[highest]

    # print(
    #     LABELS[np.argmax(s)],
    #     s[np.argmax(s)],
    #     LABELS[np.argmax(t)],
    #     t[np.argmax(t)]
    # )

    preds = s + t
    highest = np.argmax(preds)

    return LABELS[highest], preds[highest]


if __name__ == "__main__":
    while True:
        inp = input("Enter: ")
        if inp == "q":
            break
        print(LABELS[np.argmax(text_emotion(inp))])
    # print(predict_emotion(text="I am okay how are you", file="trial.wav"))
