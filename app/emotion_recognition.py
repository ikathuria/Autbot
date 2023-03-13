import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'Autbot')))
sys.path.append(os.path.abspath(os.path.join('..', 'Autbot/speech_emotion')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import pandas as pd
import numpy as np
from speech_emotion.features import extract_melspectrogram
from keras.models import load_model

print("----- Loading models -----")
DATASET = "TESS"
SPEECH_MODEL = load_model(f"model/{DATASET}_model_4.h5")
SPEECH_LABELS = pd.read_json("data/speech_labels.json", orient="index").to_dict()[0]

def speech_emotion(file="user.wav"):
    """
    Predicts the emotion of the audio file.
    """
    feature = np.array([extract_melspectrogram(file)])
    preds = SPEECH_MODEL.predict(feature)[0]

    del feature
    gc.collect()

    return preds


def text_emotion(text):
    """
    Predicts the emotion of the text.
    """
    return "happy"


def predict_emotion(text, file="user.wav"):
    """
    Predicts the emotion of the text and audio file.
    """
    s = speech_emotion(file)
    t = text_emotion(text)

    return SPEECH_LABELS[np.argmax(s)]

    if np.argmax(s) == np.argmax(t):
        return SPEECH_LABELS[np.argmax(s)]

    preds = s + t
    print(s, t, preds)

    return SPEECH_LABELS[np.argmax(preds)]




if __name__ == "__main__":
    print(predict_emotion(text="I am angry", file="user.wav"))
