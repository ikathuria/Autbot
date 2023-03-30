import warnings
warnings.filterwarnings("ignore")

import numpy as np
from transformers import pipeline

print("----- Loading models -----")
MODEL = pipeline(
    'sentiment-analysis',
    model='arpanghoshal/EmoRoBERTa',
    tokenizer='arpanghoshal/EmoRoBERTa',
    device=0
)
LABELS = {
    0: "angry", 1: "happy",
    2: "neutral", 3: "sad"
}

happy = [
    'admiration', 'optimism', 'pride', 'realization', 
    'relief', 'amusement', 'approval', 'caring', 
    'desire', 'gratitude', 'joy', 'excitement', 
    'curiosity', 'love', 'surprise'
]
angry = ['anger', 'annoyance']
sad = [
    'confusion', 'remorse', 'sadness', 'disappointment',
    'nervousness', 'disapproval', 'disgust', 'embarrassment',
    'grief', 'fear',
]


def predict_text_emotion(text):
    happy = [
        'admiration', 'optimism', 'pride', 'realization',
        'relief', 'amusement', 'approval', 'caring',
        'desire', 'gratitude', 'joy', 'excitement',
        'curiosity', 'love', 'surprise'
    ]
    angry = ['anger', 'annoyance']
    sad = [
        'confusion', 'remorse', 'sadness', 'disappointment',
        'nervousness', 'disapproval', 'disgust', 'embarrassment',
        'grief', 'fear',
    ]

    preds = [0, 0, 0, 0]
    predictions = MODEL(text, return_all_scores=True)[0]

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

    return LABELS[np.argmax(preds)], preds[np.argmax(preds)]


if __name__ == "__main__":
    text = "I am happy"
    emo, score = predict_text_emotion(text)
    print(text, emo, score)
