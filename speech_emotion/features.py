"""
This module contains functions to extract features from the audio files.

Functions:
    - extract_melspectrogram: extracts mel spectrogram from an audio file.
    - augmentation: performs data augmentation on the audio files.
    - encode_emotions: encodes the emotions into one-hot vectors.
    - load_x_y: loads the data into numpy arrays.
    - split: splits the dataset into train and test sets.
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
from tensorflow.image import random_flip_left_right, random_brightness
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from speech_emotion.datasets import load_dataset


def extract_melspectrogram(path):
    """
    Mel Spectrograms are a representation of the short-term
    power spectrum of a sound, based on a linear scale of
    frequency and a logarithmic scale of the amplitude.
    This function extracts a mel spectrogram for a given audio file.

    Args:
        path: path to the audio file.

    Returns:
        spectrogram: mel spectrogram of the audio file.
    """
    data, sr = librosa.load(path)

    # Extract Mel spectrograms using librosa
    spectrogram = librosa.feature.melspectrogram(y=data, sr=sr)
    spectrogram = librosa.util.fix_length(spectrogram, size=300)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Reshape spectrograms to match CNN input shape
    spectrogram = spectrogram.reshape(
        spectrogram.shape[0], spectrogram.shape[1], 1
    )

    return spectrogram


def augmentation(spectrogram):
    """
    Apply random augmentations to the spectrogram.

    Args:
        spectrogram: spectrogram to apply the augmentations.

    Returns:
        spectrogram: augmented spectrogram.
    """
    # Randomly flip the spectrogram horizontally
    spectrogram = random_flip_left_right(spectrogram)

    # Randomly adjust the brightness
    spectrogram = random_brightness(spectrogram, 0.2)

    return spectrogram


def encode_emotions(labels):
    """
    Encode the emotions using OneHotEncoder.

    Args:
        labels: emotions to encode.

    Returns:
        y: encoded labels.
    """
    enc = OneHotEncoder()
    y = enc.fit_transform(labels)

    print(enc.categories_)

    pd.Series(enc.categories_[0]).to_json("data/labels.json")

    return y.toarray()


def load_x_y(reset, classes, dataset="all"):
    """
    Load features and labels from the dataset.

    Args:
        df: dataframe containing the dataset.
        dataset: name of the dataset.

    Returns:
        x: features.
        y: labels.
    """
    if reset:
        df = load_dataset(classes, dataset)

    elif dataset == "ALL":
        all_datasets = [
            "RAVDESS", "CREMA", "TESS",
            "SAVEE", "EMODB", "IEMOCAP"
        ]
        df = pd.concat([
            pd.read_csv("data/" + f"{i}_data_path.csv") for i in all_datasets
        ])

        x = np.concatenate([
            np.load("data/npy/" + f"{i}_features.npy") for i in all_datasets
        ])
        y = np.concatenate([
            np.load("data/npy/" + f"{i}_labels.npy") for i in all_datasets
        ])

        return df, x, y


    else:
        df = pd.read_csv("data/" + f"{dataset}_data_path.csv")
        x = np.load("data/npy/" + f"{dataset}_features.npy")
        y = np.load("data/npy/" + f"{dataset}_labels.npy")

        return df, x, y

    x = []
    y = []

    for path, emotion in tqdm(zip(df["Path"], df["Emotions"]), total=len(df)):
        s = extract_melspectrogram(path)

        # Original spectrogram
        x.append(s)

        # 2 augmentations for sample
        x.append(augmentation(s).numpy())
        x.append(augmentation(s).numpy())

        # 3 labels per sample
        y.extend([emotion] * 3)

    x = np.array(x)
    y = np.array(y).reshape(-1, 1)

    y = encode_emotions(y)

    with open(f"data/npy/{dataset}_features.npy", "wb") as f:
        np.save(f, x)

    with open(f"data/npy/{dataset}_labels.npy", "wb") as f:
        np.save(f, y)

    return df, x, y


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
