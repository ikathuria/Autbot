"""
This module contains functions for plotting the data and model history.

Functions:
    - display_melspectrogram: Displays a mel spectrogram for a given audio file.
    - display_waveplot: Displays a waveplot for a given audio file.
    - display_spectrogram: Displays a spectrogram for a given audio file.
    - display_model_history: Plots the model history.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def display_melspectrogram(path, e):
    """
    Mel Spectrograms are a representation of the short-term power spectrum of a sound,
    based on a linear scale of frequency and a logarithmic scale of the amplitude.
    This function displays a mel spectrogram for a given audio file.

    Args:
        S_DB: mel spectrogram of the audio file.
        sr: sampling rate of the audio file.
        hop_length: hop length of the audio file.
        e: emotion of the audio file.
    """

    n_fft = 2048
    hop_length = 512
    n_mels = 128

    data, sr = librosa.load(path)

    S = librosa.feature.melspectrogram(
        y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S = librosa.util.fix_length(S, size=300)

    S_DB = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(
        S_DB, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram for audio with {} emotion".format(e), size=15)
    plt.show()


def display_waveplot(path, e):
    """
    Waveplots let us know the loudness of the audio at a given time.
    This function displays a waveplot for a given audio file.

    Args:
        path: path to the audio file.
        e: emotion of the audio file.
    """
    data, sr = librosa.load(path)

    plt.figure(figsize=(10, 3))
    plt.title("Waveplot for audio with {} emotion".format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()


def display_spectrogram(path, e):
    """
    A spectrogram is a visual representation of the spectrum of frequencies
    of sound or other signals as they vary with time. It"s a representation
    of frequencies changing with respect to time for given audio/music signals.
    This function displays a spectrogram for a given audio file.

    Args:
        path: path to the audio file.
        e: emotion of the audio file.
    """
    data, sr = librosa.load(path)

    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))

    plt.figure(figsize=(12, 3))
    plt.title("Spectrogram for audio with {} emotion".format(e), size=15)
    librosa.display.specshow(
        Xdb, sr=sr, x_axis="time", y_axis="hz"
    )

    plt.colorbar()
    plt.show()


def display_model_history(model_history, val=True):
    """
    This function displays the model history for accuracy and loss.

    Args:
        model_history: model history.
        val: if True, validation accuracy and loss are displayed.
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(model_history["accuracy"])
    if val:
        ax1.plot(model_history["val_accuracy"])
    ax1.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax1.set_title("Model Accuracy")
    if val:
        ax1.legend(["accuracy", "val_accuracy"], loc="upper left")
    else:
        ax1.legend(["accuracy"], loc="upper left")

    ax2.plot(model_history["loss"])
    if val:
        ax2.plot(model_history["val_loss"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax1.set_title("Model Loss")
    if val:
        ax2.legend(["loss", "val_loss"], loc="upper left")
    else:
        ax2.legend(["loss"], loc="upper left")

    plt.show()
