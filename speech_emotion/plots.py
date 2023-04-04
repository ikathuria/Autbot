"""
This module contains functions for plotting the data and model history.

Functions:
    - display_counts: Displays the counts of each emotion in the dataset.
    - display_melspectrogram: Displays a mel spectrogram for a given audio file.
    - display_waveplot: Displays a waveplot for a given audio file.
    - display_spectrogram: Displays a spectrogram for a given audio file.
    - display_model_history: Plots the model history.
    - evaluation_plots: Plots the confusion matrix, ROC curve, and precision-recall curve.
"""

import numpy as np
import sklearn.metrics as skmetrics
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns


def display_counts(y, dataset):
    """
    This function displays the counts of each emotion in the dataset.

    Args:
        df: dataframe containing the data.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.countplot(
        np.argmax(y, axis=1),
        ax=ax
    )

    for i in ax.patches:
        ax.text(
            i.get_x() + 0.1,
            i.get_height() + 1,
            str(round(i.get_height(), 2)),
            fontsize=12,
            color="black"
        )

    ax.set_xticks(np.arange(4), ["angry", "happy", "neutral", "sad"])
    plt.title(f"Counts of each emotion in {dataset}")
    plt.show()

    fig.savefig(
        f"./data/plots/{dataset}_counts.png",
        bbox_inches="tight",
        dpi=300,
        facecolor='white'
    )



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


def evaluation_plots(model, x_true, y_true, dataset):
    """
    Plots the confusion matrix, ROC curve, and precision-recall curve.

    Args:
        model: the model to evaluate.
        x_true: the features.
        y_true: the labels.
        dataset: the dataset name.
    """

    y_pred = model.predict(x_true)

    # confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    skmetrics.ConfusionMatrixDisplay.from_predictions(
        np.argmax(y_true, axis=1),
        np.argmax(y_pred, axis=1),
        ax=ax
    )
    plt.show()
    fig.savefig(
        f"./data/plots/{dataset}_cm.png",
        bbox_inches="tight",
        dpi=300,
        facecolor='white'
    )

    # ROC
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    skmetrics.RocCurveDisplay.from_predictions(
        y_true.ravel(),
        y_pred.ravel(),
        name="micro-average OvR",
        color="darkorange",
        ax=ax
    )
    plt.ylim([0.94, 1.01])
    plt.show()
    fig.savefig(
        f"./data/plots/{dataset}_roc.png",
        bbox_inches="tight",
        dpi=300,
        facecolor='white'
    )

    # precision-recall
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    precision = dict()
    recall = dict()
    for i in range(2):
        for j in range(2):
            class_label = i * 2 + j

            precision[class_label], recall[class_label], _ = skmetrics.precision_recall_curve(
                y_true[:, class_label],
                y_pred[:, class_label],

            )
            ax[i][j].plot(
                recall[class_label],
                precision[class_label],
                lw=2,
            )

            ax[i][j].set_title("Class {}".format(class_label))
            ax[i][j].set_xlabel("Recall")
            ax[i][j].set_ylabel("Precision")

    plt.tight_layout()
    plt.show()

    fig.savefig(
        f"./data/plots/{dataset}_pr.png",
        bbox_inches="tight",
        dpi=300,
        facecolor='white'
    )

    return y_pred
