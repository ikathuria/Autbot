import os
import numpy as np
import pandas as pd

import librosa
import librosa.display

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras import Input
from keras.models import Sequential
from keras.layers import Dropout, Bidirectional, TimeDistributed, BatchNormalization
from keras.layers import LSTM, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from hyperas.distributions import uniform, choice
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim

import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt


# DATASETS #################################################################################################
def load_ravdess(path, classes=[], save_dir='data'):
    """
    Loading RAVDESS dataset into a dataframe.

    Args:
        path: path to the dataset.
        classes: list of classes to be used for training.
        save_dir: directory to save the dataframe.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """

    ravdess_directory_list = os.listdir(path)

    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        # as their are 20 different actors in the
        # previous directory we need to extract files for each actor.
        actor = os.listdir(path + dir)
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            # third part in each file represents the emotion associated to that file.
            file_emotion.append(int(part[2]))
            file_path.append(path + dir + '/' + file)

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    data_df = pd.concat([emotion_df, path_df], axis=1)

    # changing integers to actual emotions.
    data_df.Emotions.replace({
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
        5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
    }, inplace=True)

    for x, i in data_df.iterrows():
        if i.Emotions not in classes:
            data_df.drop(x, inplace=True)

    data_df = data_df.reset_index(drop=True)

    data_df.to_csv(save_dir + "/RAVDESS_data_path.csv", index=False)

    return data_df


def load_crema(path, classes=[], save_dir='data'):
    """
    Loading CREMA dataset into a dataframe.

    Args:
        path: path to the dataset.
        classes: list of classes to be used for training.
        save_dir: directory to save the dataframe.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """

    crema_directory_list = os.listdir(path)

    file_emotion = []
    file_path = []

    for file in crema_directory_list:
        # storing file paths
        file_path.append(path + file)
        # storing file emotions
        part = file.split('_')
        if part[2] == 'SAD':
            file_emotion.append('sad')
        elif part[2] == 'ANG':
            file_emotion.append('angry')
        elif part[2] == 'DIS':
            file_emotion.append('disgust')
        elif part[2] == 'FEA':
            file_emotion.append('fear')
        elif part[2] == 'HAP':
            file_emotion.append('happy')
        elif part[2] == 'NEU':
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    data_df = pd.concat([emotion_df, path_df], axis=1)

    for x, i in data_df.iterrows():
        if i.Emotions not in classes:
            data_df.drop(x, inplace=True)

    data_df = data_df.reset_index(drop=True)

    data_df.to_csv(save_dir + "/CREMA_data_path.csv", index=False)

    return data_df


def load_tess(path, classes=[], save_dir='data'):
    """
    Loading TESS dataset into a dataframe.

    Args:
        path: path to the dataset.
        classes: list of classes to be used for training.
        save_dir: directory to save the dataframe.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """

    tess_directory_list = os.listdir(path)

    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        directories = os.listdir(path + dir)
        for file in directories:
            part = file.split('.')[0]
            part = part.split('_')[2]
            if part == 'ps':
                file_emotion.append('surprise')
            else:
                file_emotion.append(part)
            file_path.append(path + dir + '/' + file)

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    data_df = pd.concat([emotion_df, path_df], axis=1)

    for x, i in data_df.iterrows():
        if i.Emotions not in classes:
            data_df.drop(x, inplace=True)

    data_df = data_df.reset_index(drop=True)

    data_df.to_csv(save_dir + "/TESS_data_path.csv", index=False)

    return data_df


def load_savee(path, classes=[], save_dir='data'):
    """
    Loading SAVEE dataset into a dataframe.

    Args:
        path: path to the dataset.
        classes: list of classes to be used for training.
        save_dir: directory to save the dataframe.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """

    savee_directory_list = os.listdir(path)

    file_emotion = []
    file_path = []

    for file in savee_directory_list:
        file_path.append(path + file)
        part = file.split('_')[1]
        ele = part[:-6]
        if ele == 'a':
            file_emotion.append('angry')
        elif ele == 'd':
            file_emotion.append('disgust')
        elif ele == 'f':
            file_emotion.append('fear')
        elif ele == 'h':
            file_emotion.append('happy')
        elif ele == 'n':
            file_emotion.append('neutral')
        elif ele == 'sa':
            file_emotion.append('sad')
        else:
            file_emotion.append('surprise')

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    data_df = pd.concat([emotion_df, path_df], axis=1)

    for x, i in data_df.iterrows():
        if i.Emotions not in classes:
            data_df.drop(x, inplace=True)

    data_df = data_df.reset_index(drop=True)

    data_df.to_csv(save_dir + "/SAVEE_data_path.csv", index=False)

    return data_df


def load_datasets(paths=[], classes=[], save_dir='data'):
    """
    Loading all datasets into a merged dataframe.

    Args:
        paths: paths to the 4 datasets.
        classes: list of classes to be removed from the dataframe.
        save_dir: directory to save the merged dataframe.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """
    ravdess_df = load_ravdess(paths[0], classes, save_dir)
    crema_df = load_crema(paths[1], classes, save_dir)
    tess_df = load_tess(paths[2], classes, save_dir)
    savee_df = load_savee(paths[3], classes, save_dir)

    data_df = pd.concat([ravdess_df, crema_df, tess_df, savee_df], axis=0)

    for x, i in data_df.iterrows():
        if i.Emotions not in classes:
            data_df.drop(x, inplace=True)

    data_df = data_df.reset_index(drop=True)

    data_df.to_csv(save_dir + "/data_path.csv", index=False)

    return data_df


def load_data(classes, dataset, save_dir):
    """
    Loading the dataset into a dataframe.

    Args:
        classes: list of classes to be removed from the dataframe.
        dataset: name of the dataset to be loaded.
        save_dir: directory to save the dataframe.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """
    if dataset == "RAVDESS":
        data_df = load_ravdess(
            "../data/RAVDESS/", classes, save_dir
        )

    elif dataset == "CREMA":
        data_df = load_crema(
            "../data/CREMA-D/", classes, save_dir
        )

    elif dataset == "TESS":
        data_df = load_tess(
            "../data/TESS/", classes, save_dir
        )

    elif dataset == "SAVEE":
        data_df = load_savee(
            "../data/SAVEE/", classes, save_dir
        )

    elif dataset == "ALL":
        data_df = load_datasets(
            [
                "../data/RAVDESS/",
                "../data/CREMA-D/",
                "../data/TESS/",
                "../data/SAVEE/",
            ],
            classes,
            save_dir
        )

    return data_df


# EDA/PLOTS #################################################################################################
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
        x_axis='time', y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram for audio with {} emotion'.format(e), size=15)
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
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()


def display_spectrogram(path, e):
    """
    A spectrogram is a visual representation of the spectrum of frequencies
    of sound or other signals as they vary with time. It's a representation
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
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(
        Xdb, sr=sr, x_axis='time', y_axis='hz'
    )

    plt.colorbar()
    plt.show()


def display_model_history(model_history, val=True):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(model_history['accuracy'])
    if val:
        ax1.plot(model_history['val_accuracy'])
    ax1.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    if val:
        ax1.legend(['accuracy', 'val_accuracy'], loc='upper left')
    else:
        ax1.legend(['accuracy'], loc='upper left')

    ax2.plot(model_history['loss'])
    if val:
        ax2.plot(model_history['val_loss'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax1.set_title('Model Loss')
    if val:
        ax2.legend(['loss', 'val_loss'], loc='upper left')
    else:
        ax2.legend(['loss'], loc='upper left')

    plt.show()


# FEATURES #################################################################################################
# def extract_zero_crossing(data):
#     """
#     It measures the smoothness of a signal. It usually has higher
#     values for highly percussive sounds like rock and metal.
#     Function extracts Zero Crossing Rate features from the audio file.

#     Args:
#         data: audio data in np array.

#     Returns:
#         zc: Zero Crossing Rate features
#     """
#     zc = np.mean(librosa.zero_crossings(
#         y=data, pad=False
#     ))

#     return zc

    
# def extract_rms(data):
#     """
#     Function extracts RMSE features from the audio file.

#     Args:
#         data: audio data in np array.

#     Returns:
#         rmse: Spectral Centroid features
#     """
#     rms = np.mean(librosa.feature.rms(y=data)[0])

#     return rms

    
# def extract_spectral_centroid(data, sampling_rate):
#     """
#     Spectral centroid is a measure of where the "centre of mass"
#     for a sound is located.
#     Function extracts Spectral Centroid features from the audio file.

#     Args:
#         data: audio data in np array.
#         sampling_rate: sampling rate of the audio file.

#     Returns:
#         sc: Spectral Centroid features
#     """
#     sc = np.mean(librosa.feature.spectral_centroid(
#         y=data, sr=sampling_rate
#     )[0])

#     return sc


# def extract_spectral_rolloff(data, sampling_rate):
#     """
#     It is a measure of the shape of the signal. It represents
#     the frequency at which high frequencies decline to 0.
#     Function extracts Spectral Rolloff features from the audio file.

#     Args:
#         data: audio data in np array.
#         sampling_rate: sampling rate of the audio file.

#     Returns:
#         sr: Spectral Rolloff features
#     """
#     sr = np.mean(librosa.feature.spectral_rolloff(
#         y=data + 0.01, sr=sampling_rate
#     )[0])

#     return sr


# def extract_spectral_bandwidth(data, sampling_rate, p=2):
#     """
#     It is the width of the band of light at one-half the peak
#     maximum (i.e. the full width at half maximum).
#     Function extracts Spectral Bandwidth features from the audio file.

#     Args:
#         data: audio data in np array.
#         sampling_rate: sampling rate of the audio file.
#         p: order of the norm.

#     Returns:
#         sb: Spectral Bandwidth features
#     """
#     sb = np.mean(librosa.feature.spectral_bandwidth(
#         y=data + 0.01, sr=sampling_rate,
#         p=p
#     )[0])

#     return sb


# def extract_chroma(data, sampling_rate):
#     """
#     It indicates how much energy of each pitch class is present.
#     Function extracts Chroma features from the audio file.

#     Args:
#         data: audio data in np array.
#         sampling_rate: sampling rate of the audio file.

#     Returns:
#         chroma: Chroma features
#     """
#     chroma = np.mean(librosa.feature.chroma_stft(
#         y=data, sr=sampling_rate
#     ).T, axis=0)

#     return chroma


# def extract_mfcc(data, sampling_rate, n=20):
#     """
#     Mel-Frequency Cepstral Coefficients (MFCCs) are a small
#     set of features (usually about 10-20) which concisely describe
#     the overall shape of a spectral envelope.
#     Function extracts MFCC features from the audio file.

#     Args:
#         data: audio data in np array.
#         sampling_rate: sampling rate of the audio file.
#         n: number of MFCCs to return.

#     Returns:
#         mfcc: MFCC features
#     """
#     mfcc = np.mean(librosa.feature.mfcc(
#         y=data, sr=sampling_rate, n_mfcc=n
#     ).T, axis=0)

#     return mfcc


# def extract_features(path, mfcc_n=33):
#     """
#     Extract features from the audio file.
#     1. Zero Crossing Rate
#     2. Root Mean Square
#     3. Spectral Centroid
#     4. Spectral Rolloff
#     5. Spectral Bandwidth
#     6. Chroma
#     7. MFCC

#     Args:
#         path: path to the audio file.

#     Returns:
#         features: features extracted from the audio file.
#     """
#     data, sampling_rate = librosa.load(path)

#     zcr = extract_zero_crossing(data)
#     rms = extract_rms(data)
#     sc = extract_spectral_centroid(data, sampling_rate)
#     sr = extract_spectral_rolloff(data, sampling_rate)
#     sb = extract_spectral_bandwidth(data, sampling_rate)
#     chroma = extract_chroma(data, sampling_rate)
#     mfcc = extract_mfcc(data, sampling_rate, n=mfcc_n)

#     features = np.hstack([
#         zcr,
#         rms,
#         sc,
#         sr,
#         sb,
#         chroma,
#         mfcc,
#     ])

#     return features

def extract_melspectrogram(path):
    n_fft = 2048
    hop_length = 512
    n_mels = 128

    data, sr = librosa.load(path)

    S = librosa.feature.melspectrogram(
        y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S = librosa.util.fix_length(S, size=300)

    S_DB = librosa.power_to_db(S, ref=np.max)

    return S_DB


def encode_emotions(labels, save_dir='data/'):
    enc = OneHotEncoder()
    y = enc.fit_transform(labels)

    print(enc.categories_)

    pd.Series(enc.categories_[0]).to_json(save_dir + '/labels.json')

    return y.toarray()


def load_x_y(df, dataset='all', save_dir='data/'):
    """
    Load features and labels from the dataset.

    Args:
        df: dataframe containing the dataset.
        dataset: name of the dataset.
        save_dir: directory to save the features and labels.

    Returns:
        x: features.
        y: labels.
    """
    x = []
    for path in tqdm(df['Path']):
        x.append(extract_melspectrogram(path))

    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

    y = encode_emotions(
        df[['Emotions']],
        save_dir=save_dir
    )

    with open(f"{save_dir}/{dataset}_features.npy", "wb") as f:
        np.save(f, x)

    with open(f"{save_dir}/{dataset}_labels.npy", "wb") as f:
        np.save(f, y)

    return x, y


def load_train_test(classes, dataset, save_dir='data/', reset=False):
    """
    Load train and test features and labels from the dataset.
    """
    if reset:
        x, y = load_x_y(
            load_data(dataset, classes, save_dir=save_dir),
            dataset,
            save_dir=save_dir
        )
    else:
        x = np.load(f"{save_dir}/{dataset}_features.npy")
        y = np.load(f"{save_dir}/{dataset}_labels.npy")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2,
        stratify=y,
        random_state=0
    )

    return x_train, x_test, y_train, y_test


# MODELING #################################################################################################
def build_model(shape=(128, 300, 3), classes=4):
    """
    Build the model architecture.

    Returns:
        model: Keras LSTM model

    """

    model = Sequential([
        # conv1
        Conv2D(
            64, kernel_size=(3, 3), activation='relu',
            padding='valid', input_shape=shape
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),

        # conv2
        Conv2D(
            32, kernel_size=(3, 3), activation='relu',
            padding='valid',
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),
        Dropout(0.2),

        # conv3
        Conv2D(
            64, kernel_size=(3, 3), activation='relu',
            padding='valid',
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),
        Dropout(0.5),

        # flattened
        Flatten(),

        Dense(128, activation='relu'),
        Dense(64, activation='relu'),

        Dense(classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    return model


# OPTIMIZATION #################################################################################################
def data():
    x = np.load("data/melspectrogram/RAVDESS_features.npy")
    y = np.load('data/melspectrogram/RAVDESS_labels.npy')

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
            16, kernel_size=(3, 3), activation='relu',
            padding='same', input_shape=shape
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),

        Conv2D(
            32, kernel_size=(3, 3), activation='relu',
            padding='valid',
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),

        Conv2D(
            64, kernel_size=(3, 3), activation='relu',
            padding='valid',
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),
        Dropout({{uniform(0, 1)}}),

        Conv2D(
            64, kernel_size=(3, 3), activation='relu',
            padding='valid',
        ),
        MaxPooling2D(
            pool_size=(2, 2), strides=2
        ),
        Dropout({{uniform(0, 1)}}),

        Flatten(),

        Dense(128, activation='relu'),
        Dense(64, activation='relu'),

        Dense(4, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping_monitor = EarlyStopping(
        monitor='val_loss', patience=3
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
    print('\nTrain accuracy: %.4f\nTrain loss: %.4f' % (acc, score))

    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy: %.4f\nTest loss: %.4f' % (acc, score))

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(
        model=model,
        data=data,
        algo=tpe.suggest,
        max_evals=7,
        trials=Trials(),
    )
    print(best_run)
    print(best_model)
    # pass
