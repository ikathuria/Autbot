"""
This module contains functions to load the different datasets.

Functions:
    - load_ravdess: loads the RAVDESS dataset into a dataframe.
    - load_crema: loads the CREMA dataset into a dataframe.
    - load_tess: loads the TESS dataset into a dataframe.
    - load_savee: loads the SAVEE dataset into a dataframe.
    - load_emodb: loads the EMODB dataset into a dataframe.
    - load_iemocap: loads the IEMOCAP dataset into a dataframe.
    - load_all: loads the different datasets into a dataframe.
    - load_dataset: loads the data into a dataframe.
"""

import os
import pandas as pd


def load_ravdess(path, classes=[]):
    """
    Loading RAVDESS dataset into a dataframe.

    Args:
        path: path to the dataset.
        classes: list of classes to be used for training.

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
            part = file.split(".")[0]
            part = part.split("-")
            # third part in each file represents the emotion associated to that file.
            file_emotion.append(int(part[2]))
            file_path.append(path + dir + "/" + file)

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=["Emotions"])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=["Path"])
    data_df = pd.concat([emotion_df, path_df], axis=1)

    # changing integers to actual emotions.
    data_df.Emotions.replace({
        1: "neutral", 2: "calm", 3: "happy", 4: "sad",
        5: "angry", 6: "fear", 7: "disgust", 8: "surprise"
    }, inplace=True)

    for x, i in data_df.iterrows():
        if i.Emotions not in classes:
            data_df.drop(x, inplace=True)

    data_df = data_df.reset_index(drop=True)

    data_df.to_csv("data/RAVDESS_data_path.csv", index=False)

    return data_df


def load_crema(path, classes=[]):
    """
    Loading CREMA dataset into a dataframe.

    Args:
        path: path to the dataset.
        classes: list of classes to be used for training.

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
        part = file.split("_")
        if part[2] == "SAD":
            file_emotion.append("sad")
        elif part[2] == "ANG":
            file_emotion.append("angry")
        elif part[2] == "DIS":
            file_emotion.append("disgust")
        elif part[2] == "FEA":
            file_emotion.append("fear")
        elif part[2] == "HAP":
            file_emotion.append("happy")
        elif part[2] == "NEU":
            file_emotion.append("neutral")
        else:
            file_emotion.append("Unknown")

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=["Emotions"])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=["Path"])
    data_df = pd.concat([emotion_df, path_df], axis=1)

    for x, i in data_df.iterrows():
        if i.Emotions not in classes:
            data_df.drop(x, inplace=True)

    data_df = data_df.reset_index(drop=True)

    data_df.to_csv("data/CREMA_data_path.csv", index=False)

    return data_df


def load_tess(path, classes=[]):
    """
    Loading TESS dataset into a dataframe.

    Args:
        path: path to the dataset.
        classes: list of classes to be used for training.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """

    tess_directory_list = os.listdir(path)

    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        directories = os.listdir(path + dir)
        for file in directories:
            part = file.split(".")[0]
            part = part.split("_")[2]
            if part == "ps":
                file_emotion.append("surprise")
            else:
                file_emotion.append(part)
            file_path.append(path + dir + "/" + file)

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=["Emotions"])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=["Path"])
    data_df = pd.concat([emotion_df, path_df], axis=1)

    for x, i in data_df.iterrows():
        if i.Emotions not in classes:
            data_df.drop(x, inplace=True)

    data_df = data_df.reset_index(drop=True)

    data_df.to_csv("data/TESS_data_path.csv", index=False)

    return data_df


def load_savee(path, classes=[]):
    """
    Loading SAVEE dataset into a dataframe.

    Args:
        path: path to the dataset.
        classes: list of classes to be used for training.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """

    savee_directory_list = os.listdir(path)

    file_emotion = []
    file_path = []

    for file in savee_directory_list:
        file_path.append(path + file)
        part = file.split("_")[1]
        ele = part[:-6]
        if ele == "a":
            file_emotion.append("angry")
        elif ele == "d":
            file_emotion.append("disgust")
        elif ele == "f":
            file_emotion.append("fear")
        elif ele == "h":
            file_emotion.append("happy")
        elif ele == "n":
            file_emotion.append("neutral")
        elif ele == "sa":
            file_emotion.append("sad")
        else:
            file_emotion.append("surprise")

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=["Emotions"])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=["Path"])
    data_df = pd.concat([emotion_df, path_df], axis=1)

    for x, i in data_df.iterrows():
        if i.Emotions not in classes:
            data_df.drop(x, inplace=True)

    data_df = data_df.reset_index(drop=True)

    data_df.to_csv("data/SAVEE_data_path.csv", index=False)

    return data_df


def load_emodb(path, classes=[]):
    """
    Loading EMODB dataset into a dataframe.

    Args:
        path: path to the dataset.
        classes: list of classes to be used for training.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """

    emodb_directory_list = os.listdir(path)

    file_emotion = []
    file_path = []

    for file in emodb_directory_list:
        file_path.append(path + file)
        ele = file[5]
        if ele == "W":
            file_emotion.append("angry")
        elif ele == "E":
            file_emotion.append("disgust")
        elif ele == "A":
            file_emotion.append("fear")
        elif ele == "F":
            file_emotion.append("happy")
        elif ele == "N":
            file_emotion.append("neutral")
        elif ele == "T":
            file_emotion.append("sad")
        else:
            file_emotion.append("bored")

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=["Emotions"])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=["Path"])
    data_df = pd.concat([emotion_df, path_df], axis=1)

    for x, i in data_df.iterrows():
        if i.Emotions not in classes:
            data_df.drop(x, inplace=True)

    data_df = data_df.reset_index(drop=True)

    data_df.to_csv("data/EMODB_data_path.csv", index=False)

    return data_df


def load_iemocap(path, classes=[]):
    """
    Loading IEOMCAP dataset into a dataframe.

    Args:
        path: path to the dataset.
        classes: list of classes to be used for training.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """

    data_df = pd.read_csv(path)

    for x, i in data_df.iterrows():
        if i.Emotions not in classes:
            data_df.drop(x, inplace=True)

    data_df = data_df.reset_index(drop=True)

    return data_df


def load_all(paths=[], classes=[]):
    """
    Loading all datasets into a merged dataframe.

    Args:
        paths: paths to the 4 datasets.
        classes: list of classes to be removed from the dataframe.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """
    ravdess_df = load_ravdess(paths[0], classes)
    crema_df = load_crema(paths[1], classes)
    tess_df = load_tess(paths[2], classes)
    savee_df = load_savee(paths[3], classes)
    emodb_df = load_emodb(paths[4], classes)
    iemocap_df = load_iemocap(paths[5], classes)

    data_df = pd.concat([
        ravdess_df, crema_df, tess_df,
        savee_df, emodb_df, iemocap_df
    ], axis=0)

    for x, i in data_df.iterrows():
        if i.Emotions not in classes:
            data_df.drop(x, inplace=True)

    data_df = data_df.reset_index(drop=True)

    data_df.to_csv("data/ALL_data_path.csv", index=False)

    return data_df


def load_dataset(classes, dataset):
    """
    Loading the dataset into a dataframe.

    Args:
        classes: list of classes to be removed from the dataframe.
        dataset: name of the dataset to be loaded.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """
    if dataset == "RAVDESS":
        data_df = load_ravdess(
            "data/datasets/RAVDESS/", classes
        )

    elif dataset == "CREMA":
        data_df = load_crema(
            "data/datasets/CREMA-D/", classes
        )

    elif dataset == "TESS":
        data_df = load_tess(
            "data/datasets/TESS/", classes
        )

    elif dataset == "SAVEE":
        data_df = load_savee(
            "data/datasets/SAVEE/", classes
        )

    elif dataset == "EMODB":
        data_df = load_emodb(
            "data/datasets/EMO-DB/", classes
        )

    elif dataset == "IEMOCAP":
        data_df = load_iemocap(
            "data/IEMOCAP_data_path_original.csv", classes
        )

    elif dataset == "ALL":
        data_df = load_all(
            [
                "data/datasets/RAVDESS/",
                "data/datasets/CREMA-D/",
                "data/datasets/TESS/",
                "data/datasets/SAVEE/",
                "data/datasets/EMO-DB/",
                "data/IEMOCAP_data_path_original.csv",
            ],
            classes
        )

    return data_df
