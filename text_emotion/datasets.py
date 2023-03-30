"""
This module contains functions to load the different datasets.

Functions:
    - load_isear: loads the ISEAR dataset into a dataframe.
    - load_emory loads the EMORY NLP dataset into a dataframe.
    - load_meld: loads the MELD dataset into a dataframe.
    - load_all: loads the different datasets into a dataframe.
    - load_dataset: loads the data into a dataframe.
"""

import os
import json
import pandas as pd


def load_isear(path):
    data_df = pd.read_csv(path).drop('ID', axis=1)
    data_df.columns = ['Emotion', 'Text']

    data_df['Emotion'] = data_df['Emotion'].replace(
        ['joy', 'fear', 'sadness', 'anger'],
        ['happy', 'sad', 'sad', 'angry']
    )

    return data_df


def load_emory(path):
    file_paths =[
        path.format('trn'),
        path.format('dev'),
        path.format('tst')
    ]

    data = {
        'Emotion': [],
        'Text': []
    }

    for file_path in file_paths:
        temp = json.load(open(file_path, 'r', encoding='utf8'))
        for episode in temp['episodes']:
            for scene in episode['scenes']:
                for utterance in scene['utterances']:
                    data['Emotion'].append(utterance['emotion'].lower())
                    data['Text'].append(utterance['transcript'].lower())

    data_df = pd.DataFrame(data)

    data_df['Emotion'] = data_df['Emotion'].replace(
        ['joyful', 'powerful', 'mad', 'scared', 'peaceful'],
        ['happy', 'happy', 'angry', 'sad', 'happy']
    )

    return data_df


def load_meld(path):
    def format(data):
        data = data[['Emotion', 'Utterance']]
        data.columns = ['Emotion', 'Text']
        return data

    train = format(pd.read_csv(path.format('train')))
    dev = format(pd.read_csv(path.format('dev')))
    test = format(pd.read_csv(path.format('test')))

    data_df = pd.concat([train, dev, test])

    data_df['Emotion'] = data_df['Emotion'].replace(
        ['joy', 'surprise', 'sadness', 'disgust', 'fear', 'anger'],
        ['happy', 'happy', 'sad', 'sad', 'sad', 'angry']
    )

    return data_df


def load_all(paths=[]):
    """
    Loading all datasets into a merged dataframe.

    Args:
        paths: paths to the 4 datasets.
        classes: list of classes to be removed from the dataframe.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """
    isear = load_isear(paths[0])
    emory = load_emory(paths[1])
    meld = load_meld(paths[2])

    data_df = pd.concat([isear, emory, meld], axis=0)
    data_df = data_df.reset_index(drop=True)

    data_df.to_csv("data/text/ALL_data_path.csv", index=False)

    return data_df


def load_dataset(dataset):
    """
    Loading the dataset into a dataframe.

    Args:
        classes: list of classes to be removed from the dataframe.
        dataset: name of the dataset to be loaded.

    Returns:
        data_df: dataframe containing the path of the audio files and their emotions.
    """
    if dataset == "ISEAR":
        data_df = load_isear("data/datasets/ISEAR/eng_dataset.csv")

    elif dataset == "EMORY":
        data_df = load_emory("data/datasets/EMORY/emotion-detection-{}.json")
    
    elif dataset == "MELD":
        data_df = load_meld("data/datasets/MELD/{}_sent_emo.csv")

    elif dataset == "ALL":
        data_df = load_all([
            "data/datasets/ISEAR/eng_dataset.csv",
            "data/datasets/EMORY/emotion-detection-{}.json",
            "data/datasets/MELD/{}_sent_emo.csv",
        ])

    return data_df
