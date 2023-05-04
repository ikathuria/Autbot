"""
This module contains functions for the chatbot for the Flask app.

Functions:
    - User: Class to represent a user.
        - say_goodbye: Say goodbye to the user based on their emotion.

    - ChatBot: Class to represent the chatbot.
        - speech_to_text: Convert speech to text.
        - text_to_speech: Convert text to speech.
        - generate_response: Generate a response to the user's input.
        - action_english: Teach the user English.
"""
import os
import time
import json
import pandas as pd
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gingerit.gingerit import GingerIt
from keybert import KeyBERT
from icrawler.builtin import GoogleImageCrawler

from gtts import gTTS
import speech_recognition as sr

from emotion_recognition import predict_emotion

import warnings
warnings.filterwarnings("ignore")

ENGLISH_PARSER = GingerIt()
CHAT_TOKENIZER = AutoTokenizer.from_pretrained(
    "facebook/blenderbot-400M-distill"
)
CHAT_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/blenderbot-400M-distill"
)
KW_MODEL = KeyBERT(model='all-mpnet-base-v2')


class User():
    """
    User class.
    """

    def __init__(self, name="User"):
        """
        Initialize the user.
        """
        self.name = name

        self.text = None
        self.corrected_text = None

        self.emotion = "neutral"
        self.emotion_score = 0

    def say_goodbye(self):
        """
        Say goodbye to the user based on current emotion.
        """
        if self.emotion == 'happy':
            return f"I am glad I could help you {self.name}! Have a good day!"

        elif self.emotion in ["sad", "angry"]:
            return f"I am sorry I could not help you {self.name}. I am here if you want to talk again."

        else:
            return f"It was nice talking to you {self.name}. Have a good day!"


class ChatBot():
    """
    Chatbot class.
    """

    def __init__(self):
        """
        Initialize the chatbot.
        """
        print("----- Starting up chatbot -----")
        self.google_crawler = GoogleImageCrawler(
            storage={'root_dir': './app/static/images'}
        )
        
        self.user = User()
        self.response = None
        self.response_audio = 1
        self.history = []

        self.load_history()

    def speech_to_text(self, filename="user.wav"):
        """
        Listen to user and convert speech to text.
        """
        recognizer = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)

        self.user.text = "ERROR"

        try:
            self.user.text = recognizer.recognize_google(audio).lower()

            if self.user.text != "ERROR":
                self.user.emotion, self.user.emotion_score = predict_emotion(
                    text=self.user.text,
                    file=filename
                )

                self.user.corrected_text = ENGLISH_PARSER.parse(
                    self.user.text
                )['result']

                self.history.append({
                    "text": [f"{self.user.text}", f"({self.user.corrected_text})"],
                    "emotion": self.user.emotion,
                    "isReceived": False
                })

                print('\n\n\n')
                print(f"User: {self.user.text}")
                print(f"{self.user.emotion}: {self.user.emotion_score:.2f}")
                print('\n\n\n')

        except Exception as e:
            print(e)

    def text_to_speech(self):
        """
        Convert text to speech.
        """
        # remove old audio
        try:
            os.remove(f"./app/static/audio/{self.response_audio - 1}.mp3")
        except Exception as e:
            pass

        speaker = gTTS(
            text=self.response,
            lang="en",
            slow=False
        )

        speaker.save(f"./app/static/audio/{self.response_audio}.mp3")
        self.response_audio += 1
        time.sleep(1)

    def get_image(self):
        """
        Get an image related to the response.
        """
        # remove old images
        try:
            os.remove(f"./app/static/images/000001.jpg")
        except Exception as e1:
            try:
                os.remove(f"./app/static/images/000001.png")
            except Exception as e2:
                pass

        # find keyword from response
        kw = KW_MODEL.extract_keywords(
            self.response,
            keyphrase_ngram_range=(1, 1),
            stop_words='english',
            highlight=False,
            top_n=1
        )[0][0]

        self.google_crawler.crawl(
            keyword=kw + " gif",
            max_num=1,
            overwrite=True
        )

    def generate_response(self):
        """
        Generate a response based on the user input.
        """
        if self.user.text == "ERROR":
            self.response = "Sorry, I couldn't understand you. Could you repeat that?"

        # say goodbye
        elif any(i in self.user.text for i in ["bye", "exit", "close"]):
            self.response = self.user.say_goodbye()

        # learn english
        # elif "learn english" in self.user.text:
        #     pass

        # conversation
        else:
            prompt = f"{self.user.text}. My emotion is {self.user.emotion}."
            input_ids = CHAT_TOKENIZER([prompt], return_tensors="pt")

            chat_response_ids = CHAT_MODEL.generate(**input_ids)

            self.response = CHAT_TOKENIZER.batch_decode(
                chat_response_ids, skip_special_tokens=True
            )[0]

        self.history.append({
            "text": self.response,
            "isReceived": True
        })

        # find keyword from response
        kw = KW_MODEL.extract_keywords(
            self.response,
            keyphrase_ngram_range=(1, 1),
            stop_words='english',
            highlight=False,
            top_n=1
        )[0][0]

        # convert response to speech and save
        self.text_to_speech()

        # download image related to response
        self.get_image()

        # save chat history
        self.save_history()

    def action_english(self):
        """
        Teach english to the user.

        1. Correct grammatical mistakes.
        """
        pass

    def load_history(self):
        """
        Load the conversation history from a file.
        """
        with open(f"./app/static/history/{self.user.name}.json") as file:
            self.history = json.load(file)
        
        self.response = self.history[-1]['text']
        self.get_image()
        self.text_to_speech()

        for i in self.history:
            if not i['isReceived']:
                self.user.text = i['text'][0]
                self.user.corrected_text = i['text'][1]
                self.user.emotion = i['emotion']
                break

    def save_history(self):
        """
        Save the conversation history to a file.
        """
        with open(f"./app/static/history/{self.user.name}.json", "w") as file:
            json.dump(self.history, file)


if __name__ == "__main__":
    chatbot = ChatBot()

    # wake up the chatbot
    chatbot.wake_up()

    while chatbot.awake:
        chatbot.speech_to_text()
        chatbot.generate_response()
        chatbot.text_to_speech()
        # chatbot.system_clear()

    print("----- Closing down chatbot -----")
