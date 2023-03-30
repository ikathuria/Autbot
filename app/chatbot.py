"""
This module contains functions for the chatbot for the CLI.

Functions:
    - User: Class to represent a user.
        - say_goodbye: Say goodbye to the user based on their emotion.

    - ChatBot: Class to represent the chatbot.
        - system_setup: Setup the system for the chatbot.
        - system_clear: Clear the screen.
        - listen: Listen to user and convert speech to text.
        - speak: Convert text to speech and speak to user.
        - generate_response: Generate a response to the user's input.
        - wake_up: Wake up the chatbot.
        - action_english: Teach the user English.
        - action_time: Tell the user the current time.
"""

import os
import time
import datetime
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gingerit.gingerit import GingerIt

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

        self.emotion = None
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

    def __init__(self, name="Autbot"):
        """
        Initialize the chatbot.
        """
        print("----- Starting up chatbot -----")
        self.system_setup()

        self.name = name

        self.user = User()

        self.response = None
        self.awake = True

    def system_setup(self):
        """
        Setup the system for the chatbot.
        """
        if os.name == "nt":
            self.system = "windows"
        else:
            self.system = "mac"

    def system_clear(self):
        """
        Clear the screen.
        """
        if self.system == "windows":
            os.system("cls")
        else:
            os.system("clear")

    def listen(self):
        """
        Listen to user and convert speech to text.
        """
        recognizer = sr.Recognizer()
        print("Listening...")
        with sr.Microphone() as source:
            audio = recognizer.listen(source)

        with open("user.wav", "wb") as f:
            f.write(audio.get_wav_data())

        self.user.text = "ERROR"

        try:
            self.user.text = recognizer.recognize_google(audio).lower()

            if self.user.text != "ERROR":
                self.user.emotion, self.user.emotion_score = predict_emotion(
                    text=self.user.text,
                    file="user.wav"
                )

                print('\n\n\n')
                print(f"{self.user.emotion}: {self.user.emotion_score:.2f}")
                print('\n\n\n')

        except Exception as e:
            print(e)

        self.user.corrected_text = ENGLISH_PARSER.parse(
            self.user.text
        )['result']
        print(self.user.name, "  --> ", self.user.text)
        print(f"({self.user.corrected_text})")

    def speak(self):
        """
        Convert text to speech.
        """
        print("Autbot --> ", self.response)

        speaker = gTTS(
            text=self.response, lang="en", slow=False
        )
        speaker.save("response.mp3")

        statbuf = os.stat("response.mp3")
        mbytes = statbuf.st_size / 1024
        duration = mbytes / 200

        if self.system == "windows":
            os.system("start response.mp3")
        else:
            os.system("afplay response.mp3")

        time.sleep(int(60 * duration))

    def generate_response(self):
        """
        Generate a response based on the user input.
        """
        if self.user.text == "ERROR":
            self.response = "Sorry, I couldn't understand you. Could you repeat that?"

        # say goodbye
        elif any(i in self.user.text for i in ["bye", "exit", "close"]):
            self.response = self.user.say_goodbye()
            # self.save_history(self.user.name)

            os.remove("./response.mp3")
            os.remove("./user.wav")

            self.awake = False

        # learn english
        elif "learn english" in self.user.text:
            pass

        # see current time
        elif "time" in self.user.text:
            self.response = self.action_time()

        # conversation
        else:
            prompt = f"{self.user.text}. My emotion is {self.user.emotion}."
            input_ids = CHAT_TOKENIZER([prompt], return_tensors="pt")

            chat_response_ids = CHAT_MODEL.generate(**input_ids)

            self.response = CHAT_TOKENIZER.batch_decode(
                chat_response_ids, skip_special_tokens=True
            )[0]

    def wake_up(self):
        """
        Wake up the chatbot.
        """
        self.response = "Hello, I am Autbot. What is your name?"
        self.speak()
        self.listen()

        while self.user.text == "ERROR":
            self.response = "Sorry, I couldn't understand you. Could you repeat that?"
            self.speak()
            self.listen()

        self.user.name = self.user.text
        # self.load_history(self.user.name)

        self.response = f"Nice to meet you {self.user.name}. What would you like to talk about today?"
        self.speak()

    def action_english(self):
        """
        Teach english to the user.

        1. Correct grammatical mistakes.

        """
        pass

    @staticmethod
    def action_time():
        """
        Return the current time.
        """
        return datetime.datetime.now().time().strftime('%H:%M')


if __name__ == "__main__":
    chatbot = ChatBot()

    # wake up the chatbot
    chatbot.wake_up()

    while chatbot.awake:
        chatbot.listen()
        chatbot.generate_response()
        chatbot.speak()
        # chatbot.system_clear()

    print("----- Closing down chatbot -----")
