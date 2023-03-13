import os
import time
import datetime
import numpy as np

import speech_recognition as sr
from gtts import gTTS

from emotion_recognition import predict_emotion

import transformers
import warnings
warnings.filterwarnings("ignore")


class User():
    """
    User class.
    """

    def __init__(self, name="User"):
        """
        Initialize the user.
        """
        self.name = name
        self.emotion = None
        self.text = None

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

    def __init__(self, model=None, name="Autbot"):
        """
        Initialize the chatbot.
        """
        print("----- Starting up chatbot -----")
        self.system_setup()

        self.model = model
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
            self.user.emotion = predict_emotion(
                text=self.user.text,
                file="user.wav"
            )

            print('\n\n\n')
            print(self.user.emotion)
            print('\n\n\n')

        except Exception as e:
            print(e)

        print(self.user.name, "  --> ", self.user.text)

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
        elif any(i in chatbot.user.text for i in ["bye", "exit", "close"]):
            self.response = self.user.say_goodbye()

            os.remove("response.mp3")
            os.remove("user.wav")

            self.awake = False

        # see current time
        elif "time" in self.user.text:
            self.response = self.action_time()

        # conversation
        else:
            chat = nlp(
                transformers.Conversation(self.user.text),
                pad_token_id=50256
            )
            response = str(chat)
            self.response = response[response.find("bot >> ") + 6:].strip()

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
        self.response = f"Nice to meet you {self.user.name}. How can I help you?"
        self.speak()

    @staticmethod
    def action_time():
        """
        Return the current time.
        """
        return datetime.datetime.now().time().strftime('%H:%M')


if __name__ == "__main__":
    nlp = transformers.pipeline(
        "conversational", model="microsoft/DialoGPT-medium"
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    chatbot = ChatBot(model=nlp)

    # wake up the chatbot
    chatbot.wake_up()

    while chatbot.awake:
        chatbot.listen()
        chatbot.generate_response()
        chatbot.speak()
        # chatbot.system_clear()

    print("----- Closing down chatbot -----")
