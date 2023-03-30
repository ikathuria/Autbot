"""
This module contains the Flask interface for the chatbot.

Functions:
    - home: Home page.
    - get_input: Form response for chatbot.
    - about: About page.
"""

import os
import numpy as np
from datetime import datetime

# user interface
from flask import Flask
from flask import request, render_template, url_for, Response, make_response
from flask import redirect, send_from_directory
from record_audio import AudioRecording

# chatbot
from flask_chat import ChatBot


APP = Flask(__name__)

# get user's voice input
USER_RECORDING = AudioRecording()
CHATBOT = ChatBot()

# HOME ---------------------------------------------------------------------------
@APP.route("/")
def home():
    """
    Home page.
    """
    return render_template(
        "index.html"
    )


@APP.route("/", methods=["POST"])
def get_input():
    """
    Form response for chatbot.
    """
    if request.form["user_input"] == "start":
        print("record")
        USER_RECORDING.start_audio()

    if request.form["user_input"] == "stop":
        print("stop record")
        USER_RECORDING.stop_audio(filename="user.wav")
        CHATBOT.speech_to_text(filename="user.wav")
        CHATBOT.generate_response()

    return render_template(
        "index.html",
        user_input=CHATBOT.user.text,
        response=CHATBOT.response
    )


# ABOUT ---------------------------------------------------------------------------
@APP.route("/about")
def about():
    """
    About page.
    """
    return render_template('about.html')


if __name__ == "__main__":
    APP.config["ENV"] = "development"
    APP.config["DEBUG"] = True
    APP.run(debug=True, threaded=True)
