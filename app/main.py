import os
from datetime import datetime

from flask import Flask
from flask import request, render_template, url_for, Response, make_response
from flask import redirect, send_from_directory

import sounddevice
from scipy.io.wavfile import write

import transformers
# from emotion_recognition import predict_emotion


APP = Flask(__name__)

# model definition
NLP = transformers.pipeline(
    "conversational", model="microsoft/DialoGPT-medium"
)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# HOME ---------------------------------------------------------------------------
@APP.route("/")
def home():
    """
    Home page.
    """
    return render_template(
        "index.html", start_datetime=datetime.now().strftime("%H:%M")
    )


@APP.route("/", methods=["POST"])
def get_input():
    """
    Form response for chatbot.
    """
    print("here")

    if request.form["user_input"] == "stop":
        print(request.files)
        f = request.files['audio_data']
        with open('../user.wav', 'wb') as audio:
            f.save(audio)

    return render_template(
        "index.html", start_datetime=datetime.now().strftime("%H:%M")
    )


# RESPONSE ---------------------------------------------------------------------------
@APP.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    # return chatbot_response(userText)


# ABOUT ---------------------------------------------------------------------------
@APP.route("/about")
def about():
    """
    About page.
    """
    return render_template('about.html')


@APP.errorhandler(404)
def not_found():
    """
    Page not found.
    """
    return make_response(
        render_template("404.html"), 404
    )


if __name__ == "__main__":
    APP.config["ENV"] = "development"
    APP.config["DEBUG"] = True
    APP.run()
