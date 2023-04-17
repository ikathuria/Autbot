"""
This module contains the Flask interface for the chatbot.

Functions:
    - home: Home page.
    - get_input: Form response for chatbot.
    - about: About page.
"""
from threading import Thread
from datetime import datetime

# user interface
from flask import Flask
from flask import request, render_template
from flask import url_for, redirect, jsonify

# chatbot
from record_audio import AudioRecording
from flask_chat import ChatBot


APP = Flask(__name__)

# get user's voice input
USER_RECORDING = AudioRecording()
CHATBOT = ChatBot()


# INACTIVITY ---------------------------------------------------------------------
@APP.route("/inactivity")
def check_inactivity():
    """
    Check if the user is inactive for more than 1 minute.
    """
    global CHATBOT, USER_RECORDING

    if USER_RECORDING.last_chat == None:
        return jsonify({"time_diff": 0})

    while not USER_RECORDING.recording:
        curr_time = datetime.now()
        time_diff = (curr_time - USER_RECORDING.last_chat).seconds

        if time_diff >= 45:
            CHATBOT.history.append({
                "text": "Do you want to talk about something else?",
                "isReceived": True
            })
            CHATBOT.save_history()
            break

        return jsonify({"time_diff": time_diff})
    return jsonify({"time_diff": time_diff})


# HOME ---------------------------------------------------------------------------
@APP.route("/", methods=["GET", "POST"])
def home():
    """
    Homepage for the chatbot.
    """
    global CHATBOT, USER_RECORDING

    if request.method == "POST":
        if request.form["user_input"] == "start":
            USER_RECORDING.last_chat = None
            USER_RECORDING.start_audio()

        elif request.form["user_input"] == "stop":
            USER_RECORDING.stop_audio(filename="user.wav")

            # work the chatbot magic
            CHATBOT.speech_to_text(filename="user.wav")
            CHATBOT.generate_response()

            USER_RECORDING.last_chat = datetime.now()
            # Thread(target=check_inactivity).start()

    return render_template(
        "index.html",
        user_emotion=CHATBOT.user.emotion,
        history=CHATBOT.history[::-1],
        last_chat=USER_RECORDING.last_chat,
        recording=USER_RECORDING.recording,
    )



# ABOUT ---------------------------------------------------------------------------
@APP.route("/about")
def about():
    """
    About page.
    """
    return render_template('about.html')


if __name__ == "__main__":
    APP.config.update(
        ENV = "development",
        DEBUG = True,
        TESTING = True,
        # SERVER_NAME = "localhost:5000",
    )
    APP.run(threaded=True)
