import os
from flask import Flask
from flask import request, render_template, url_for, Response, make_response
from flask import redirect, send_from_directory


APP = Flask(__name__)


# @APP.errorhandler(404)
# def not_found():
#     """
#     Page not found.
#     """
#     return make_response(
#         render_template("404.html"), 404
#     )


@APP.route("/")
def home():
    """
    Home page.
    """
    return render_template('index.html')


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
    APP.run()
