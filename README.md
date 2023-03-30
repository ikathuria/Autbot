<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>

<h3 align="center">Autbot</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]() [![GitHub Issues](https://img.shields.io/github/issues/ikathuria/Autbot.svg)](https://github.com/ikathuria/Autbot/issues) [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/ikathuria/Autbot.svg)](https://github.com/ikathuria/Autbot/pulls) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> Autbot is a friendly conversational agent made to help chaildren with Autism improve their socialization and language skills by learning with a emapthetic AI chatbot.
    <br> 
</p>

## 📝 Table of Contents

- [📝 Table of Contents](#-table-of-contents)
- [🧐 About](#-about)
- [🏁 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
- [🎈 Usage](#-usage)
- [📂 Datasets Used](#-datasets-used)
  - [Speech](#speech)
  - [Text](#text)
- [⛏️ Built Using](#️-built-using)
- [✍️ Authors](#️-authors)

## 🧐 About
Autbot is a friendly conversational agent made to help chaildren with Autism improve their socialization and language skills by learning with a emapthetic AI chatbot.

## 🏁 Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. First, clone the repository to your local machine.

```
git clone https://github.com/ikathuria/Autbot.git 
```

### Prerequisites
You can use [pip](https://pip.pypa.io/en/stable/) to install the required packages. Run the following command in the project directory.

```
pip install -r requirements.txt
```

## 🎈 Usage
To test the chatbot, go into the app directory and run [main.py](/app/main.py) for starting the flask interface or [chatbot.py](/app/chatbot.py) for starting the chatbot in the terminal.

To train your own emotion recognition model, run the [speech_main.ipynb](/speech_main.ipynb).

## 📂 Datasets Used
### Speech
1. [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://zenodo.org/record/1188976#.Y9dkm3ZBy3A)
2. [Crowd-sourced Emotional Mutimodal Actors Dataset (CREMA-D)](https://github.com/CheyneyComputerScience/CREMA-D)
3. [Surrey Audio-Visual Expressed Emotion (SAVEE)](http://kahlan.eps.surrey.ac.uk/savee/Database.html)
4. [Toronto emotional speech set (TESS)](https://tspace.library.utoronto.ca/handle/1807/24487)
5. [Berlin Database of Emotional Speech (EMODB)](http://emodb.bilderbar.info/docu/)
6. [The Interactive Emotional Dyadic Motion Capture Database (IEMOCAP)](https://sail.usc.edu/iemocap/index.html)

### Text
1. [International Survey On Emotion Antecedents And Reactions (ISEAR)](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)
2. [EmoryNLP](https://github.com/emorynlp/emotion-detection)
3. [Multimodal EmotionLines Dataset (MELD)](https://affective-meld.github.io/)

[Detailed description of datasets](/DATASET.MD)

## ⛏️ Built Using
- [Python](https://www.python.org/) - Programming Language
- [Flask](https://flask.palletsprojects.com/en/1.1.x/) - Web Framework
- [Tensorflow](https://www.tensorflow.org/) - Deep Learning Library
- [HuggingFace](https://huggingface.co/) - Model Repository

## ✍️ Authors
- [@ikathuria](https://github.com/ikathuria)
- [@Kamad11](https://github.com/Kamad11)

See also the list of [contributors](https://github.com/ikathuria/Autbot/contributors) who participated in this project.

<!-- ## 🎉 Acknowledgements -->
