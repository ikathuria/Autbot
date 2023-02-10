<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>

<h3 align="center">Project Title</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/ikathuria/Autbot.svg)](https://github.com/ikathuria/Autbot/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/ikathuria/Autbot.svg)](https://github.com/ikathuria/Autbot/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> Few lines describing your project.
    <br> 
</p>

## 📝 Table of Contents

- [📝 Table of Contents](#-table-of-contents)
- [🧐 About ](#-about-)
- [🏁 Getting Started ](#-getting-started-)
  - [Prerequisites](#prerequisites)
  - [Datasets](#datasets)
  - [Installing](#installing)
- [🔧 Running the tests ](#-running-the-tests-)
  - [Break down into end to end tests](#break-down-into-end-to-end-tests)
  - [And coding style tests](#and-coding-style-tests)
- [🎈 Usage ](#-usage-)
- [🚀 Deployment ](#-deployment-)
- [⛏️ Built Using ](#️-built-using-)
- [✍️ Authors ](#️-authors-)
- [🎉 Acknowledgements ](#-acknowledgements-)

## 🧐 About <a name = "about"></a>

1. English Grammar Learning + Hindi? 
2. Emotion recongition with
   * Face expressions
   * Speech?
   * Chat (speech to text)

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them.

```
Give examples
```

### Datasets

1. [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://zenodo.org/record/1188976#.Y9dkm3ZBy3A)  
   Here is the filename identifiers as per the official RAVDESS website:
   * Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
   * Vocal channel (01 = speech, 02 = song).
   * Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
   * Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
   * Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
   * Repetition (01 = 1st repetition, 02 = 2nd repetition).
   * Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

   So, here's an example of an audio filename. 02-01-06-01-02-01-12.mp4  
   This means the meta data for the audio file is:
   * Video-only (02)
   * Speech (01)
   * Fearful (06)
   * Normal intensity (01)
   * Statement "dogs" (02)
   * 1st Repetition (01)
   * 12th Actor (12) - Female (as the actor ID number is even)
2. [Crowd-sourced Emotional Mutimodal Actors Dataset (CREMA-D)](https://github.com/CheyneyComputerScience/CREMA-D)
3. [Surrey Audio-Visual Expressed Emotion (SAVEE)](http://kahlan.eps.surrey.ac.uk/savee/Database.html)  
   The audio files in this dataset are named in such a way that the prefix letters describes the emotion classes as follows:

   * 'a' = 'anger'
   * 'd' = 'disgust'
   * 'f' = 'fear'
   * 'h' = 'happiness'
   * 'n' = 'neutral'
   * 'sa' = 'sadness'
   * 'su' = 'surprise'
4. [Toronto emotional speech set (TESS)](https://tspace.library.utoronto.ca/handle/1807/24487)

### Installing

A step by step series of examples that tell you how to get a development env running.

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo.

## 🔧 Running the tests <a name = "tests"></a>

Explain how to run the automated tests for this system.

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## 🎈 Usage <a name="usage"></a>

Add notes about how to use the system.

## 🚀 Deployment <a name = "deployment"></a>

Add additional notes about how to deploy this on a live system.

## ⛏️ Built Using <a name = "built_using"></a>

- [MongoDB](https://www.mongodb.com/) - Database
- [Express](https://expressjs.com/) - Server Framework
- [VueJs](https://vuejs.org/) - Web Framework
- [NodeJs](https://nodejs.org/en/) - Server Environment

## ✍️ Authors <a name = "authors"></a>

- [@ikathuria](https://github.com/ikathuria)
- [@Kamad11](https://github.com/Kamad11)

See also the list of [contributors](https://github.com/ikathuria/Autbot/contributors) who participated in this project.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Hat tip to anyone whose code was used
- Inspiration
- References
