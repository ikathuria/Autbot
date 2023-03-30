# Index
- [Index](#index)
- [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](#ryerson-audio-visual-database-of-emotional-speech-and-song-ravdess)
- [Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D)](#crowd-sourced-emotional-multimodal-actors-dataset-crema-d)
- [Toronto emotional speech set (TESS)](#toronto-emotional-speech-set-tess)
- [Surrey Audio-Visual Expressed Emotion (SAVEE)](#surrey-audio-visual-expressed-emotion-savee)
- [Berlin Database of Emotional Speech (EMODB)](#berlin-database-of-emotional-speech-emodb)
  - [Information about the speakers](#information-about-the-speakers)
  - [Code of emotions](#code-of-emotions)
- [The Interactive Emotional Dyadic Motion Capture Database (IEMOCAP)](#the-interactive-emotional-dyadic-motion-capture-database-iemocap)
- [International Survey On Emotion Antecedents And Reactions (ISEAR)](#international-survey-on-emotion-antecedents-and-reactions-isear)
- [EmoryNLP](#emorynlp)
- [Multimodal EmotionLines Dataset (MELD)](#multimodal-emotionlines-dataset-meld)

# [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://zenodo.org/record/1188976#.Y9dkm3ZBy3A)

The filename identifiers as per the official RAVDESS website:

* Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
* Vocal channel (01 = speech, 02 = song).
* Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
* Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
* Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
* Repetition (01 = 1st repetition, 02 = 2nd repetition).
* Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

So, here's an example of an audio filename. 02-01-06-01-02-01-12.mp4This means the meta data for the audio file is:

* Video-only (02)
* Speech (01)
* Fearful (06)
* Normal intensity (01)
* Statement "dogs" (02)
* 1st Repetition (01)
* 12th Actor (12) - Female (as the actor ID number is even)

# [Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D)](https://github.com/CheyneyComputerScience/CREMA-D)

# [Toronto emotional speech set (TESS)](https://tspace.library.utoronto.ca/handle/1807/24487)

# [Surrey Audio-Visual Expressed Emotion (SAVEE)](http://kahlan.eps.surrey.ac.uk/savee/Database.html)

The audio files in this dataset are named in such a way that the prefix letters describes the emotion classes as follows:

| letter |  emotion  |
| :----: | :-------: |
|   a    |   anger   |
|   d    |  disgust  |
|   f    |   fear    |
|   h    | happiness |
|   n    |  neutral  |
|   sa   |  sadness  |
|   su   | surprise  |

# [Berlin Database of Emotional Speech (EMODB)](http://emodb.bilderbar.info/docu/)

Every utterance is named according to the same scheme:

* Positions 1-2: number of speaker
* Positions 3-5: code for text
* Position 6: emotion (sorry, letter stands for german emotion word)
* Position 7: if there are more than two versions these are numbered a, b, c,....

Example: 03a01Fa.wav is the audio file from Speaker 03 speaking text a01 with the emotion "Freude" (Happiness).

## Information about the speakers

* 03 - male, 31 years old
* 08 - female, 34 years
* 09 - female, 21 years
* 10 - male, 32 years
* 11 - male, 26 years
* 12 - male, 30 years
* 13 - female, 32 years
* 14 - female, 35 years
* 15 - male, 25 years
* 16 - female, 31 years

## Code of emotions

| letter | emotion (english) | emotion (german) |
| :----: | :---------------: | :--------------: |
|   W   |       anger        |   Ärger (Wut)    |
|   L   |      boredom       |    Langeweile    |
|   E   |      disgust       |       Ekel       |
|   A   |   anxiety/fear     |      Angst       |
|   F   |     happiness      |      Freude      |
|   T   |      sadness       |      Trauer      |
|   N   |      neutral       |     Neutral      |

# [The Interactive Emotional Dyadic Motion Capture Database (IEMOCAP)](https://sail.usc.edu/iemocap/index.html)

# [International Survey On Emotion Antecedents And Reactions (ISEAR)](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)
Over a period of many years during the 1990s, a large group of psychologists all over the world collected data in the ISEAR project, directed by Klaus R. Scherer and Harald Wallbott. Student respondents, both psychologists and non-psychologists, were asked to report situations in which they had experienced all of 7 major emotions (joy, fear, anger, sadness, disgust, shame, and guilt). In each case, the questions covered the way they had appraised the situation and how they reacted. The final data set thus contained reports on seven emotions each by close to 3000 respondents in 37 countries on all 5 continents.

# [EmoryNLP]()
EmoryNLP comprises 97 episodes, 897 scenes, and 12,606 utterances, where each utterance is annotated with one of the seven emotions borrowed from the six primary emotions in the Willcox (1982)'s feeling wheel, sad, mad, scared, powerful, peaceful, joyful, and a default emotion of neutral.

# [Multimodal EmotionLines Dataset (MELD)](https://affective-meld.github.io/)
MELD has more than 1400 dialogues and 13000 utterances from Friends TV series. Multiple speakers participated in the dialogues. Each utterance in a dialogue has been labeled by any of these seven emotions -- Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear. MELD also has sentiment (positive, negative and neutral) annotation for each utterance.

<img src="https://affective-meld.github.io/sc4.png" width=500px>
