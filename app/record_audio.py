"""
This module contains functions for recording audio from microphone and saving it to a file.

Functions:
    - AudioRecording: Class to record audio from microphone and save to file.
        - start_audio: Start recording audio from microphone.
        - stop_audio: Stop recording audio from microphone.
"""

import wave
import pyaudio


class AudioRecording():
    """
    Class to record audio from microphone and save to file.
    """

    def __init__(self):
        """
        Initialize the class.
        """
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.bits_per_sample = 16

        self.recording = False
        self.last_chat = None

    def start_audio(self):
        """
        Start recording audio.
        """
        # set recording to true
        self.recording = True

        # array to store frames
        self.frames = []

        self.stop = False
        self.audio = pyaudio.PyAudio()

        # start stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        print("--- Recording ---")
        while not self.stop:
            self.frames.append(
                self.stream.read(self.chunk)
            )

    def stop_audio(self, filename="user.wav"):
        """
        Stop recording audio and saves to wav file.

        Args:
            filename (str): Name of the file to save the audio to.
        """
        self.stop = True
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        # set recording to false
        self.recording = False

        print("--- audio recording complete ---")
