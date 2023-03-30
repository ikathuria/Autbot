"""
This module contains functions for recording audio from microphone and saving it to a file.

Functions:
    - AudioRecording: Class to record audio from microphone and save to file.
        - genHeader: Generate a WAV header.
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
        self.channels = 2
        self.rate = 44100
        self.bits_per_sample = 16

        # array to store frames
        self.frames = []

    def genHeader(self, sampleRate, bitsPerSample, channels):
        """
        Generate a WAV header.

        Args:
            sampleRate (int): Sample rate of the audio.
            bitsPerSample (int): Bits per sample of the audio.
            channels (int): Number of channels in the audio.
        
        Returns:
            o: WAV header.
        """
        datasize = 2000 * 10**6
        # (4byte) Marks file as RIFF
        o = bytes("RIFF", 'ascii')
        # (4byte) File size in bytes excluding this and RIFF marker
        o += (datasize + 36).to_bytes(4, 'little')
        # (4byte) File type
        o += bytes("WAVE", 'ascii')
        # (4byte) Format Chunk Marker
        o += bytes("fmt ", 'ascii')
        # (4byte) Length of above format data
        o += (16).to_bytes(4, 'little')
        # (2byte) Format type (1 - PCM)
        o += (1).to_bytes(2, 'little')
        # (2byte)
        o += (channels).to_bytes(2, 'little')
        # (4byte)
        o += (sampleRate).to_bytes(4, 'little')
        # (4byte)
        o += (sampleRate * channels * bitsPerSample // 8
              ).to_bytes(4, 'little')
        # (2byte)
        o += (channels * bitsPerSample // 8
              ).to_bytes(2, 'little')
        # (2byte)
        o += (bitsPerSample).to_bytes(2, 'little')
        # (4byte) Data Chunk Marker
        o += bytes("data", 'ascii')
        # (4byte) Data size in bytes
        o += (datasize).to_bytes(4, 'little')

        return o

    def start_audio(self):
        """
        Start recording audio.
        """
        print("Recording audio")
        self.stop = False
        self.audio = pyaudio.PyAudio()

        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            # input_device_index=1,
            # output=True,
            frames_per_buffer=self.chunk
        )

        while True:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

            if self.stop == True:
                break

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

        print("Audio recording complete")
