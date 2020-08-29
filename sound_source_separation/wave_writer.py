import wave

import numpy as np


def normalize(data):
    """normalize audio signal to int16 array(that can be wrote as 16bit .wav file by write() method)."""
    data = data / data.max()
    return (data * np.iinfo(np.int16).max).astype(np.int16)


def write(name: str, data, f_rate=44100):
    with wave.open(name, 'w') as wave_out:
        wave_out.setnchannels(1)
        wave_out.setsampwidth(2)
        wave_out.setframerate(f_rate)
        wave_out.writeframes(data)
