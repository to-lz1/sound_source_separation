import wave
import numpy as np
import sounddevice as sd


def play(wav_file):
    data = wav_file.readframes(wav_file.getnframes())
    data = np.frombuffer(data, dtype=np.int16)

    sd.play(data, wav_file.getframerate())
    sd.wait()


if __name__ == '__main__':
    # load sample data(download it before run this code. see: download.py)
    with wave.open("./CMU_ARCTIC/cmu_us_axb_arctic/wav/arctic_a0004.wav") as wav:
        play(wav)
