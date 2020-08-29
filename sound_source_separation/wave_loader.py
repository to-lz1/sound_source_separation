import wave
import numpy as np
import sounddevice as sd


def load_to_mono_array(wav_file):
    data = wav_file.readframes(wav_file.getnframes())
    data = np.frombuffer(data, dtype=np.int16)

    n_channels = wav_file.getnchannels()
    if n_channels == 2:
        data = data[::2]  # convert to L mono. r_channel is: data[1::2]
    return data


def play(wav_file):
    data = load_to_mono_array(wav_file)
    sd.play(data, wav_file.getframerate())
    sd.wait()


if __name__ == '__main__':
    with wave.open("./CMU_ARCTIC/cmu_us_axb_arctic/wav/arctic_a0001.wav") as wav:
        play(wav)
