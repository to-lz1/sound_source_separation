import numpy as np
import wave

from sound_source_separation import player


def white_noise(n_sample: int):
    np.random.seed(42)
    data = np.random.normal(scale=0.1, size=n_sample)
    data = (data * np.iinfo(np.int16).max).astype(np.int16)
    return data


def write_wave(data, name: str):
    with wave.open(name, 'w') as wave_out:
        wave_out.setnchannels(1)
        wave_out.setsampwidth(2)
        wave_out.setframerate(44100)
        wave_out.writeframes(data)


if __name__ == '__main__':
    sample_file_name = "./sample_white_noise.wav"
    write_wave(white_noise(80000), sample_file_name)

    wav = wave.open(sample_file_name)
    player.play(wav)
