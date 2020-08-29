import numpy as np
import wave

from sound_source_separation import wave_loader, wave_writer


def white_noise(n_sample: int):
    np.random.seed(42)
    data = np.random.normal(scale=0.1, size=n_sample)
    data = (data * np.iinfo(np.int16).max).astype(np.int16)
    return data


if __name__ == '__main__':
    sample_file_name = "./sample_white_noise.wav"
    wave_writer.write(sample_file_name, white_noise(88200))

    with wave.open(sample_file_name) as wav:
        wave_loader.play(wav)
