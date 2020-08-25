import scipy.signal as sp
import numpy as np
import wave


def short_term_fourier_transform(wav_file):
    data = wav_file.readframes(wav_file.getnframes())
    data = np.frombuffer(data, dtype=np.int16)

    return sp.stft(data, fs=wav_file.getframerate(), nperseg=512, noverlap=256)


if __name__ == '__main__':
    with wave.open("./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0002.wav") as wav:
        f, t, stft_data = short_term_fourier_transform(wav)
        print(f"shape: {np.shape(stft_data)}")
        print(f"f[Hz]: {f}")
        print(f"t[sec]: {t}")
