import matplotlib.pyplot as plt
import numpy as np
import wave


def show_spectrogram(wav_file):
    data = wav_file.readframes(wav_file.getnframes())
    data = np.frombuffer(data, dtype=np.int16)

    fig = plt.figure(figsize=(10, 4))
    spectrum, freq, t, im = plt.specgram(data,
                                         NFFT=512,
                                         noverlap=512/16*15,
                                         Fs=wav_file.getframerate(),
                                         cmap='gray')

    fig.colorbar(im).set_label('Intensity[dB]')
    plt.xlabel('Time[sec]')
    plt.ylabel('Frequency[Hz]')
    plt.show()


if __name__ == '__main__':
    with wave.open("./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav") as wav:
        show_spectrogram(wav)
