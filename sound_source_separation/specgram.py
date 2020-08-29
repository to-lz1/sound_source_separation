import matplotlib.pyplot as plt
import numpy as np
import wave

from sound_source_separation.wave_loader import load_to_mono_array


def show_spectrogram(wav_file):
    data = load_to_mono_array(wav_file)

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
