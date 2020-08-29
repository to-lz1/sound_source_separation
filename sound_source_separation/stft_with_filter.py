import scipy.signal as sp
import numpy as np
import wave

from sound_source_separation import wave_writer, wave_loader, specgram
from sound_source_separation.stft import short_term_fourier_transform


with wave.open("./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0003.wav") as wav:
    f, t, stft_data = short_term_fourier_transform(wav)

    cut_index = 100
    print(f"cut frequency factor that is higher than {f[cut_index]}Hz.")
    stft_data[cut_index:, :] = 0

    _, istft_data = sp.istft(stft_data, fs=wav.getframerate(), nperseg=512, noverlap=256)
    istft_data = istft_data.astype(np.int16)
    wave_writer.write("sample_low_passed.wav", istft_data, wav.getframerate())

with wave.open("sample_low_passed.wav") as wav:
    wave_loader.play(wav)

with wave.open("sample_low_passed.wav") as wav:
    specgram.show_spectrogram(wav)
