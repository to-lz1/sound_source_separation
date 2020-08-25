import wave

import numpy as np
import scipy.signal as sp

from sound_source_separation import wave_writer, wave_player, specgram
from sound_source_separation.stft import short_term_fourier_transform

with wave.open("sample_noise_mixed.wav") as wav:
    f, t, stft_data = short_term_fourier_transform(wav)
    amp = np.abs(stft_data)
    phase = stft_data / np.maximum(amp, 1.e-20)

    n_noise_only = 40000
    n_noise_only_frame = np.sum(t < n_noise_only / wav.getframerate())
    noise_part_amp = amp[:, :n_noise_only_frame]
    noise_amp_mean = np.sqrt(np.mean(np.power(noise_part_amp, 2), axis=1, keepdims=True))

    p = 1.0
    alpha = 3.3
    eps = 0.01 * np.power(amp, p)

    processed_amp = np.power(np.maximum(np.power(amp, p) - alpha * np.power(noise_amp_mean, p), eps), 1./p)
    processed_amp = processed_amp * phase

    _, istft_data = sp.istft(processed_amp, fs=wav.getframerate(), nperseg=512, noverlap=256)
    istft_data = istft_data.astype(np.int16)
    wave_writer.write("sample_noise_reduced.wav", istft_data, wav.getframerate())

with wave.open("sample_noise_reduced.wav") as wav:
    wave_player.play(wav)

with wave.open("sample_noise_reduced.wav") as wav:
    specgram.show_spectrogram(wav)
