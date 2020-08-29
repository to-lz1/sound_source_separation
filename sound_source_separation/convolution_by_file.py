import wave

import numpy as np
from scipy import signal

from sound_source_separation import wave_writer, wave_loader, specgram
from sound_source_separation.wave_loader import load_to_mono_array


# NOTE: this script needs your own 16bit impulse response(IR) .wav file.
with wave.open("./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav") as wav_speech, \
        wave.open("my_impulse_response.wav") as wav_ir:
    speech_data = load_to_mono_array(wav_speech)
    impulse_data = load_to_mono_array(wav_ir) / np.iinfo(np.int16).max

    # calculate convolution, and normalize to int16 array
    convoluted_data = signal.convolve(speech_data, impulse_data, mode='full')
    convoluted_data = wave_writer.normalize(convoluted_data)

    wave_writer.write("sample_convolution.wav", convoluted_data, f_rate=wav_speech.getframerate())


with wave.open("sample_convolution.wav") as wav:
    wave_loader.play(wav)

with wave.open("sample_convolution.wav") as wav:
    specgram.show_spectrogram(wav)
