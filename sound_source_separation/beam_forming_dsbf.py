"""Delay and Sum beam former example"""
import wave

import numpy as np
import pyroomacoustics as pra
import scipy.signal as sp

from sound_source_separation import wave_writer, wave_loader
from sound_source_separation.beam_forming import calculate_steering_vector

room_dim = [7., 8., 9.]  # meters
room = pra.ShoeBox(room_dim, fs=16000, max_order=0)  # room with no reverb

with wave.open("./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav") as wav_speech:
    # place source in the room
    speech_data = wave_loader.load_to_mono_array(wav_speech)
    frame_rate = wav_speech.getframerate()
    source_locs = np.c_[
        [2.5, 4.90, 1.76]
    ]
    room.add_source([2.5, 4.90, 1.76], signal=speech_data)

    # place the microphone array in the room
    # (generally speaking, the more mic placed in the room, the more calculated sound become clear)
    mic_locs = np.c_[
        [6.3, 4.87, 1.2],
        [6.3, 4.93, 1.2],
        [5.3, 2.93, 3.2],
        [5.3, 6.33, 2.2],
        [2.3, 3.93, 6.2],
        [2.3, 5.93, 8.2],
    ]
    room.add_microphone_array(mic_locs)

    room.simulate(snr=15.)
    signals = room.mic_array.signals
    signal_sample = wave_writer.normalize(signals[0])
    wave_writer.write("sample_simulated_with_noise.wav", signal_sample, f_rate=frame_rate)

    with wave.open("sample_simulated_with_noise.wav") as wav:
        print("playing simulated speech data, with noise")
        wave_loader.play(wav)

    f, t, stft_data = sp.stft(signals, fs=frame_rate, nperseg=1024)
    vector_near = calculate_steering_vector(mic_locs, source_locs, frequencies=f, use_far=False)
    s_hat = np.einsum("ksm,mkt->skt", np.conjugate(vector_near), stft_data)
    c_hat = np.einsum("skt,ksm->mskt", s_hat, vector_near)
    _, ds_out = sp.istft(c_hat[0], fs=frame_rate, nperseg=1024)

    ds_out = wave_writer.normalize(ds_out)
    wave_writer.write("sample_simulated_with_noise_reduced.wav", ds_out, f_rate=frame_rate)

    with wave.open("sample_simulated_with_noise_reduced.wav") as wav:
        print("playing noise-reduced speech data, with Delay and Sum beam former")
        wave_loader.play(wav)
