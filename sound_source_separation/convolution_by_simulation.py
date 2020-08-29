import wave

import numpy as np
import pyroomacoustics as pra

from sound_source_separation import wave_writer, wave_loader

# The desired reverberation time and dimensions of the room
reverberation_time60 = 1.5  # seconds
room_dim = [7., 8., 9.]  # meters

# get room material parameter to achieve the desired reverberation time
e_absorption, max_order = pra.inverse_sabine(reverberation_time60, room_dim)
room = pra.ShoeBox(
    room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order
)

with wave.open("./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav") as wav_speech_1, \
        wave.open("./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0005.wav") as wav_speech_2:
    # place sources in the room
    speech_data_1 = wave_loader.load_to_mono_array(wav_speech_1)
    room.add_source([2.5, 4.90, 1.76], signal=speech_data_1)

    speech_data_2 = wave_loader.load_to_mono_array(wav_speech_2)
    room.add_source([4.5, 1.24, 8.76], signal=speech_data_2, delay=3.)

    # place the microphone array in the room
    mic_locs = np.c_[
        [6.3, 4.87, 1.2],  # mic 1
        [6.3, 4.93, 1.2],  # mic 2
    ]
    room.add_microphone_array(mic_locs)

    room.compute_rir()
    rir_data = room.rir[1][0]  # rir of mic 1 and source 0
    rir_data = wave_writer.normalize(rir_data)
    wave_writer.write("sample_ir.wav", rir_data, 16000)

    room.simulate(snr=95.)
    signal = room.mic_array.signals[1]  # simulated signal of mic 0
    signal = wave_writer.normalize(signal)
    wave_writer.write("sample_simulated.wav", signal, wav_speech_1.getframerate())


with wave.open("sample_ir.wav") as wav:
    wave_loader.play(wav)

with wave.open("sample_simulated.wav") as wav:
    wave_loader.play(wav)
