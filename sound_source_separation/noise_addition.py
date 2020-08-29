import wave

from sound_source_separation import wave_writer, wave_loader, specgram
from sound_source_separation.noise_generator import white_noise

with wave.open("./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0003.wav") as wav:
    speech_data = wave_loader.load_to_mono_array(wav)

    n_speech = wav.getnframes()
    n_noise_only = 40000
    noise_data = white_noise(n_sample=n_noise_only+n_speech)

    mixed_data = noise_data
    mixed_data[n_noise_only:] += speech_data

    wave_writer.write("sample_noise_mixed.wav", mixed_data, f_rate=wav.getframerate())

with wave.open("sample_noise_mixed.wav") as wav:
    wave_loader.play(wav)

with wave.open("sample_noise_mixed.wav") as wav:
    specgram.show_spectrogram(wav)
