import wave


def write(name: str, data, f_rate=44100):
    with wave.open(name, 'w') as wave_out:
        wave_out.setnchannels(1)
        wave_out.setsampwidth(2)
        wave_out.setframerate(f_rate)
        wave_out.writeframes(data)
