import sounddevice as sd
from scipy.io.wavfile import write



def record_audio(file_name, duration, sample_rate):
    print(f"Nagrywanie... Powiedz coś! (czas nagrywania: {duration} sekund)")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    write(file_name, sample_rate, audio)
    print(f"Nagranie zapisane jako {file_name}")
