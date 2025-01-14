import sounddevice as sd
from scipy.io.wavfile import write
import os

def record_audio(OUTPUT_FILE, duration, sample_rate):


    folder_name = "Our_Records_5s"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    full_path = os.path.join(folder_name, OUTPUT_FILE)

    if not full_path.endswith(".wav"):
        full_path += ".wav"

    print(f"Nagrywanie... Powiedz coś! (czas nagrywania: {duration} sekund)")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    write(full_path, sample_rate, audio)
    print(f"Nagranie zapisane jako {full_path}")