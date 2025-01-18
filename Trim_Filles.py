import os
import librosa
import soundfile as sf
import numpy as np

input_folder = "Our_Records_5s"
output_folder = "Our_Records_trimmed"


os.makedirs(output_folder, exist_ok=True)

def trim_audio_files(input_folder, output_folder, sample_rate=22050, top_db_value=24):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            try:
                y, sr = librosa.load(input_file_path, sr=sample_rate)

                start_sample = int(0.2 * sr)
                y = y[start_sample:]


                end_sample = int(0.18 * sr)
                y = y[:-end_sample]


                y_trimmed, _ = librosa.effects.trim(y, top_db=top_db_value)

                sf.write(output_file_path, y_trimmed, sr)
                print(f"Przetworzono plik: {file_name}")
            except Exception as e:
                print(f"Błąd przy przetwarzaniu {file_name}: {e}")


trim_audio_files(input_folder, output_folder)
print("Przetwarzanie zakończone.")
