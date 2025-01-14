import os
import librosa
import soundfile as sf

input_folder = "C:/Users/Wojtek/PycharmProjects/speech_recognition/Our_Records_5s"
output_folder = "C:/Users/Wojtek/PycharmProjects/speech_recognition/Our_Records_trimmed"

os.makedirs(output_folder, exist_ok=True)

def trim_audio_files(input_folder, output_folder, sample_rate=22050):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            try:
                y, sr = librosa.load(input_file_path, sr=sample_rate)
                y_trimmed, _ = librosa.effects.trim(y, top_db=27)
                sf.write(output_file_path, y_trimmed, sr)
                print(f"Przetworzono plik: {file_name}")
            except Exception as e:
                print(f"Błąd przy przetwarzaniu {file_name}: {e}")

trim_audio_files(input_folder, output_folder)
print("Przetwarzanie zakończone.")
