import os
import librosa
import soundfile as sf

input_folder = "C:/Users/Wojtek/PycharmProjects/speech_recognition/Our_Records_5s"
output_folder = "C:/Users/Wojtek/PycharmProjects/speech_recognition/Our_Records_trimmed"

os.makedirs(output_folder, exist_ok=True)


def trim_audio_files(input_folder, output_folder, sample_rate=22050, top_db_value=22):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            try:
                y, sr = librosa.load(input_file_path, sr=sample_rate)
                # Remove the first 200 ms
                start_sample = int(0.7 * sr)
                y = y[start_sample:]

                # Remove the last 500 ms
                end_sample = int(0.6 * sr)
                y = y[:-end_sample]

                # Trim with top_db
                y_trimmed, _ = librosa.effects.trim(y, top_db=top_db_value)
                # Save the trimmed audio
                sf.write(output_file_path, y_trimmed, sr)
                print(f"Przetworzono plik: {file_name}")
            except Exception as e:
                print(f"Błąd przy przetwarzaniu {file_name}: {e}")


trim_audio_files(input_folder, output_folder)  # Default top_db to 18
print("Przetwarzanie zakończone.")
