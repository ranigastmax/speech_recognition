import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import glob
import librosa
import os
import wave

# #input_file = "AudioData/validation_list.txt"
# input_file = "AudioData/testing_list.txt"
#
# #output_file = "AudioData/validation_list_filtered.txt"
# output_file = "AudioData/testing_list_filtered.txt"
#
#
# exclusions = ["cat", "wow", "visual","tree","sheila","marvin","learn","house","happy","forward","follow","dog","cat","bird","bed","backward","background_noise"]
#
#
# with open(input_file, "r") as infile:
#     paths = infile.readlines()
#
#
# filtered_paths = [
#     path for path in paths if not any(path.startswith(exclusion + "/") for exclusion in exclusions)
# ]
#
# with open(output_file, "w") as outfile:
#     outfile.writelines(filtered_paths)
#
# print(f"Sciezke zapisano w '{output_file}'.")





filtered_paths_file = "AudioData/testing_list_filtered.txt"

main_folder = "AudioData"

# Słownik do przechowywania grup (label -> lista plików)
grouped_files = {}

# Iteracja po wszystkich podfolderach w głównym katalogu
for folder in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder)

    if os.path.isdir(folder_path):  # Sprawdzamy, czy to jest folder
        # Wyszukaj wszystkie pliki .wav w podfolderze
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))

        if wav_files:
            grouped_files[folder] = wav_files  # Dodaj folder i pliki do grupy

# Tworzenie DataFrame z grup
data = []
for label, files in grouped_files.items():
    for file in files:
        # Odczytanie pliku .wav i obliczenie długości w sekundach
        with wave.open(file, 'rb') as wav_file:
            params = wav_file.getparams()
            num_channels, sample_width, frame_rate, num_frames = params[:4]
            duration = num_frames / float(frame_rate)  # Długość w sekundach

            data.append({
                "Label": label,
                "FilePath": file,
                "FileName": os.path.basename(file),
                "Duration (seconds)": duration
            })

# Konwersja na DataFrame
df = pd.DataFrame(data)

# Wyświetlenie DataFrame
print(df)

df.to_csv("grouped_files_with_duration.csv", index=False)