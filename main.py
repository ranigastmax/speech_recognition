import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import glob
import librosa
import os
import wave


def process_wav_files(main_folder):
   
    categories = ['down', 'up', 'one', 'eight', 'five']  # To pozniej do zmiany na razie wczytujemy po jednym zeby nie zabic komputera
    selected_files = [] 

 
    for category in categories:
        folder_path = os.path.join(main_folder, category)
        if os.path.isdir(folder_path):
         
            files = glob.glob(os.path.join(folder_path, '*.wav'))
            if files:  
                first_file = files[0]
                selected_files.append((category, first_file))
            else:
                print(f"Folder '{category}' jest pusty lub brak  .wav!")
        else:
            print(f"Folder '{category}' nie istnieje")

   
    for category, file_path in selected_files:
        print(f"Spektogram pliku: {file_path} z grupy {category}")

        try:
          
            y, sr = librosa.load(file_path)

            fft= librosa.stft(y)  # FFT


            S_db = librosa.amplitude_to_db(np.abs(fft), ref=np.max)   #spektogram


        #display
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_db, x_axis='time', y_axis='hz', sr=sr)
            plt.title(f'Spektrogram dla {os.path.basename(file_path)} z grupy: {category}')
            plt.colorbar(format='%+2.0f dB')
            plt.show()

            # Zwrócenie rozmiaru spektrum
            print(f"Spectrogram shape: {S_db.shape}")

        except Exception as e:
            print(f"Error {file_path}: {e}")

#---------------------------------------------------------usuniecie zbyt duzej ilosci datasetu
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

#------------------------------------------------------------------------------------------


#-------------------------------------------------------------To pozniej sie moze przydac jak bedziemy wgrywac dane w siec CNN
# filtered_paths_file = "AudioData/testing_list_filtered.txt"
#
# main_folder = "AudioData"
#

# grouped_files = {}
# for folder in os.listdir(main_folder):
#     folder_path = os.path.join(main_folder, folder)
#
#     if os.path.isdir(folder_path):
#         wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
#
#         if wav_files:
#             grouped_files[folder] = wav_files
#
# data = []
# for label, files in grouped_files.items():
#     for file in files:
#
#         with wave.open(file, 'rb') as wav_file: #dodajemy czas kazdego pliku w sekundach
#             params = wav_file.getparams()
#             num_channels, sample_width, frame_rate, num_frames = params[:4]
#             duration = num_frames / float(frame_rate)
#
#             data.append({
#                 "Label": label,
#                 "FilePath": file,
#                 "FileName": os.path.basename(file),
#                 "Duration (seconds)": duration
#             })
#
# df = pd.DataFrame(data)
# print(df)
# df.to_csv("grouped_files_with_duration.csv", index=False)


main_folder = "AudioData"

process_wav_files(main_folder)

