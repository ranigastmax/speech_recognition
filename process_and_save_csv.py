import os
import glob
import wave
import pandas as pd

def process_and_save_wav_info(main_folder, output_csv="grouped_files_with_duration.csv"):
    grouped_files = {}
    for folder in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder)

        if os.path.isdir(folder_path):
            wav_files = glob.glob(os.path.join(folder_path, "*.wav"))

            if wav_files:
                grouped_files[folder] = wav_files

    data = []
    for label, files in grouped_files.items():
        for file in files:
            with wave.open(file, 'rb') as wav_file:  # dodajemy czas każdego pliku w sekundach
                params = wav_file.getparams()
                num_channels, sample_width, frame_rate, num_frames = params[:4]
                duration = num_frames / float(frame_rate)

                data.append({
                    "Label": label,
                    "FilePath": file,
                    "FileName": os.path.basename(file),
                    "Duration (seconds)": duration
                })

    df = pd.DataFrame(data)
    print(df)
    df.to_csv(output_csv, index=False)
