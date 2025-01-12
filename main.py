import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import wave
import pandas as pd
import LimitFiles
import PrepareWavFiles
import process_and_save_csv
import process_and_save_csv as Pcsv
from LimitFiles import limit_files_in_subfolders





main_folder = "AudioData"
output_folder = "MelSpectrograms"



# process_and_save_csv.process_and_save_wav_info(main_folder,output_csv="grouped_files_with_duration.csv")

PrepareWavFiles.process_and_save_mel_spectrograms(main_folder, output_folder, n_mels=128, image_size=(128, 128))

#limit_files_in_subfolders(main_folder,2000) #usuniecie plikow z podfolderu jesli jest powyzej 2000


# process_wav_files(main_folder)




# limit_files_in_subfolders(main_folder, max_files=2000)