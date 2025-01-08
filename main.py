import matplotlib.pylab as plt
import numpy as np
import pandas
import glob
import librosa


#input_file = "AudioData/validation_list.txt"
input_file = "AudioData/testing_list.txt"

#output_file = "AudioData/validation_list_filtered.txt"
output_file = "AudioData/testing_list_filtered.txt"


exclusions = ["cat", "wow", "visual","tree","sheila","marvin","learn","house","happy","forward","follow","dog","cat","bird","bed","backward","background_noise"]


with open(input_file, "r") as infile:
    paths = infile.readlines()


filtered_paths = [
    path for path in paths if not any(path.startswith(exclusion + "/") for exclusion in exclusions)
]

with open(output_file, "w") as outfile:
    outfile.writelines(filtered_paths)

print(f"Sciezke zapisano w '{output_file}'.")
