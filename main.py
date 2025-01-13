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
import split_files
import train


# Zapis spektrogramów
main_folder = "AudioData"
output_folder = "MelSpectrograms"

# Selekcjonowanie danych train, val, test
main_split = "Melspectrograms"
output_split = "MelSpectrograms_splitted"

# Limitowanie plików w folderze
#limit_files_in_subfolders(main_folder, 2000)  # Usunięcie plików z podfolderu jeśli jest ich powyżej 2000

# Zapisywanie do CSV
#process_and_save_csv.process_and_save_wav_info(main_folder, output_csv="grouped_files_with_duration.csv")

# Zapisywanie spektrogramu do zdjęcia, przygotowanie
#PrepareWavFiles.process_and_save_mel_spectrograms(main_folder, output_folder, n_mels=128, image_size=(128, 128))

# Podział na: train, val, test
#split_files.split_dataset(main_split, output_split)

# Ścieżki do danych

train_dir = "MelSpectrograms_splitted/train"
val_dir = "MelSpectrograms_splitted/val"
test_dir = "MelSpectrograms_splitted/test"

batch_size = 128  # Liczba próbek w jednym kroku treningowym
num_classes = 11  # Ponieważ ma 11 klas

# Definiowanie rozmiaru
input_shape = (128, 128, 3)

#
# # Trenowanie, ładowanie danych do modelu
train_dataset, val_dataset, test_dataset = train.load_data(train_dir, val_dir, test_dir, batch_size=batch_size)

train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))
test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y))


# Tworzenie modelu
model = train.build_model(input_shape, num_classes)

# Kompilacja modelu
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Ponieważ to klasyfikacja wieloklasowa
    metrics=['accuracy']
)

# Trening modelu
history = model.fit(
    train_dataset,
    epochs=10,            # Liczba epok
    validation_data=val_dataset   # Zbiór walidacyjny (do monitorowania procesu)
)

# Ocena modelu na zbiorze testowym
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Wykres dokładności i straty
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Zapis modelu
model.save('model.h5')



