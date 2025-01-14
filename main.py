import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import wave
import pandas as pd
from keras.src.callbacks import EarlyStopping

import LimitFiles
import PrepareWavFiles
import process_and_save_csv
import process_and_save_csv as Pcsv
from LimitFiles import limit_files_in_subfolders
import split_files
import train
import command_record


# Zapis spektrogramów
main_folder = "AudioData"
output_folder = "MelSpectrograms"

# Selekcjonowanie danych train, val, test
main_split = "Melspectrograms"
output_split = "MelSpectrograms_splitted"

# Ścieżki do danych
train_dir = "MelSpectrograms_splitted/train"
val_dir = "MelSpectrograms_splitted/val"
test_dir = "MelSpectrograms_splitted/test"


# Limitowanie plików w folderze
#limit_files_in_subfolders(main_folder, 2000)  # Usunięcie plików z podfolderu jeśli jest ich powyżej 2000

# Zapisywanie do CSV
#process_and_save_csv.process_and_save_wav_info(main_folder, output_csv="grouped_files_with_duration.csv")

# Zapisywanie spektrogramu do zdjęcia, przygotowanie
#PrepareWavFiles.process_and_save_mel_spectrograms(main_folder, output_folder, n_mels=128, image_size=(128, 128))

# Podział na: train, val, test
#split_files.split_dataset(main_split, output_split)



batch_size = 128
num_classes = 11
input_shape = (128, 128, 3)

# Ładowanie danych
train_dataset, val_dataset, test_dataset = train.load_data(train_dir, val_dir, test_dir, batch_size=batch_size)
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))
test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y))

# Tworzenie modelu
model = train.build_model(input_shape, num_classes)

# Kompilacja modelu
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Dodanie EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Trening modelu
history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)

# Ocena modelu
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Wykresy
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
model.save('model_b1.h5')

# Zapis historii treningu
pd.DataFrame(history.history).to_csv('training_history_model_b1.csv', index=False)