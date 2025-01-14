import os
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

# Ładowanie modelu
model = tf.keras.models.load_model('Models/model_b1.h5')


# Funkcja do konwersji pliku WAV na spektrogram Mel
def wav_to_mel_spectrogram(wav_path, image_size=(128, 128), n_mels=128):
    y, sr = librosa.load(wav_path, sr=22050)
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_mel_db = librosa.amplitude_to_db(S_mel, ref=np.max)

    # Normalizacja
    S_mel_db_normalized = (S_mel_db - np.mean(S_mel_db)) / np.std(S_mel_db)
    S_mel_db_normalized = np.clip(S_mel_db_normalized, -2, 2)
    S_mel_db_normalized = (S_mel_db_normalized + 2) / 4  # Zakres [0, 1]

    # Konwersja do RGB za pomocą mapy kolorów viridis
    resized_S_mel_db = plt.cm.viridis(S_mel_db_normalized)
    resized_S_mel_db = resized_S_mel_db[:, :, :3]  # RGB (3 kanały)

    # Zmiana rozmiaru obrazu
    resized_S_mel_db = tf.image.resize(resized_S_mel_db, image_size).numpy()
    return resized_S_mel_db


# Funkcja do przewidywania komendy na podstawie pliku WAV
def predict_command_from_wav(wav_path):
    spectrogram = wav_to_mel_spectrogram(wav_path)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    predictions = model.predict(spectrogram)

    class_labels = sorted(['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes', 'backward'])
    predicted_class = np.argmax(predictions, axis=-1)[0]
    predicted_label = class_labels[predicted_class]
    return predicted_label


def test_all_wav_files(folder_path):
    total_files = 0
    correct_predictions = 0
    predictions = []
    actual_labels = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):

            file_path = os.path.join(folder_path, file_name)

            group = file_name.split('_')[0]  # Wszystko przed znakiem '_'

            # Przewidywanie komendy
            predicted_label = predict_command_from_wav(file_path)
            print(f"Plik: {file_name}, Grupa: {group}, Przewidywana komenda: {predicted_label}")

            # Wyodrębnianie grupy z nazwy pliku
            actual_label = file_name.split('_')[0]  # Wszystko przed znakiem '_'
            predicted_label = predict_command_from_wav(file_path)

            # Zliczanie poprawnych przewidywań
            if actual_label == predicted_label:
                correct_predictions += 1

            predictions.append(predicted_label)
            actual_labels.append(actual_label)
            total_files += 1

    accuracy = (correct_predictions / total_files) * 100
    print(f"Liczba przetworzonych plików: {total_files}")
    print(f"Poprawne przewidywania: {correct_predictions}")
    print(f"Dokładność: {accuracy:.2f}%")

    # Tworzenie wykresu
    plt.figure(figsize=(10, 5))
    counter_actual = Counter(actual_labels)
    counter_predicted = Counter(predictions)

    labels = sorted(counter_actual.keys() | counter_predicted.keys())
    actual_counts = [counter_actual.get(label, 0) for label in labels]
    predicted_counts = [counter_predicted.get(label, 0) for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, actual_counts, width, label='Rzeczywiste')
    plt.bar(x + width / 2, predicted_counts, width, label='Przewidywane')

    plt.xlabel('Komendy')
    plt.ylabel('Liczba')
    plt.title('Porównanie liczby rzeczywistych i przewidywanych etykiet')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return accuracy


# Wywołanie funkcji testującej
test_all_wav_files("Our_Records_trimmed")