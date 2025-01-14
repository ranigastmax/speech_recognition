import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Ładowanie modelu
model = tf.keras.models.load_model('model_b1.h5')


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

    print(f'Przewidywana komenda: {predicted_label}')
    return predicted_label


# Testowanie na pojedynczym pliku WAV
predict_command_from_wav("off2.wav")
