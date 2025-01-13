import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import glob

def process_and_save_mel_spectrograms(main_folder, output_folder, n_mels=128, image_size=(128, 128)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith('.wav'):
                wav_file = os.path.join(root, file)

                output_file = os.path.join(output_folder, root[len(main_folder):].strip(os.path.sep), file.replace(".wav", ".png"))

                print(f'Wysyłam plik: {wav_file} do: {output_file}')
                try:
                    y, sr = librosa.load(wav_file, sr=22050)

                    # Wyliczenie mel-spektrogramu
                    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                    S_mel_db = librosa.amplitude_to_db(S_mel, ref=np.max)

                    # Skalowanie
                    S_mel_db_normalized = (S_mel_db - np.mean(S_mel_db)) / np.std(S_mel_db)
                    S_mel_db_normalized = np.clip(S_mel_db_normalized, -2, 2)
                    S_mel_db_normalized = (S_mel_db_normalized + 2) / 4  # Skaluje wartości do zakresu [0, 1]

                    # Przekształcenie dla (na pozniej) CNN
                    resized_S_mel_db = plt.cm.viridis(S_mel_db_normalized)  # Skalowanie barwne
                    resized_S_mel_db = resized_S_mel_db[:, :, :3]  # Zmiana wymiarów (3 kanały kolorów)

                    #mel-spektrogram do PNG
                    if not os.path.exists(os.path.dirname(output_file)):
                        os.makedirs(os.path.dirname(output_file))
                    plt.imsave(output_file, resized_S_mel_db)
                    plt.close()

                except Exception as e:
                    print(f"Błąd przy przetwarzaniu pliku {wav_file}: {e}")




# Iteruje tylko po wybranych kategoriach
def process_wav_files_categories(main_folder):
    categories = sorted(['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes', 'backward'])

    selected_files = []

    for category in categories:
        folder_path = os.path.join(main_folder, category)
        if os.path.isdir(folder_path):
            files = glob.glob(os.path.join(folder_path, '*.wav'))
            if files:
                first_file = files[0]
                selected_files.append((category, first_file))
            else:
                print(f"Folder '{category}' pust/brak .wav!")
        else:
            print(f"Folder '{category}' nie istnieje")

    for category, file_path in selected_files:
        print(f"Przetwarzanie pliku: {file_path} z grupy {category}")
        try:
            # Próba wczytania pliku
            y, sr = librosa.load(file_path, sr=None)
            print(f"Plik {file_path} załadowany ")

            # Spektrogram
            fft = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(fft), ref=np.max)

            # Wyświetlenie spektrogramu
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_db, x_axis='time', y_axis='hz', sr=sr)
            plt.title(f'Spektrogram dla {os.path.basename(file_path)} z grupy: {category}')
            plt.colorbar(format='%+2.0f dB')
            plt.show()

            # Mel spektogram
            S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_mel_db = librosa.amplitude_to_db(S_mel, ref=np.max)

            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_mel_db, x_axis='time', y_axis='mel', sr=sr)
            plt.title(f'MEL Spektrogram dla {os.path.basename(file_path)} z grupy: {category}')
            plt.colorbar(format='%+2.0f dB')
            plt.show()

        except Exception as e:
            print(f"Błąd przy przetwarzaniu pliku {file_path}: {e}")
