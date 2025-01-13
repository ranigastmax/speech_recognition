import os
import shutil
import random


def split_dataset(input_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Tworzenie folderów docelowych
    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "val")
    test_folder = os.path.join(output_folder, "test")

    for folder in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Przechodzenie przez każdą kategorię w danych wejściowych
    for category in os.listdir(input_folder):
        category_path = os.path.join(input_folder, category)
        if not os.path.isdir(category_path):
            continue

        # Pliki w bieżącej kategorii
        files = [f for f in os.listdir(category_path) if f.endswith('.png')]
        random.shuffle(files)  # Mieszanie plików dla losowego podziału

        # Obliczanie liczby plików na zbiór
        total_files = len(files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        test_end = val_end + int(total_files * test_ratio)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:test_end]

        # Przenoszenie plików do odpowiednich folderów
        for file_group, target_folder in zip([train_files, val_files, test_files],
                                             [train_folder, val_folder, test_folder]):
            category_target_folder = os.path.join(target_folder, category)
            if not os.path.exists(category_target_folder):
                os.makedirs(category_target_folder)

            for file in file_group:
                src = os.path.join(category_path, file)
                dest = os.path.join(category_target_folder, file)
                shutil.copy(src, dest)  # Kopiowanie plików
                print(f"Skopiowano {src} -> {dest}")
