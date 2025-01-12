import os
import glob

def limit_files_in_subfolders(main_folder, max_files):
    for label in os.listdir(main_folder):
        label_path = os.path.join(main_folder, label)
        if os.path.isdir(label_path):

            wav_files = glob.glob(os.path.join(label_path, "*.wav"))

            if len(wav_files) > max_files:

                wav_files.sort()

                files_to_remove = wav_files[max_files:]
                for file_path in files_to_remove:
                    try:
                        os.remove(file_path)
                        print(f"Usunięto plik: {file_path}")
                    except Exception as e:
                        print(f"Błąd  usuwania pliku {file_path}: {e}")
            else:
                print(f"Folder {label}mniej niż {max_files} plików.")