import tensorflow as tf
import matplotlib.pyplot as plt

#ładowanie danych treningowych
def load_data(train_dir, val_dir, test_dir, image_size=(128, 128), batch_size=32):

    # Ładowanie danych treningowych
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    # Ładowanie danych walidacyjnych
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    # Ładowanie danych testowych
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    return train_dataset, val_dataset, test_dataset


def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        # Warstwa wejściowa z zdefiniowanym kształtem wejścia
        tf.keras.layers.Input(shape=input_shape),

        # Warstwa konwolucyjna 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Warstwa konwolucyjna 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Warstwa konwolucyjna 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Warstwa spłaszczająca
        tf.keras.layers.Flatten(),

        # Warstwa gęsta (Fully connected)
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Dropout dla regularizacji

        # Warstwa wyjściowa (softmax dla klasyfikacji wieloklasowej)
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model


