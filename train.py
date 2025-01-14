import tensorflow as tf
import os

# Funkcja do ładowania danych treningowych, walidacyjnych i testowych
def load_data(train_dir, val_dir, test_dir, image_size=(128, 128), batch_size=128):

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


# Funkcja do budowania modelu
def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        # Warstwa konwolucyjna 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Warstwa konwolucyjna 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Warstwa konwolucyjna 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Warstwa konwolucyjna 4
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Warstwy gęste
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


# Funkcja do trenowania modelu
def train_model(train_dir, val_dir, test_dir, image_size=(128, 128), batch_size=32, epochs=10):
    # Ładowanie danych
    train_dataset, val_dataset, test_dataset = load_data(train_dir, val_dir, test_dir, image_size, batch_size)

    # Budowanie modelu
    model = build_model(input_shape=(128, 128, 3), num_classes=len(train_dataset.class_names))

    # Kompilowanie modelu
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Dodanie EarlyStopping i ModelCheckpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5",
        monitor="val_loss",
        save_best_only=True
    )

    # Trening modelu
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Ocena modelu na danych testowych
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')

    return model, history
