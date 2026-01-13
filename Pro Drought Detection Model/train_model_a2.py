# train_model_a2.py

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from cnn_lstm_models import build_model
from data_generator import SequenceDataGenerator, get_region_dirs

def train_model_a2():
    model_type = "A2"
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    all_dirs = get_region_dirs("data/sequences")
    random.shuffle(all_dirs)
    split = int(0.8 * len(all_dirs))
    train_dirs = all_dirs[:split]
    val_dirs = all_dirs[split:]

    train_gen = SequenceDataGenerator(train_dirs, batch_size=16)
    val_gen = SequenceDataGenerator(val_dirs, batch_size=16, shuffle=False)

    model = build_model(model_type)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(output_dir, f"best_model_{model_type}.h5"),
            monitor='val_loss', save_best_only=True
        )
    ]

    class_weights = {0: 1.0, 1: 5.0}

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title('AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_plot_{model_type}.png"))
    plt.close()

    print(f"✅ Model A2 trained and saved in {output_dir}")

if __name__ == "__main__":
    train_model_a2()
