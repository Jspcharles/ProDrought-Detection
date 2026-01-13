# data_generator.py

import os
import numpy as np
import random
from tensorflow.keras.utils import Sequence

class SequenceDataGenerator(Sequence):
    def __init__(self, region_dirs, batch_size=16, shuffle=True):
        self.sample_paths = []
        for region_dir in region_dirs:
            for file in os.listdir(region_dir):
                if file.endswith(".npy") and not file.endswith("_label.npy"):
                    label_file = file.replace(".npy", "_label.npy")
                    label_path = os.path.join(region_dir, label_file)
                    if os.path.exists(label_path):
                        self.sample_paths.append((os.path.join(region_dir, file), label_path))

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.sample_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.sample_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X, batch_Y = [], []

        for x_path, y_path in batch_paths:
            try:
                x = np.load(x_path)
                y = np.load(y_path)
                if x.shape == (10, 32, 32, 10) and not np.isnan(y):
                    batch_X.append(x)
                    batch_Y.append(y)
            except:
                continue

        return np.array(batch_X, dtype=np.float32), np.array(batch_Y, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.sample_paths)

def get_region_dirs(base_path="data/sequences"):
    return [os.path.join(base_path, region) for region in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, region))]
