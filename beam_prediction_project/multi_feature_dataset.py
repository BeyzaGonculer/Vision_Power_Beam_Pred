import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class MultiFeatureDataset(Dataset):
    def __init__(self, csv_path, camera_dir, power_dir, transform=None):
        import pandas as pd
        self.data = pd.read_csv(csv_path)
        self.camera_dir = camera_dir
        self.power_dir = power_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Görüntü yolunu oluştur
        image_path = os.path.join(self.camera_dir, os.path.basename(row['unit1_rgb_1']))
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Güç vektörünü oku
        power_path = os.path.join(self.power_dir, os.path.basename(row['unit1_pwr_1']))
        with open(power_path, 'r') as f:
            power_values = [float(val) for val in f.read().strip().split()]
        power_tensor = torch.tensor(power_values, dtype=torch.float32)

        # Beam index label
        label = int(row['beam_index_1'])

        return image, power_tensor, label
