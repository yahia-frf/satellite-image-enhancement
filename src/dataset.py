# src/dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class EuroSATDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the file path (e.g., 'AnnualCrop/AnnualCrop_1275.jpg')
        filename = self.data.iloc[idx]["Filename"]
        label = int(self.data.iloc[idx]["Label"])

        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
