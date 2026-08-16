import os
from torch.utils.data import Dataset
from config import CLOUDY_DIR, CLEAR_DIR
from utils import load_image

class SatelliteDataset(Dataset):
    def __init__(self):
        if not os.path.exists(CLOUDY_DIR) or not os.path.exists(CLEAR_DIR):
            raise FileNotFoundError(f"Ensure directories exist: '{CLOUDY_DIR}' and '{CLEAR_DIR}'")

        self.cloudy_images = sorted(os.listdir(CLOUDY_DIR))
        self.clear_images = sorted(os.listdir(CLEAR_DIR))

        if len(self.cloudy_images) == 0 or len(self.clear_images) == 0:
            raise ValueError("Dataset folders must contain at least 1 image.")

        # Match by available pairs
        self.pair_count = min(len(self.cloudy_images), len(self.clear_images))

    def __len__(self):
        return self.pair_count

    def __getitem__(self, index):
        cloudy_path = os.path.join(CLOUDY_DIR, self.cloudy_images[index])
        clear_path = os.path.join(CLEAR_DIR, self.clear_images[index])

        cloudy_image = load_image(cloudy_path)
        clear_image = load_image(clear_path)

        return cloudy_image, clear_image