import torch

# ==========================
# Dataset Configuration
# ==========================

CLOUDY_DIR = "dataset/cloudy"
CLEAR_DIR = "dataset/clear"

# ==========================
# Image Configuration
# ==========================

IMAGE_SIZE = 256
IMAGE_CHANNELS = 3

# ==========================
# Training Configuration
# ==========================

BATCH_SIZE = 2
NUM_EPOCHS = 1000
LEARNING_RATE = 0.0002

# ==========================
# Model Configuration
# ==========================

MODEL_PATH = "saved_models/akashalens_model.pth"

# ==========================
# Device Configuration
# ==========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {DEVICE}")