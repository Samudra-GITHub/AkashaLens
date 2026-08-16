import os
import sys
import torch
from PIL import Image
from torchvision import transforms

from config import *
from models.unet import UNet

if len(sys.argv) < 2:
    print("Usage: python predict.py <path_to_cloudy_image>")
    sys.exit(1)

input_image_path = sys.argv[1]

if not os.path.exists(input_image_path):
    print(f"Error: Image {input_image_path} does not exist.")
    sys.exit(1)

# -----------------------------
# Create output folder
# -----------------------------
os.makedirs("output", exist_ok=True)

# -----------------------------
# Load Model
# -----------------------------
model = UNet().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}. Please train the model first.")
    sys.exit(1)

# -----------------------------
# Image Transform matches Flask/Train
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

print(f"Using image: {input_image_path}")

image = Image.open(input_image_path).convert("RGB")
input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():
    output = model(input_tensor)

output = output.squeeze(0).cpu()

# -----------------------------
# Save Output
# -----------------------------
output_image = transforms.ToPILImage()(output)
filename = os.path.basename(input_image_path)
output_path = os.path.join("output", f"reconstructed_{filename}")

output_image.save(output_path)

print("\nPrediction Complete!")
print(f"Saved to: {output_path}")