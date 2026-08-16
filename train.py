import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import SatelliteDataset
from models.unet import UNet
from config import *

# -----------------------------
# Setup
# -----------------------------
os.makedirs("saved_models", exist_ok=True)
dataset = SatelliteDataset()
total_images = len(dataset)

print(f"Total Image Pairs Loaded: {total_images}")

# Initialize loaders
val_loader = None

if total_images < 10:
    print(f"\nPrototype Mode: Dataset has {total_images} pairs.")
    print("Training on all pairs directly without validation split.")
    train_loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, total_images), shuffle=True)
else:
    train_size = int(0.8 * total_images)
    val_size = total_images - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training Pairs: {train_size} | Validation Pairs: {val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Initialize Model
# -----------------------------
model = UNet().to(DEVICE)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_loss = float('inf')

# -----------------------------
# Training Loop
# -----------------------------
print("\nStarting Training Pipeline...\n")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_train_loss = 0.0

    for cloudy, clear in train_loader:
        cloudy = cloudy.to(DEVICE)
        clear = clear.to(DEVICE)

        optimizer.zero_grad()
        prediction = model(cloudy)
        loss = criterion(prediction, clear)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    
    # Validation / Save Logic
    if val_loader is not None:
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for cloudy, clear in val_loader:
                cloudy = cloudy.to(DEVICE)
                clear = clear.to(DEVICE)
                prediction = model(cloudy)
                val_loss = criterion(prediction, clear)
                running_val_loss += val_loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(" >> Best model weights saved!")
    else:
        # Small dataset prototype saving
        if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1 or avg_train_loss < best_loss:
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.6f} >> Model saved to {MODEL_PATH}")
        else:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.6f}")

print("\nTraining Complete!")
print(f"Model successfully saved at: {MODEL_PATH}")