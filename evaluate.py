import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import SatelliteDataset
from models.unet import UNet
from config import MODEL_PATH, DEVICE
from utils import calculate_metrics

def evaluate_model():
    print("--- AkashaLens Model Evaluation ---")
    
    os.makedirs("output/eval", exist_ok=True)

    # 1. Load Model
    model = UNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("Model loaded successfully.\n")
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_PATH}. Cannot evaluate.")
        return

    # 2. Load Dataset
    try:
        dataset = SatelliteDataset()
    except Exception as e:
        print(f"Dataset error: {e}")
        return

    total_images = len(dataset)
    if total_images == 0:
        print("Dataset is empty.")
        return

    total_mae, total_psnr, total_ssim = 0.0, 0.0, 0.0

    # 3. Evaluation Loop
    with torch.no_grad():
        for i in range(total_images):
            cloudy_tensor, clear_tensor = dataset[i]
            
            # Add batch dimension and move to device
            input_tensor = cloudy_tensor.unsqueeze(0).to(DEVICE)
            target_tensor = clear_tensor.unsqueeze(0).to(DEVICE)
            
            # Predict
            output_tensor = model(input_tensor)
            
            # Squeeze batch dimension for metric calculation
            pred = output_tensor.squeeze(0).cpu()
            target = target_tensor.squeeze(0).cpu()
            cloudy_vis = cloudy_tensor.cpu()
            
            # Compute Metrics
            mae, psnr, ssim = calculate_metrics(pred, target)
            total_mae += mae
            total_psnr += psnr
            total_ssim += ssim
            
            print(f"Image [{i+1}/{total_images}] | MAE: {mae:.4f} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

            # 4. Generate Visual Comparison
            c_np = cloudy_vis.permute(1, 2, 0).numpy()
            t_np = target.permute(1, 2, 0).numpy()
            p_np = pred.permute(1, 2, 0).numpy()
            
            # Difference map (Absolute error scaled for visibility)
            diff_np = np.abs(p_np - t_np)
            # Enhance difference contrast slightly for viewing
            diff_np = np.clip(diff_np * 2.0, 0, 1)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(c_np)
            axes[0].set_title("Cloudy Input")
            axes[0].axis('off')
            
            axes[1].imshow(t_np)
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')
            
            axes[2].imshow(p_np)
            axes[2].set_title(f"Reconstructed\nPSNR: {psnr:.1f} | SSIM: {ssim:.2f}")
            axes[2].axis('off')
            
            axes[3].imshow(diff_np)
            axes[3].set_title("Difference Map")
            axes[3].axis('off')
            
            plt.tight_layout()
            save_path = f"output/eval/comparison_{i+1}.png"
            plt.savefig(save_path, dpi=150)
            plt.close()

    # 5. Summary
    print("\n--- Final Average Metrics ---")
    print(f"Average MAE  : {total_mae/total_images:.4f}")
    print(f"Average PSNR : {total_psnr/total_images:.2f} dB")
    print(f"Average SSIM : {total_ssim/total_images:.4f}")
    print(f"\nVisual comparisons saved to 'output/eval/'")

if __name__ == "__main__":
    evaluate_model()