import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib as mpl
mpl.use('Agg')  # Non-interactive backend for server environments
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

from config import IMAGE_SIZE

# --------------------------------
# Image Transform
# --------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# --------------------------------
# Load Image
# --------------------------------
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    if not isinstance(image, torch.Tensor):
        raise TypeError("Image failed to convert to tensor.")
    return image

# --------------------------------
# Save Image
# --------------------------------
def save_image(tensor, output_path):
    image = transforms.ToPILImage()(tensor.cpu())
    image.save(output_path)

# --------------------------------
# Phase 2: Cloud Detection & Masking
# --------------------------------
def detect_clouds(tensor_image, threshold_lum=0.68, threshold_sat=0.28):
    np_img = tensor_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    np_img = np.asarray(np_img) 
    
    luminance = 0.299 * np_img[:, :, 0] + 0.587 * np_img[:, :, 1] + 0.114 * np_img[:, :, 2]
    
    max_rgb = np.max(np_img, axis=2)
    min_rgb = np.min(np_img, axis=2)
    delta = max_rgb - min_rgb
    saturation = np.zeros_like(max_rgb)
    nonzero_mask = max_rgb > 1e-5
    saturation[nonzero_mask] = delta[nonzero_mask] / max_rgb[nonzero_mask]
    
    is_cloud = (luminance > threshold_lum) & (saturation < threshold_sat)
    
    cloud_pixels = np.sum(is_cloud)
    total_pixels = is_cloud.size
    cloud_cover_pct = (float(cloud_pixels) / total_pixels) * 100.0
    
    cloud_mask = (is_cloud * 255).astype(np.uint8)
    
    return cloud_mask, round(cloud_cover_pct, 1), luminance, is_cloud

# --------------------------------
# Phase 3: Explainable Confidence Mapping
# --------------------------------
def generate_confidence_map(is_cloud, luminance):
    h, w = is_cloud.shape
    confidence = np.ones((h, w), dtype=np.float32)

    if np.any(is_cloud):
        raw_distances = distance_transform_edt(is_cloud)
        cloud_distances = np.asarray(raw_distances, dtype=np.float32)
        
        max_dist = float(np.max(cloud_distances))
        if max_dist <= 0:
            max_dist = 1.0
            
        norm_dist = cloud_distances / max_dist 

        cloud_conf = 0.85 - (0.45 * norm_dist) - (0.20 * luminance)
        cloud_conf = np.clip(cloud_conf, 0.15, 0.85)

        confidence[is_cloud] = cloud_conf[is_cloud]
    
    overall_confidence_score = round(float(np.mean(confidence) * 100.0), 1)

    colormap = mpl.colormaps['RdYlGn']
    heatmap_colored = colormap(confidence)[:, :, :3]
    heatmap_rgb = (heatmap_colored * 255).astype(np.uint8)

    return heatmap_rgb, overall_confidence_score

# --------------------------------
# Phase 4: Evaluation Metrics
# --------------------------------
def calculate_metrics(pred_tensor, target_tensor):
    """
    Calculates MAE, PSNR, and SSIM between prediction and ground truth.
    Tensors should be [C, H, W] in range [0, 1].
    """
    # Convert to numpy arrays in format [H, W, C] for scikit-image
    pred_np = pred_tensor.permute(1, 2, 0).cpu().numpy()
    target_np = target_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Scale to 0-255 for standard image metrics
    pred_img = (pred_np * 255).astype(np.uint8)
    target_img = (target_np * 255).astype(np.uint8)

    # 1. Mean Absolute Error (MAE)
    mae = np.mean(np.abs(pred_np - target_np))
    
    # 2. Peak Signal-to-Noise Ratio (PSNR)
    # Using data_range=255 because we scaled the images to uint8
    psnr = psnr_metric(target_img, pred_img, data_range=255)
    
    # 3. Structural Similarity Index (SSIM)
    # multichannel=True (deprecated in newer skimage, channel_axis=-1 is the standard)
    ssim = ssim_metric(target_img, pred_img, channel_axis=-1, data_range=255)
    
    return round(float(mae), 4), round(float(psnr), 2), round(float(ssim), 4)