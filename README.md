# AkashaLens 🛰️

Revealing Earth Beyond the Clouds. AkashaLens is a satellite-image reconstruction system designed to remove and reconstruct cloud-occluded regions in optical satellite imagery using a PyTorch-based U-Net architecture.

## Overview
Optical satellite sensors (like LISS-IV) frequently suffer from cloud occlusion, resulting in missing ground data. AkashaLens provides an end-to-end AI pipeline that detects cloud cover, calculates a spatial confidence heatmap, and generates a reconstructed image of the underlying terrain.

This prototype features a Flask REST API and a responsive, asynchronous frontend dashboard for real-time inference and evaluation.

## Features
*   **Explainable Cloud Detection:** Uses physics-grounded luminance and saturation thresholding in the HSV space to isolate clouds and calculate precise cloud cover percentages.
*   **Confidence Heatmap:** Generates a spatial confidence map based on cloud optical density and distance-decay from clear terrain boundaries.
*   **AI Reconstruction:** Utilizes a Convolutional U-Net (Encoder-Bottleneck-Decoder with skip connections) to predict clear terrain features.
*   **Live Evaluation Metrics:** Dual-upload system allows users to submit a Ground Truth image alongside the cloudy input to calculate real-time **MAE**, **PSNR**, and **SSIM** scores directly in the browser.
*   **Asynchronous Processing:** Powered by the Fetch API and `FormData` for seamless UI updates without page reloads.

## Tech Stack
*   **Backend:** Python, Flask, Werkzeug
*   **AI / Machine Learning:** PyTorch, Torchvision
*   **Image Processing:** NumPy, SciPy, scikit-image, Pillow, Matplotlib
*   **Frontend:** HTML5, Vanilla JavaScript, CSS3

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/AkashaLens.git](https://github.com/yourusername/AkashaLens.git)
   cd AkashaLens