import os
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms

from models.unet import UNet
from config import MODEL_PATH, DEVICE, IMAGE_SIZE
from utils import detect_clouds, generate_confidence_map, calculate_metrics

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load Model
model = UNet().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"WARNING: Model not found at {MODEL_PATH}.")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/output/<filename>")
def send_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if not file or not file.filename:
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(str(file.filename))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Preprocess input image
        image = Image.open(filepath).convert("RGB")
        img_tensor = transform(image)
        if not isinstance(img_tensor, torch.Tensor):
            return jsonify({"error": "Image preprocessing failed."}), 500
            
        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        # 1. Cloud Masking & Percentage
        cloud_mask_np, cloud_cover_pct, luminance, is_cloud = detect_clouds(img_tensor)
        mask_filename = f"mask_{filename}"
        mask_filepath = os.path.join(app.config['OUTPUT_FOLDER'], mask_filename)
        Image.fromarray(cloud_mask_np).save(mask_filepath)

        # 2. Confidence Heatmap
        heatmap_rgb, confidence_score = generate_confidence_map(is_cloud, luminance)
        heatmap_filename = f"heatmap_{filename}"
        heatmap_filepath = os.path.join(app.config['OUTPUT_FOLDER'], heatmap_filename)
        Image.fromarray(heatmap_rgb).save(heatmap_filepath)

        # 3. Model Inference
        with torch.no_grad():
            output = model(input_tensor)
        
        output = output.squeeze(0).cpu()
        output_image = transforms.ToPILImage()(output)
        
        out_filename = f"reconstructed_{filename}"
        out_filepath = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)
        output_image.save(out_filepath)

        # 4. Optional Ground Truth Evaluation
        eval_type = "Inference-only prediction"
        metrics = {
            "MAE": "N/A",
            "PSNR": "N/A",
            "SSIM": "N/A"
        }

        if 'ground_truth' in request.files:
            gt_file = request.files['ground_truth']
            if gt_file and gt_file.filename != '':
                gt_filename = secure_filename(str(gt_file.filename))
                gt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"gt_{gt_filename}")
                gt_file.save(gt_filepath)
                
                gt_image = Image.open(gt_filepath).convert("RGB")
                gt_tensor = transform(gt_image)
                if isinstance(gt_tensor, torch.Tensor):
                    mae, psnr, ssim = calculate_metrics(output, gt_tensor)
                    eval_type = "Ground-truth Verified"
                    metrics = {
                        "MAE": f"{mae}",
                        "PSNR": f"{psnr} dB",
                        "SSIM": f"{ssim}"
                    }

        return jsonify({
            "status": "success",
            "original_image": filename,
            "reconstructed_image_url": f"/output/{out_filename}",
            "cloud_mask_url": f"/output/{mask_filename}",
            "heatmap_url": f"/output/{heatmap_filename}",
            "cloud_cover": f"{cloud_cover_pct}%",
            "confidence_score": f"{confidence_score}%",
            "evaluation_type": eval_type,
            "metrics": metrics
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Unknown processing error"}), 500

if __name__ == "__main__":
    app.run(debug=True)