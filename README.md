# AkashaLens

**AI-powered satellite cloud reconstruction.**

A deep-learning pipeline that reconstructs cloud-occluded satellite imagery, turning partially obscured Sentinel scenes into clear, usable ground imagery.

<br/>

<img src="./assets/hero.png" width="100%" alt="AkashaLens hero" />

<br/>

## Before / After

<table width="100%">
<tr>
<td width="50%"><img src="./assets/screenshots/before.png" width="100%" alt="Cloud-occluded input" /><br/><sub align="center">Cloud-occluded input</sub></td>
<td width="50%"><img src="./assets/screenshots/after.png" width="100%" alt="Reconstructed output" /><br/><sub align="center">Reconstructed output</sub></td>
</tr>
</table>

<br/>

## Problem Statement

Cloud cover is one of the largest practical obstacles in optical Earth observation — a significant fraction of any given satellite pass over land is unusable because of it. AkashaLens treats cloud removal as an image-reconstruction problem: given a cloudy scene, predict the clear scene underneath.

<br/>

## AI Pipeline

```
Sentinel imagery ──▶ dataset pairing (cloudy / clear) ──▶ U-Net ──▶ reconstructed output
                                                              │
                                                              ▼
                                                  SSIM / PSNR evaluation
```

- **Ingestion** — `scripts/get_copernicus.py` pulls source imagery from the Copernicus program
- **Dataset** — paired `cloudy/` and `clear/` scenes, loaded via a custom PyTorch `Dataset` (`dataset.py`)
- **Model** — a U-Net (`models/unet.py`) trained end-to-end on 256×256 image pairs
- **Evaluation** — reconstruction quality measured with SSIM and PSNR (`scikit-image`)
- **Serving** — a Flask app (`app.py`) exposes the trained model for interactive prediction

<br/>

## Dataset

Training pairs live under `dataset/clear/` and `dataset/cloudy/` — matched clear and cloud-occluded versions of the same scene, resized to 256×256 before training.

<br/>

## Sentinel Imagery

Source imagery is pulled from the **Copernicus** program (`scripts/get_copernicus.py`), which distributes Sentinel satellite data.

<br/>

## Model Architecture

A U-Net encoder-decoder, trained with a configurable batch size and learning rate (`config.py`), on `IMAGE_SIZE = 256` inputs. Trained weights are saved to `saved_models/akashalens_model.pth` and loaded by both `predict.py` and the Flask app for inference.

<br/>

## Folder Structure

```
akashalens/
├── app.py               # Flask serving app
├── config.py             # Dataset paths, training + model config
├── dataset.py             # PyTorch Dataset for cloudy/clear pairs
├── dataset/
│   ├── clear/
│   └── cloudy/
├── models/
│   └── unet.py            # U-Net architecture
├── scripts/
│   └── get_copernicus.py  # Sentinel imagery ingestion
├── static/                # Flask app assets
├── templates/              # Flask app templates
├── train.py
├── evaluate.py
├── predict.py
├── test.py / test_unet.py / test_dataset.py
└── utils.py                # Image loading, SSIM/PSNR metrics
```

<br/>

## Installation

```bash
git clone https://github.com/Samudra-GITHub/AkashaLens.git
cd AkashaLens
pip install -r requirements.txt

# train
python train.py

# evaluate
python evaluate.py

# run the demo app
python app.py
```

<br/>

## Tech Stack

`PyTorch` · `torchvision` · `Flask` · `Pillow` · `NumPy` · `SciPy` · `scikit-image` · `Matplotlib`

<br/>

## Results

Reconstruction quality is tracked with SSIM (structural similarity) and PSNR (peak signal-to-noise ratio) against the held-out clear ground truth — see `evaluate.py` for the current evaluation pass.

<br/>

## ISRO Hackathon Journey

AkashaLens was built for **ISRO Hackathon 2026**, framed around a real constraint in Earth observation: usable imagery is often blocked by cloud cover exactly when it's needed most.

<br/>

## Future Improvements

- [ ] Expand the training set beyond the current clear/cloudy pairs
- [ ] Push reconstruction accuracy toward ISRO-grade requirements
- [ ] Evaluate alternative architectures (e.g. attention-based U-Net variants) against the current baseline
- [ ] Add batch inference for full scene tiles, not just fixed 256×256 crops

<br/>

## License

MIT — see [LICENSE](./LICENSE).

<br/>

<sub>Part of the Samudra OS product ecosystem. See the [profile](https://github.com/Samudra-GITHub) for the full lineup.</sub>
