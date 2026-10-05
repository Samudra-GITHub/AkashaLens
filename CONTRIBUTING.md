# Contributing to AkashaLens

Thanks for considering a contribution to a research-oriented computer-vision project.

## Getting set up

```bash
git clone https://github.com/Samudra-GITHub/AkashaLens.git
cd AkashaLens
pip install -r requirements.txt
```

Dataset pairs go under `dataset/clear/` and `dataset/cloudy/` — see [README](./README.md#data-preparation).

## Before opening a PR

```bash
python -m tests.test_load_image
python -m tests.test_unet
python -m tests.test_dataset
```

If you change training or model config (`config.py`), include before/after SSIM/PSNR numbers from `evaluate.py` in the PR description — these are the metrics the project is actually judged on.

## Scope

- Model/architecture changes belong in `models/`.
- Dataset or ingestion changes belong in `dataset.py` / `scripts/`.
- Demo app changes belong in `app.py`, `static/`, `templates/`.

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/`.
