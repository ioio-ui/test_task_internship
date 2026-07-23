# Task 2 — Sentinel-2 Image Matching

Fine-tuning **LoFTR** (Detector-Free Local Feature Matching) to match Sentinel-2 satellite images taken of the same location in different seasons.

## Solution

**Dataset.** Sentinel-2 tile pairs covering the same geographic area across different seasons were downloaded and co-registered. Each pair was exported as grayscale PNGs. During training, synthetic pairs are generated on-the-fly via random homographies (rotation, scale, shift) applied to image patches, providing dense ground-truth correspondences without manual annotation.

**Model.** LoFTR from `kornia`, pre-trained on indoor scenes, fine-tuned on 256×256 satellite patches. LoFTR works without explicit keypoint detection — it produces dense matches directly from feature maps, making it robust to the appearance changes caused by seasonal variation. Large images are processed by splitting into 512×64-overlap tiles and merging results. RANSAC filters geometrically inconsistent matches.

## Project structure

```
task2/
├── dataset_pairs/
│   ├── pairs/                   # PNG image pairs
│   └── pairs_metadata.json      # pair index
├── dataset-creation.ipynb       # dataset preparation process
├── demo.ipynb                   # keypoints and matches visualisation
├── train.py                     # fine-tuning script
├── inference.py                 # inference script (single pair or batch)
├── postprocess_ransac.py        # RANSAC geometric filter
├── checkpoints/
│   └── loftr_satellite.pth      # fine-tuned weights
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset & model weights

Dataset (Sentinel-2 tile pairs) and fine-tuned weights are available on Google Drive. Place the dataset as `dataset_pairs/` and the weights as `checkpoints/loftr_satellite.pth`.

## Usage

**Train:**
```bash
python train.py --data-dir dataset_pairs/pairs --metadata dataset_pairs/pairs_metadata.json
```

**Inference — single pair:**
```bash
python inference.py --img-a dataset_pairs/pairs/pair_000_a.png --img-b dataset_pairs/pairs/pair_000_b.png --weights checkpoints/loftr_satellite.pth --output results/matches.png
```

**Inference — batch:**
```bash
python inference.py --batch --data-dir dataset_pairs/pairs --metadata dataset_pairs/pairs_metadata.json --weights checkpoints/loftr_satellite.pth --output-dir results/
```

**Demo notebook:**
```bash
jupyter lab demo.ipynb
```
