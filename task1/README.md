# Task 1 — Mountain NER

Fine-tuning `bert-base-cased` to detect mountain names in free text using BIO tagging.

## Solution

**Dataset.** A synthetic dataset of 581 labeled sentences was generated from a gazetteer of 43 real mountains. Sentences were built from ~30 templates filled with mountain names, heights, and varied verbs/adjectives. To make the model robust, 100 hard-negative sentences were added using rivers, lakes, cities, and people names — all tagged `O`. Labels follow BIO notation: `B-MOUNTAIN`, `I-MOUNTAIN`, `O`. Split: 80/10/10 → 464 train / 58 val / 59 test.

**Model.** `bert-base-cased` with a linear token-classification head. The cased variant is important since mountain names are always capitalised. Fine-tuned for 5 epochs with AdamW (lr=2e-5, weight decay=0.01), best checkpoint selected by entity-level F1 on validation.

**Results.** F1 = 1.00 on the held-out test set (precision=1.00, recall=1.00).

## Project structure

```
task1/
├── dataset/
│   ├── train.jsonl          # 464 labelled sentences
│   ├── val.jsonl            # 58 sentences
│   ├── test.jsonl           # 59 sentences
│   └── dataset_conll.txt    # full dataset in CoNLL format
├── dataset_creation.ipynb   # dataset generation process
├── train.py                 # fine-tuning script
├── inference.py             # inference script
├── app.py                   # Gradio web interface
├── demo.ipynb               # end-to-end demo with evaluation
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Model weights

Trained weights are available on Google Drive. Download and place into `model_output/`.

## Usage

**Train:**
```bash
python train.py
```

**Inference:**
```bash
python inference.py --text "We climbed Mount Everest and K2 last summer."
python inference.py --interactive
```

**Web app:**
```bash
python app.py
```

**Demo notebook:**
```bash
jupyter lab demo.ipynb
```
