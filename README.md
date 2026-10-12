# Cancer Cell Detection Pipeline

Binary cancer detection from histopathology tissue patches using the open-source **PatchCamelyon (PCam)** dataset and a fine-tuned **EfficientNet-B0** classifier.

## Setup

```bash
cd cancer_cell_detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Train the Model
Downloads PCam from Hugging Face automatically on first run (cached locally after).
```bash
cd cancer_cell_detection
python src/train.py
```
Best model saved to `models/best_model.pt`.

### 2. Run Inference on an Image
```bash
python src/inference.py --image path/to/patch.jpg
```

### 3. Evaluate on Test Set
```bash
python src/evaluate.py
```
Plots saved to `reports/`.

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501)

## Clinical Advisory Tiers

| Level | Cancer Probability | Advisory |
|---|---|---|
| 🚨 High | > 90% | Seek immediate medical evaluation |
| ⚠️ Borderline | 70–90% | Consult a pathologist to confirm |
| ✅ Low | < 70% | Low likelihood — verify with doctor |
| 💬 Uncertain | Model < 60% confident | Please consult a qualified physician |

> ⚠️ **Disclaimer**: This tool is for research and educational purposes only. It is not a certified medical device and does not replace professional diagnosis.

## Project Structure

```
cancer_cell_detection/
├── data/               # PCam data (auto-downloaded)
├── models/             # Saved checkpoints
├── reports/            # Evaluation plots
├── src/
│   ├── dataset.py      # PCam dataset + transforms
│   ├── model.py        # EfficientNet-B0 classifier
│   ├── train.py        # Training script
│   ├── evaluate.py     # Test-set evaluation + plots
│   └── inference.py    # Inference pipeline + advisory logic
├── app.py              # Streamlit dashboard
└── requirements.txt
```

## Data Source

- **Dataset**: PatchCamelyon (PCam) — [1aurent/PatchCamelyon on Hugging Face](https://huggingface.co/datasets/1aurent/PatchCamelyon)
- **License**: CC0 Public Domain
- **Size**: 327,680 × 96×96 px patches from histology slides
