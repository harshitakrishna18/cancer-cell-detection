import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from dataset import get_dataloaders
from model import load_model

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "models/best_model.pt"
REPORT_DIR = "reports"

def plot_styled_cm():
    if not os.path.exists(CKPT_PATH):
        print("Trained model not found. Please train first.")
        return

    print("Loading data and model...")
    _, _, test_loader = get_dataloaders(batch_size=64, max_test=500)
    model = load_model(CKPT_PATH, device=DEVICE)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    # Custom styling
    plt.figure(figsize=(8, 6), dpi=150)
    sns.set_theme(style="white")
    
    # Heatmap with custom colors (matching the pro architecture colors)
    cmap = sns.light_palette("#2563EB", as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, 
                xticklabels=["Normal", "Cancer"], 
                yticklabels=["Normal", "Cancer"],
                annot_kws={"size": 16, "weight": "bold"},
                cbar=False)
    
    plt.xlabel("Predicted Label", fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel("True Label", fontsize=12, fontweight='bold', labelpad=10)
    plt.title("Confusion Matrix: Cancer Cell Detection", fontsize=16, fontweight='bold', pad=20)
    
    # Layout adjustment
    plt.tight_layout()
    
    save_path = os.path.join(REPORT_DIR, "confusion_matrix_styled.png")
    os.makedirs(REPORT_DIR, exist_ok=True)
    plt.savefig(save_path)
    print(f"✅ Confusion Matrix plotted and saved to: {save_path}")
    
    # Display the image location
    return save_path

if __name__ == "__main__":
    plot_styled_cm()
