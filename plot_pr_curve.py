import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from dataset import get_dataloaders
from model import load_model

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "models/best_model.pt"
REPORT_DIR = "reports"

def plot_pr_curve():
    if not os.path.exists(CKPT_PATH):
        print("Trained model not found. Please train first.")
        return

    print("Loading data and model...")
    _, _, test_loader = get_dataloaders(batch_size=64, max_test=500)
    model = load_model(CKPT_PATH, device=DEVICE)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    
    # Custom styling
    plt.figure(figsize=(8, 6), dpi=150)
    sns.set_theme(style="whitegrid")
    
    plt.plot(recall, precision, color='#2563EB', lw=2, label=f'PR curve (area = {pr_auc:0.3f})')
    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("Precision-Recall Curve: Cancer Cell Detection", fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower left", fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Layout adjustment
    plt.tight_layout()
    
    save_path = os.path.join(REPORT_DIR, "pr_curve.png")
    os.makedirs(REPORT_DIR, exist_ok=True)
    plt.savefig(save_path)
    print(f"✅ PR Curve plotted and saved to: {save_path}")
    
    return save_path

if __name__ == "__main__":
    plot_pr_curve()
