import torch
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add root to path so we can import the plotting scripts and src modules
sys.path.insert(0, os.getcwd())
from src.dataset import get_dataloaders
from src.model import load_model
from plot_cm import plot_styled_cm
from plot_pr_curve import plot_pr_curve

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "models/best_model.pt"

def evaluate_model():
    if not os.path.exists(CKPT_PATH):
        print("Trained model not found. Please train first.")
        return

    print("Evaluating model on test set...")
    _, _, test_loader = get_dataloaders(batch_size=64)
    model = load_model(CKPT_PATH, device=DEVICE)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print("\n--- Test Set Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("------------------------")
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    plot_styled_cm()
    plot_pr_curve()
    
    print("\nEvaluation complete! Plots saved to reports/")

if __name__ == "__main__":
    evaluate_model()
