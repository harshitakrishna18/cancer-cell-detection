import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from dataset import get_dataloaders
from model import CancerClassifier

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = "models"
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CKPT_DIR, "best_model.pt")

def train_model(epochs=5):
    print(f"Training on device: {DEVICE}")
    train_loader, val_loader, _ = get_dataloaders(batch_size=64)
    
    model = CancerClassifier(pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0.0
        
        # Training loop
        for imgs, labels in tqdm(train_loader, desc="Training"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, CKPT_PATH)
            print(f"✨ Saved new best model to {CKPT_PATH}")

if __name__ == "__main__":
    train_model()
