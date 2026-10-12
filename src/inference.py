import torch
import argparse
from PIL import Image
from torchvision import transforms
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from model import load_model

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "models/best_model.pt"

def get_advisory_tier(probability):
    prob_percent = probability * 100
    if prob_percent > 90:
        return "🚨 High", "Seek immediate medical evaluation"
    elif prob_percent >= 70:
        return "⚠️ Borderline", "Consult a pathologist to confirm"
    elif prob_percent >= 40 and prob_percent <= 60:
        return "💬 Uncertain", "Please consult a qualified physician"
    else:
        return "✅ Low", "Low likelihood — verify with doctor"

def predict_image(image_path):
    if not os.path.exists(CKPT_PATH):
        print("Trained model not found. Please train first.")
        return None, None, None

    model = load_model(CKPT_PATH, device=DEVICE)
    
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    tensor_img = transform(image).unsqueeze(0).to(DEVICE)
    
    prob = model.predict_proba(tensor_img).item()
    tier, advisory = get_advisory_tier(prob)
    
    print(f"\n--- Inference Results ---")
    print(f"Image: {image_path}")
    print(f"Cancer Probability: {prob * 100:.2f}%")
    print(f"Advisory Tier: {tier}")
    print(f"Recommendation: {advisory}\n")
    
    return prob, tier, advisory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cancer Cell Detection Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the patch image")
    args = parser.parse_args()
    
    predict_image(args.image)
