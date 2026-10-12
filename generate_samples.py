import os
import sys
from PIL import Image
import numpy as np
from datasets import load_dataset

def main():
    print("Loading PatchCamelyon dataset...")
    hf = load_dataset("1aurent/PatchCamelyon")
    val_data = hf["valid"]
    
    os.makedirs("data/samples", exist_ok=True)
    
    # Save a positive (cancer) and a negative (no cancer) example
    saved_pos = False
    saved_neg = False
    
    for i in range(100):
        item = val_data[i]
        label = item["label"]
        img = item["image"]
        
        if not saved_pos and label == 1:
            img.save(f"data/samples/sample_cancer_{i}.jpg")
            saved_pos = True
            print(f"Saved cancer sample to data/samples/sample_cancer_{i}.jpg")
            
        if not saved_neg and label == 0:
            img.save(f"data/samples/sample_normal_{i}.jpg")
            saved_neg = True
            print(f"Saved normal sample to data/samples/sample_normal_{i}.jpg")
            
        if saved_pos and saved_neg:
            break

if __name__ == "__main__":
    main()
