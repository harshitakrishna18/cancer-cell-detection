import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

class PCamDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        
        # Ensure image is in RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def get_dataloaders(batch_size=64, max_test=None):
    """
    Downloads and prepares dataloaders for the PCam dataset.
    max_test (int): optionally limit the test set size for quicker evaluation.
    """
    print("Loading PCam dataset from Hugging Face...")
    hf_dataset = load_dataset("1aurent/PatchCamelyon")
    
    # Standard ImageNet normalization for EfficientNet
    transform_train = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_ds = PCamDataset(hf_dataset["train"], transform=transform_train)
    val_ds = PCamDataset(hf_dataset["valid"], transform=transform_test)
    
    test_data = hf_dataset["test"]
    if max_test is not None:
        test_data = test_data.select(range(min(max_test, len(test_data))))
    test_ds = PCamDataset(test_data, transform=transform_test)

    # Use persistent_workers=False and num_workers=0 to avoid multiprocessing issues in some environments
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
