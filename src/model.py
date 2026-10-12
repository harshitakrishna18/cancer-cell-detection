"""
EfficientNet-B0 binary classifier for cancer patch detection.
"""
import torch
import torch.nn as nn
import timm


class CancerClassifier(nn.Module):
    """EfficientNet-B0 with a single sigmoid output for binary classification."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,          # Remove classifier head
            global_pool="avg",
        )
        in_features = self.backbone.num_features  # 1280 for B0
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits.squeeze(1)   # (B,)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probability of being cancerous (0-1)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def load_model(checkpoint_path: str, device: str = "cpu") -> CancerClassifier:
    """Loads a trained model from a checkpoint file."""
    model = CancerClassifier(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model
