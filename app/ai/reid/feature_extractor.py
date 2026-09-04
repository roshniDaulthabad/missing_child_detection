import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.config import Config

class ReIDBackbone(nn.Module):
    """Lightweight CNN for Person Re-Identification appearance descriptor extraction on CPU."""
    def __init__(self, feature_dim: int = 256):
        super(ReIDBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, feature_dim, bias=False)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.bn.bias.requires_grad = False

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.bn(self.fc(x))
        return F.normalize(x, p=2, dim=1)


class PersonReIDExtractor:
    """Extracts 256-D normalized appearance embeddings from pedestrian crops on CPU."""

    def __init__(self, model_path: str = None):
        self.device = torch.device("cpu")
        self.model = ReIDBackbone(feature_dim=256)
        
        # Load weights or initialize deterministically
        torch.manual_seed(42)
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.model.eval()
        self.model.to(self.device)

    def extract_descriptor(self, person_bgr: np.ndarray) -> np.ndarray:
        """Extracts 256-D normalized appearance vector from person image (H x W x 3)."""
        if person_bgr is None or person_bgr.size == 0:
            return np.zeros((256,), dtype=np.float32)

        # Re-ID standard input: 256 height x 128 width
        resized = cv2.resize(person_bgr, (128, 256), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1)
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) * 255.0
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) * 255.0
        tensor = (tensor - mean) / std

        with torch.no_grad():
            feat = self.model(tensor.unsqueeze(0).to(self.device))
            return feat.cpu().numpy()[0]

    @staticmethod
    def appearance_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        if feat1 is None or feat2 is None:
            return 0.0
        sim = float(np.dot(feat1, feat2))
        return round(float(np.clip(sim, -1.0, 1.0)), 4)

reid_extractor_service = PersonReIDExtractor()
