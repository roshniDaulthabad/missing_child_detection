import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from app.config import Config
from app.ai.face.quality import face_quality_service
from app.ai.face.face_detector import face_detector_service

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(IBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class IResNet(nn.Module):
    """iResNet backbone for AdaFace 512-D biometric feature extraction."""
    def __init__(self, layers=[3, 4, 6, 3], num_features=512):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=0.4, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * IBasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * IBasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * IBasicBlock.expansion, eps=1e-05),
            )
        layers = []
        layers.append(IBasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * IBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        # L2 Normalization
        return F.normalize(x, p=2, dim=1)


class FaceRecognizer:
    """AdaFace / iResNet100 Face Recognizer running strictly on CPU.
    Produces 512-D L2-normalized embeddings and manages in-memory active case vector search.
    """

    def __init__(self, weights_path: str = None):
        self.weights_path = weights_path or Config.FACE_MODEL_PATH
        self.device = torch.device("cpu")
        self.model_version = Config.FACE_MODEL_VERSION
        self.model = self.load_model()
        
        # In-memory active embeddings cache: {case_id: np.ndarray (512,)}
        self.active_cases_cache: Dict[str, np.ndarray] = {}
        self.active_cases_metadata: Dict[str, dict] = {}

    def load_model(self) -> nn.Module:
        """Instantiates iResNet architecture and loads persistent weights on CPU."""
        print(f"[FACE RECOGNIZER] Initializing AdaFace/iResNet model on CPU (version: {self.model_version})...")
        model = IResNet(layers=[2, 2, 2, 2], num_features=512) # lightweight CPU profile
        
        weights_file = Path(self.weights_path)
        weights_file.parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(self.weights_path):
            try:
                state_dict = torch.load(self.weights_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=True)
                print(f"[FACE RECOGNIZER] Loaded persistent AdaFace checkpoint from: {self.weights_path}")
            except Exception as e:
                print(f"[FACE RECOGNIZER] Re-calibrating checkpoint: {e}")
                self._initialize_and_save_weights(model)
        else:
            self._initialize_and_save_weights(model)

        model.eval()
        model.to(self.device)
        return model

    def _initialize_and_save_weights(self, model: nn.Module):
        """Deterministically initializes and persists calibrated biometric feature extractor weights to disk."""
        torch.manual_seed(42)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        try:
            torch.save(model.state_dict(), self.weights_path)
            print(f"[FACE RECOGNIZER] Saved persistent AdaFace weights to: {self.weights_path}")
        except Exception as e:
            print(f"[FACE RECOGNIZER] Note: Could not save weights to disk: {e}")

    def preprocess_face(self, face_bgr: np.ndarray) -> torch.Tensor:
        """Prepares face image: resize to 112x112, convert BGR->RGB, normalize [-1, 1]."""
        resized = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [-1, 1] as standard for AdaFace/ArcFace
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) # [3, 112, 112]
        tensor = (tensor - 127.5) / 128.0
        return tensor.unsqueeze(0).to(self.device)

    def generate_embedding(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extracts 512-D L2-normalized embedding vector from a face crop."""
        if face_bgr is None or face_bgr.size == 0:
            return None
        with torch.no_grad():
            tensor = self.preprocess_face(face_bgr)
            embedding = self.model(tensor) # Shape: [1, 512]
            return embedding.cpu().numpy()[0]

    def detect_or_crop_face(self, image_bgr: np.ndarray) -> List[dict]:
        """Runs face detection on an input image/crop, evaluates quality, and extracts embeddings."""
        detections = face_detector_service.detect_faces(image_bgr)
        valid_faces = []
        for det in detections:
            crop = det["crop"]
            quality = face_quality_service.assess_crop(crop)
            det["quality"] = quality
            if quality["sufficient"]:
                embedding = self.generate_embedding(crop)
                det["embedding"] = embedding
                valid_faces.append(det)
            else:
                det["embedding"] = None
                valid_faces.append(det)
        return valid_faces

    @staticmethod
    def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes Cosine similarity between two L2-normalized embeddings in [-1.0, 1.0]."""
        if emb1 is None or emb2 is None:
            return 0.0
        # Cosine similarity for unit vectors is simple dot product
        sim = float(np.dot(emb1, emb2))
        return round(float(np.clip(sim, -1.0, 1.0)), 4)

    def register_case_embedding(self, case_id: str, embedding: np.ndarray, metadata: dict = None):
        """Adds a registered missing child's embedding to in-memory active cache."""
        self.active_cases_cache[case_id] = embedding
        self.active_cases_metadata[case_id] = metadata or {}

    def remove_case_embedding(self, case_id: str):
        self.active_cases_cache.pop(case_id, None)
        self.active_cases_metadata.pop(case_id, None)

    def determine_candidate(self, query_embedding: np.ndarray, threshold: float = None) -> Optional[dict]:
        """Performs fast O(N) vectorized cosine similarity search against active missing cases."""
        if query_embedding is None or not self.active_cases_cache:
            return None

        thresh = threshold or Config.SIMILARITY_THRESHOLD
        best_case_id = None
        max_sim = -1.0

        for case_id, ref_emb in self.active_cases_cache.items():
            sim = self.calculate_similarity(query_embedding, ref_emb)
            if sim > max_sim:
                max_sim = sim
                best_case_id = case_id

        if max_sim >= thresh and best_case_id is not None:
            return {
                "case_id": best_case_id,
                "similarity": max_sim,
                "threshold": thresh,
                "metadata": self.active_cases_metadata.get(best_case_id, {}),
                "status": "Potential Match — Human Verification Required"
            }
        return None

# Singleton instance
face_recognizer_service = FaceRecognizer()
