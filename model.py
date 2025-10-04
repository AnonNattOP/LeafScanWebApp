import io
import json
from pathlib import Path
from typing import IO, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


class PlantDiseaseModel:
    def __init__(self, model_path: str | Path, labels_path: str | Path | None = None, device: str | None = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, num_classes = self._load_model()
        self.class_names = self._load_class_names(labels_path, num_classes)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load_model(self) -> Tuple[nn.Module, int]:
        raw = torch.load(self.model_path, map_location=self.device)

        if isinstance(raw, nn.Module):
            model = raw.to(self.device)
            model.eval()
            try:
                num_classes = model.classifier[1].out_features  # type: ignore[index]
            except (AttributeError, IndexError):
                raise ValueError("Loaded model does not expose classifier information") from None
            return model, num_classes

        if isinstance(raw, dict):
            state_dict = self._extract_state_dict(raw)
        else:
            raise TypeError("Unexpected model checkpoint format")

        if "classifier.1.weight" not in state_dict:
            raise KeyError("Classifier weights not found in checkpoint")

        num_classes = state_dict["classifier.1.weight"].shape[0]
        backbone = efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features  # type: ignore[index]
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        backbone.load_state_dict(state_dict)
        backbone.to(self.device)
        backbone.eval()
        return backbone, num_classes

    @staticmethod
    def _extract_state_dict(raw: dict) -> dict:
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state_dict = raw["state_dict"]
        elif "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
            state_dict = raw["model_state_dict"]
        else:
            state_dict = raw

        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                cleaned[key[len("module."):]] = value
            else:
                cleaned[key] = value
        return cleaned

    def _load_class_names(self, labels_path: str | Path | None, num_classes: int) -> List[str]:
        labels_file = Path(labels_path) if labels_path else self.model_path.with_name("class_names.json")
        if labels_file.exists():
            try:
                data = json.loads(labels_file.read_text(encoding="utf-8"))
                if isinstance(data, list) and len(data) == num_classes:
                    return [str(item) for item in data]
            except json.JSONDecodeError:
                pass
        return [f"Class {idx}" for idx in range(num_classes)]

    def predict(self, file: IO[bytes]) -> dict:
        buffer = io.BytesIO(file.read())
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        confidence, predicted_idx = torch.max(probabilities, dim=0)
        class_name = self.class_names[predicted_idx.item()]

        return {
            "label": class_name,
            "confidence": float(confidence.item()),
            "probabilities": [
                {
                    "label": self.class_names[i],
                    "confidence": float(probabilities[i].item()),
                }
                for i in range(len(self.class_names))
            ],
        }

    def get_class_names(self) -> List[str]:
        return list(self.class_names)
