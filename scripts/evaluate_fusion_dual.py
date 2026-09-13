"""Evaluates DualBranchFusionDetector on Local Validation and New-Data Validation."""

import json
import os
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.architectures.fusion import DualBranchFusionDetector
from model.dataset_splits import get_official_splits, get_v2_splits
from model.dataset import SignalScopeDataset, get_default_transforms, get_native_transforms
from model.evaluate import calculate_metrics

class DualInputDataset(SignalScopeDataset):
    def __init__(self, samples, transform_spatial, transform_native):
        super().__init__(samples, transform=transform_spatial)
        self.transform_native = transform_native

    def __getitem__(self, idx):
        path, label, gen = self.samples[idx]
        with Image.open(path) as img:
            pil_img = img.convert("RGB")
            if max(pil_img.size) > 512:
                pil_img.thumbnail((448, 448), Image.Resampling.BILINEAR)
        spatial_tensor = self.transform(pil_img)
        native_tensor = self.transform_native(pil_img)
        return spatial_tensor, native_tensor, torch.tensor([label], dtype=torch.float32), gen

def eval_fusion(model, samples, device, batch_size=64):
    transform_spatial = get_default_transforms(image_size=224, is_training=False)
    transform_native = get_native_transforms(image_size=32)
    ds = DualInputDataset(samples, transform_spatial, transform_native)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x_spatial, x_native, targets, _ in loader:
            x_spatial = x_spatial.to(device, non_blocking=True)
            x_native = x_native.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x_spatial, x_native=x_native)
                probs = torch.sigmoid(logits)
            all_targets.extend(targets.squeeze(-1).cpu().numpy().astype(int).tolist())
            all_probs.extend(probs.squeeze(-1).cpu().numpy().astype(float).tolist())

    y_true = np.array(all_targets, dtype=int)
    y_scores = np.array(all_probs, dtype=float)
    return calculate_metrics(y_true, y_scores, operating_threshold=0.50)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = PROJECT_ROOT / "checkpoints" / "fusion" / "best_model.pt"
    model = DualBranchFusionDetector(pretrained=False, frequency_feature_dim=128).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)

    _, val_old = get_official_splits(seed=42)
    _, val_new = get_v2_splits()

    print("Evaluating Fusion Model on LOCAL VALIDATION (15k)...")
    m_old = eval_fusion(model, val_old, device)
    print(f"Local Val: AUC={m_old['roc_auc']:.4f}, F1={m_old['macro_f1']:.4f}, Acc={m_old['accuracy']:.4f}, FPR={m_old['fpr']:.4f}")

    print("Evaluating Fusion Model on NEW-DATA VALIDATION (1.1k)...")
    m_new = eval_fusion(model, val_new, device)
    print(f"New Data Val: AUC={m_new['roc_auc']:.4f}, F1={m_new['macro_f1']:.4f}, Acc={m_new['accuracy']:.4f}, FPR={m_new['fpr']:.4f}")

    res = {
        "fusion_local_val": m_old,
        "fusion_new_val": m_new,
    }
    out_file = PROJECT_ROOT / "outputs" / "evaluation" / "fusion_dual_eval.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved results to {out_file}")
