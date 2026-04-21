#!/usr/bin/env python3
"""
Idea-B: Video Interpolation Semantic Anchor — Lite version
50 pairs for timing validation, then decide.

Hypothesis: DINOv2 L2(I,A)+L2(I,B) predicts L2(A,B) for semantically distant pairs.
With only DINOv2 as metric, we test whether interpolation artifact magnitude
scales with source semantic distance (as captured by DINOv2 L2(A,B)).

50 pairs for quick timing: expected ~4.5 min
100 pairs: expected ~9 min
"""

import json, os, sys
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy import stats
from pathlib import Path
import time

DEVICE = "cpu"
ARTIFACT_DIR = Path("/home/kas/.openclaw/workspace-domain/research/autonomous-research-window-0422-am")
ARTIFACT_DIR.mkdir(exist_ok=True)
os.chdir(ARTIFACT_DIR)

t_start = time.time()
print(f"Device: {DEVICE}")
print(f"Start: {time.strftime('%H:%M:%S')}")

# ── Load CIFAR-10 ──────────────────────────────────────────────────────────
print("\n[1/5] Loading CIFAR-10...")
dataset = torchvision.datasets.CIFAR10(
    root=str(ARTIFACT_DIR / "data"), train=False, download=True
)
print(f"  {len(dataset)} images")

# ── Load DINOv2 ─────────────────────────────────────────────────────────────
print("\n[2/5] Loading DINOv2 ViT-S/14...")
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cpu().eval()
print(f"  {sum(p.numel() for p in dinov2.parameters())/1e6:.1f}M params")

def resize_224(img_pil):
    return img_pil.resize((224, 224), Image.BILINEAR)

def to_tensor(img_pil):
    tensor = torchvision.transforms.ToTensor()(resize_224(img_pil))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std

def dino_l2(img_a_pil, img_b_pil):
    with torch.no_grad():
        f_a = dinov2(to_tensor(img_a_pil).unsqueeze(0))
        f_b = dinov2(to_tensor(img_b_pil).unsqueeze(0))
    return torch.norm(f_a - f_b, p=2).item()

# ── Sample pairs from DIFFERENT classes ───────────────────────────────────
N_PAIRS = 50  # lite version for timing
print(f"\n[3/5] Sampling {N_PAIRS} pairs from different classes...")
np.random.seed(42)

class_images = {c: [] for c in range(10)}
for idx, (img, label) in enumerate(dataset):
    class_images[label].append((idx, img))

pairs = []
attempts = 0
while len(pairs) < N_PAIRS and attempts < 100000:
    attempts += 1
    c1, c2 = np.random.choice(10, 2, replace=False)
    if len(class_images[c1]) == 0 or len(class_images[c2]) == 0:
        continue
    idx1 = np.random.randint(len(class_images[c1]))
    idx2 = np.random.randint(len(class_images[c2]))
    pairs.append((class_images[c1][idx1][0], class_images[c2][idx2][0], c1, c2))

print(f"  Sampled {len(pairs)} pairs")

# ── Run experiment ──────────────────────────────────────────────────────────
print(f"\n[4/5] Processing {N_PAIRS} pairs...")

results = []
t_pair = time.time()
for i, (idx_a, idx_b, c_a, c_b) in enumerate(pairs):
    if i % 10 == 0:
        elapsed = time.time() - t_start
        print(f"  Pair {i}/{len(pairs)}... ({elapsed:.0f}s elapsed)")

    img_a = dataset[idx_a][0]
    img_b = dataset[idx_b][0]

    # Bilinear interpolation at α=0.5
    arr_a = np.array(img_a).astype(np.float32) / 255.0
    arr_b = np.array(img_b).astype(np.float32) / 255.0
    arr_i = 0.5 * arr_a + 0.5 * arr_b
    img_i = Image.fromarray((np.clip(arr_i, 0, 1) * 255).astype(np.uint8))

    l2_ia = dino_l2(img_i, img_a)
    l2_ib = dino_l2(img_i, img_b)
    l2_ab = dino_l2(img_a, img_b)

    results.append({
        "pair_id": i,
        "class_a": int(c_a), "class_b": int(c_b),
        "dino_l2_ia": float(l2_ia),
        "dino_l2_ib": float(l2_ib),
        "dino_l2_ab": float(l2_ab),
        "dino_sum":   float(l2_ia + l2_ib),
    })

# ── Compute correlations ────────────────────────────────────────────────────
print("\n[5/5] Computing correlations...")

l2_ia_arr  = np.array([r["dino_l2_ia"] for r in results])
l2_ib_arr  = np.array([r["dino_l2_ib"] for r in results])
l2_ab_arr  = np.array([r["dino_l2_ab"] for r in results])
dino_sum_arr = np.array([r["dino_sum"] for r in results])

r_sum_ab, p_sum_ab = stats.pearsonr(dino_sum_arr, l2_ab_arr)
r_sp_ab, p_sp_ab   = stats.spearmanr(dino_sum_arr, l2_ab_arr)
r_ia_ab, p_ia_ab   = stats.pearsonr(l2_ia_arr, l2_ab_arr)
r_ib_ab, p_ib_ab   = stats.pearsonr(l2_ib_arr, l2_ab_arr)

print(f"\nDINOv2_sum vs L2(A,B) [primary] (n={N_PAIRS})")
print(f"  Pearson r  = {r_sum_ab:.4f}, p = {p_sum_ab:.2e}")
print(f"  Spearman ρ = {r_sp_ab:.4f}, p = {p_sp_ab:.2e}")
print(f"\nL2(I,A) vs L2(A,B) (n={N_PAIRS})")
print(f"  Pearson r  = {r_ia_ab:.4f}, p = {p_ia_ab:.2e}")
print(f"\nL2(I,B) vs L2(A,B) (n={N_PAIRS})")
print(f"  Pearson r  = {r_ib_ab:.4f}, p = {p_ib_ab:.2e}")

print(f"\nDINOv2_sum: mean={dino_sum_arr.mean():.2f}, std={dino_sum_arr.std():.2f}")
print(f"L2(A,B):     mean={l2_ab_arr.mean():.2f}, std={l2_ab_arr.std():.2f}")

# ── Decision ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if r_sum_ab > 0.5:
    decision = "CONFIRM"
    print(f"  [{decision}] r={r_sum_ab:.4f} > 0.5 — interpolation artifact scales with source distance")
elif r_sum_ab < 0.3:
    decision = "FAIL"
    print(f"  [{decision}] r={r_sum_ab:.4f} < 0.3 — DINOv2 L2 does NOT predict source distance")
else:
    decision = "INCONCLUSIVE"
    print(f"  [{decision}] r={r_sum_ab:.4f} in [0.3, 0.5] — weak signal")
print("="*60)

t_total = time.time() - t_start
print(f"\nTotal runtime: {t_total/60:.1f} min ({t_total:.0f}s)")

# ── Save ───────────────────────────────────────────────────────────────────
output = {
    "decision": decision,
    "r_pearson": float(r_sum_ab),
    "p_pearson": float(p_sum_ab),
    "r_spearman": float(r_sp_ab),
    "p_spearman": float(p_sp_ab),
    "r_l2ia_ab": float(r_ia_ab),
    "r_l2ib_ab": float(r_ib_ab),
    "n_pairs": N_PAIRS,
    "dino_sum_mean": float(dino_sum_arr.mean()),
    "dino_sum_std": float(dino_sum_arr.std()),
    "l2_ab_mean": float(l2_ab_arr.mean()),
    "l2_ab_std": float(l2_ab_arr.std()),
    "runtime_min": round(t_total/60, 1),
    "model": "DINOv2 ViT-S/14 (CPU-only, CLIP unavailable)",
}
with open(ARTIFACT_DIR / "idea-b-results.json", "w") as f:
    json.dump(output, f, indent=2)

import csv
with open(ARTIFACT_DIR / "idea-b-results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print(f"Results → {ARTIFACT_DIR / 'idea-b-results.json'}")