#!/usr/bin/env python3
"""
Nova-Idea-B: Video Interpolation Semantic Anchor
Optimized version — DINOv2 only first pass, then CLIP on small subset
"""

import sys
import os
import time
import random

import torch
import torchvision
import numpy as np
from scipy import stats

# ========== SETUP ==========
DEVICE = "cpu"
SEED = 42
N_PAIRS = 500
N_CLIP_SUBSET = 100
INTERPOLATION_ALPHA = 0.5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("Idea-B Experiment: Video Interpolation Semantic Anchor")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"N_pairs: {N_PAIRS}")
print(f"N_clip_subset: {N_CLIP_SUBSET}")
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
sys.stdout.flush()

# ========== LOAD CIFAR-10 ==========
print("\n[1] Loading CIFAR-10 test set...")
data_path = os.path.expanduser("~/data")
os.makedirs(data_path, exist_ok=True)
cifar_test = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)
print(f"  Loaded {len(cifar_test)} images")
sys.stdout.flush()

# Build class index lists
class_to_indices = {c: [] for c in range(10)}
for idx, (_, label) in enumerate(cifar_test):
    class_to_indices[label].append(idx)

# ========== SAMPLE PAIRS ==========
print(f"\n[2] Sampling {N_PAIRS} pairs from different classes...")
pairs = []
for _ in range(N_PAIRS):
    c1, c2 = random.sample(range(10), 2)
    idx1 = random.choice(class_to_indices[c1])
    idx2 = random.choice(class_to_indices[c2])
    pairs.append((idx1, idx2, c1, c2))
print(f"  Sampled {len(pairs)} pairs")
sys.stdout.flush()

# ========== DINOv2 ==========
print(f"\n[3] Loading DINOv2 ViT-S/14...")
sys.stdout.flush()
t0 = time.time()
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE).eval()
print(f"  Loaded in {time.time()-t0:.1f}s")
sys.stdout.flush()

dinov2_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
dinov2_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def dinov2_preprocess(img_pil):
    """img_pil: PIL Image -> [3, 224, 224] tensor"""
    img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(224, 224), 
                                          mode='bilinear', align_corners=False).squeeze(0)
    img = (img - dinov2_mean) / dinov2_std
    return img

def get_dinov2_feature(img_pil):
    with torch.no_grad():
        x = dinov2_preprocess(img_pil).unsqueeze(0)
        return dinov2(x).squeeze(0)

# ========== DINOv2 EXTRACTION ==========
print(f"\n[4] Extracting DINOv2 features for {N_PAIRS} pairs...")
sys.stdout.flush()
t0 = time.time()

dino_l2_sums = []
for i, (idx1, idx2, c1, c2) in enumerate(pairs):
    img1, _ = cifar_test[idx1]
    img2, _ = cifar_test[idx2]
    
    # Interpolation
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    I_arr = (INTERPOLATION_ALPHA * arr1 + (1 - INTERPOLATION_ALPHA) * arr2).astype(np.uint8)
    
    from PIL import Image
    I_pil = Image.fromarray(I_arr)
    
    f_I = get_dinov2_feature(I_pil)
    f_A = get_dinov2_feature(img1)
    f_B = get_dinov2_feature(img2)
    
    l2_IA = torch.norm(f_I - f_A, p=2).item()
    l2_IB = torch.norm(f_I - f_B, p=2).item()
    dino_l2_sums.append(l2_IA + l2_IB)
    
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (N_PAIRS - i - 1)
        print(f"  [{i+1}/{N_PAIRS}] elapsed={elapsed:.0f}s ETA={eta:.0f}s")
        sys.stdout.flush()

dino_l2_sums = np.array(dino_l2_sums)
print(f"  DINOv2 done: mean={dino_l2_sums.mean():.4f}, std={dino_l2_sums.std():.4f}")
sys.stdout.flush()

# ========== CLIP ==========
print(f"\n[5] Loading CLIP ViT-B/32...")
sys.stdout.flush()
t0 = time.time()

import clip
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
print(f"  CLIP loaded in {time.time()-t0:.1f}s")
sys.stdout.flush()

def clip_cs(img1_pil, img2_pil):
    with torch.no_grad():
        x1 = clip_preprocess(img1_pil).unsqueeze(0).to(DEVICE)
        x2 = clip_preprocess(img2_pil).unsqueeze(0).to(DEVICE)
        f1 = clip_model.encode_image(x1)
        f2 = clip_model.encode_image(x2)
        f1 = f1 / f1.norm(dim=-1, keepdim=True)
        f2 = f2 / f2.norm(dim=-1, keepdim=True)
        return (f1 @ f2.T).item()

# ========== CLIP SUBSET ==========
print(f"\n[6] CLIP on {N_CLIP_SUBSET} subset pairs...")
sys.stdout.flush()
t0 = time.time()

subset_indices = list(range(N_PAIRS))
random.shuffle(subset_indices)
subset_indices = subset_indices[:N_CLIP_SUBSET]

clip_cs_ab = []
clip_cs_IA = []
clip_cs_IB = []

for count, pair_idx in enumerate(subset_indices):
    idx1, idx2, c1, c2 = pairs[pair_idx]
    img1, _ = cifar_test[idx1]
    img2, _ = cifar_test[idx2]
    
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    I_arr = (INTERPOLATION_ALPHA * arr1 + (1 - INTERPOLATION_ALPHA) * arr2).astype(np.uint8)
    from PIL import Image
    I_pil = Image.fromarray(I_arr)
    
    cs_ab = clip_cs(img1, img2)
    cs_IA = clip_cs(I_pil, img1)
    cs_IB = clip_cs(I_pil, img2)
    
    clip_cs_ab.append(cs_ab)
    clip_cs_IA.append(cs_IA)
    clip_cs_IB.append(cs_IB)
    
    if (count + 1) % 20 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (count + 1) * (N_CLIP_SUBSET - count - 1)
        print(f"  [{count+1}/{N_CLIP_SUBSET}] elapsed={elapsed:.0f}s ETA={eta:.0f}s")
        sys.stdout.flush()

clip_cs_ab = np.array(clip_cs_ab)
clip_cs_IA = np.array(clip_cs_IA)
clip_cs_IB = np.array(clip_cs_IB)
print(f"  CLIP done")
sys.stdout.flush()

# ========== ANALYSIS ==========
print(f"\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

dino_subset = dino_l2_sums[subset_indices]
semantic_drift = 1.0 - clip_cs_ab

r_primary, p_primary = stats.pearsonr(dino_subset, semantic_drift)
print(f"\nr(DINOv2_L2_sum, 1 - CLIP_CS(A,B)) = {r_primary:.4f}")
print(f"p-value = {p_primary:.6f}")

# Bootstrap CI
n_boot = 1000
r_boot = []
n = len(dino_subset)
for _ in range(n_boot):
    idx = np.random.choice(n, size=n, replace=True)
    r_b, _ = stats.pearsonr(dino_subset[idx], semantic_drift[idx])
    r_boot.append(r_b)
r_boot = np.array(r_boot)
ci_low, ci_high = np.percentile(r_boot, [2.5, 97.5])
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# Failure condition
print(f"\nFailure condition check:")
if r_primary > 0.5:
    verdict = "CONFIRM"
elif r_primary >= 0.3:
    verdict = "INCONCLUSIVE"
else:
    verdict = "FAIL"
print(f"  r={r_primary:.4f} -> {verdict}")

# CLIP CS stats
print(f"\nCLIP CS statistics:")
print(f"  CS(A,B): mean={clip_cs_ab.mean():.4f}, std={clip_cs_ab.std():.4f}")
print(f"  CS(I,A): mean={clip_cs_IA.mean():.4f}, std={clip_cs_IA.std():.4f}")
print(f"  CS(I,B): mean={clip_cs_IB.mean():.4f}, std={clip_cs_IB.std():.4f}")

print(f"\nTotal time: {time.strftime('%H:%M:%S')}")
print(f"\nSUMMARY:")
print(f"  n_pairs={N_PAIRS}")
print(f"  n_clip={N_CLIP_SUBSET}")
print(f"  r={r_primary:.4f}")
print(f"  p={p_primary:.6f}")
print(f"  ci=[{ci_low:.4f}, {ci_high:.4f}]")
print(f"  verdict={verdict}")
print(f"  dino_l2_mean={dino_l2_sums.mean():.4f}")
print(f"  dino_l2_std={dino_l2_sums.std():.4f}")

# Save to results file
results = {
    "n_pairs": N_PAIRS,
    "n_clip": N_CLIP_SUBSET,
    "r": float(r_primary),
    "p": float(p_primary),
    "ci_low": float(ci_low),
    "ci_high": float(ci_high),
    "verdict": verdict,
    "dino_l2_mean": float(dino_l2_sums.mean()),
    "dino_l2_std": float(dino_l2_sums.std()),
    "clip_cs_ab_mean": float(clip_cs_ab.mean()),
    "clip_cs_IA_mean": float(clip_cs_IA.mean()),
    "clip_cs_IB_mean": float(clip_cs_IB.mean()),
}

import json
out_path = os.path.expanduser("~/data/idea_b_results.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")