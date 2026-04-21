#!/usr/bin/env python3
"""
Nova-Idea-B: Video Interpolation Semantic Anchor
Protocol: research/autonomous-research-window-0421-late/idea-b-protocol.md

Hypothesis: DINOv2 L2 distance on interpolated frames predicts CLIP semantic discrepancy.
Correlation: r(DINOv2_L2(I,A) + DINOv2_L2(I,B), 1 - CLIP_CS(A,B))

CPU-only, RAM ~1.5GB constraint.
DINOv2 ViT-S/14, CLIP ViT-B/32 (subset only), batch_size=1
"""

import sys
import os
import time
import random
import math

import torch
import torchvision
import numpy as np
from scipy import stats

# ========== SETUP ==========
DEVICE = "cpu"
SEED = 42
N_PAIRS = 500        # total pairs for DINOv2
N_CLIP_SUBSET = 100  # CLIP subset for RAM constraint
INTERPOLATION_ALPHA = 0.5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("Idea-B Experiment: Video Interpolation Semantic Anchor")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"N_pairs (DINOv2): {N_PAIRS}")
print(f"N_pairs (CLIP subset): {N_CLIP_SUBSET}")
print(f"Interpolation alpha: {INTERPOLATION_ALPHA}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ========== LOAD CIFAR-10 TEST SET ==========
print("[1/6] Loading CIFAR-10 test set...")
data_path = os.path.expanduser("~/data")
os.makedirs(data_path, exist_ok=True)
cifar_test = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)
print(f"  CIFAR-10 test set loaded: {len(cifar_test)} images")

# Build class index lists
class_to_indices = {c: [] for c in range(10)}
for idx, (_, label) in enumerate(cifar_test):
    class_to_indices[label].append(idx)

print(f"  Class distribution: {[len(v) for v in class_to_indices.values()]}")

# ========== SAMPLE 500 PAIRS FROM DIFFERENT CLASSES ==========
print(f"\n[2/6] Sampling {N_PAIRS} pairs from different classes...")
pairs = []
attempts = 0
max_attempts = N_PAIRS * 20
while len(pairs) < N_PAIRS and attempts < max_attempts:
    attempts += 1
    c1, c2 = random.sample(range(10), 2)  # two different classes
    idx1 = random.choice(class_to_indices[c1])
    idx2 = random.choice(class_to_indices[c2])
    pairs.append((idx1, idx2, c1, c2))

print(f"  Sampled {len(pairs)} pairs in {attempts} attempts")
# Count class pair distribution
class_pair_counts = {}
for _, _, c1, c2 in pairs:
    key = tuple(sorted([c1, c2]))
    class_pair_counts[key] = class_pair_counts.get(key, 0) + 1
print(f"  Unique class pairs: {len(class_pair_counts)}")

# ========== DINOv2 SETUP ==========
print(f"\n[3/6] Loading DINOv2 ViT-S/14...")
t0 = time.time()
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2 = dinov2.to(DEVICE).eval()
print(f"  DINOv2 loaded in {time.time()-t0:.1f}s")

# DINOv2 preprocess: ImageNet normalization
dinov2_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
dinov2_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def dinov2_preprocess(img_tensor):
    """img_tensor: [H, W, C] uint8 [0,255] -> [3, 224, 224] normalized DINOv2 input"""
    # img_tensor is [H, W, C] from numpy array
    img = img_tensor.permute(2, 0, 1).float() / 255.0  # -> [C, H, W]
    # Resize 32x32 -> 224x224 for ViT
    img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
    img = (img - dinov2_mean) / dinov2_std
    return img

# ========== DINOv2 FEATURE EXTRACTION (500 pairs) ==========
print(f"\n[4/6] Extracting DINOv2 features for {N_PAIRS} pairs...")
dino_l2_sums = []  # DINOv2_L2(I,A) + DINOv2_L2(I,B)
clip_cs_ab_list = []  # CLIP_CS(A,B) placeholder — filled later

t0 = time.time()
with torch.no_grad():
    for i, (idx1, idx2, c1, c2) in enumerate(pairs):
        # Load images
        img1, _ = cifar_test[idx1]
        img2, _ = cifar_test[idx2]
        
        # To tensor
        t1 = torch.from_numpy(np.array(img1))
        t2 = torch.from_numpy(np.array(img2))
        
        # Interpolation: I = alpha*A + (1-alpha)*B
        I = INTERPOLATION_ALPHA * t1 + (1 - INTERPOLATION_ALPHA) * t2
        
        # DINOv2 features (batch_size=1)
        feat_I = dinov2(dinov2_preprocess(I).unsqueeze(0))
        feat_A = dinov2(dinov2_preprocess(t1).unsqueeze(0))
        feat_B = dinov2(dinov2_preprocess(t2).unsqueeze(0))
        
        # L2 distances
        l2_IA = torch.norm(feat_I - feat_A, p=2).item()
        l2_IB = torch.norm(feat_I - feat_B, p=2).item()
        dino_l2_sums.append(l2_IA + l2_IB)
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_PAIRS - i - 1)
            print(f"  [{i+1}/{N_PAIRS}] DINOv2 done | elapsed={elapsed:.1f}s | ETA={eta:.1f}s")

dino_l2_sums = np.array(dino_l2_sums)
print(f"  DINOv2 extraction complete: {len(dino_l2_sums)} values")
print(f"  DINOv2_L2_sum stats: mean={dino_l2_sums.mean():.4f}, std={dino_l2_sums.std():.4f}, min={dino_l2_sums.min():.4f}, max={dino_l2_sums.max():.4f}")

# ========== CLIP SETUP ==========
print(f"\n[5/6] Loading CLIP ViT-B/32...")
t0 = time.time()
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
print(f"  CLIP loaded in {time.time()-t0:.1f}s")

def clip_preprocess_fn(img_tensor):
    """img_tensor: [3, 32, 32] uint8 [0,255] -> CLIP preprocessed tensor"""
    # Convert to PIL-like: HWC -> resize -> tensor
    img_np = img_tensor.numpy().astype(np.uint8)
    from PIL import Image
    pil_img = Image.fromarray(img_np)
    # Resize to 224x224
    pil_img = pil_img.resize((224, 224), Image.BILINEAR)
    return clip_preprocess(pil_img)

def clip_cosine_sim(img1_tensor, img2_tensor):
    """Compute CLIP cosine similarity between two image tensors."""
    with torch.no_grad():
        img1_prep = clip_preprocess_fn(img1_tensor).unsqueeze(0).to(DEVICE)
        img2_prep = clip_preprocess_fn(img2_tensor).unsqueeze(0).to(DEVICE)
        feat1 = clip_model.encode_image(img1_prep)
        feat2 = clip_model.encode_image(img2_prep)
        feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
        feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
        return (feat1 @ feat2.T).item()

# ========== CLIP CS COMPUTATION (subset only) ==========
print(f"\n[6/6] Computing CLIP CS for {N_CLIP_SUBSET} subset pairs...")
t0 = time.time()
clip_cs_ab = []  # CLIP_CS(A,B)
clip_cs_IA = []  # CLIP_CS(I,A)  
clip_cs_IB = []  # CLIP_CS(I,B)
pair_indices = list(range(N_PAIRS))
random.shuffle(pair_indices)
subset_indices = pair_indices[:N_CLIP_SUBSET]

for count, pair_idx in enumerate(subset_indices):
    idx1, idx2, c1, c2 = pairs[pair_idx]
    
    img1, _ = cifar_test[idx1]
    img2, _ = cifar_test[idx2]
    
    t1 = torch.from_numpy(np.array(img1))
    t2 = torch.from_numpy(np.array(img2))
    I = INTERPOLATION_ALPHA * t1 + (1 - INTERPOLATION_ALPHA) * t2
    
    cs_ab = clip_cosine_sim(t1, t2)
    cs_IA = clip_cosine_sim(I, t1)
    cs_IB = clip_cosine_sim(I, t2)
    
    clip_cs_ab.append(cs_ab)
    clip_cs_IA.append(cs_IA)
    clip_cs_IB.append(cs_IB)
    
    if (count + 1) % 20 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (count + 1) * (N_CLIP_SUBSET - count - 1)
        print(f"  [{count+1}/{N_CLIP_SUBSET}] CLIP done | elapsed={elapsed:.1f}s | ETA={eta:.1f}s")

clip_cs_ab = np.array(clip_cs_ab)
clip_cs_IA = np.array(clip_cs_IA)
clip_cs_IB = np.array(clip_cs_IB)
print(f"  CLIP computation complete: {len(clip_cs_ab)} values")
print(f"  CLIP_CS(A,B) stats: mean={clip_cs_ab.mean():.4f}, std={clip_cs_ab.std():.4f}")
print(f"  CLIP_CS(I,A) stats: mean={clip_cs_IA.mean():.4f}, std={clip_cs_IA.std():.4f}")
print(f"  CLIP_CS(I,B) stats: mean={clip_cs_IB.mean():.4f}, std={clip_cs_IB.std():.4f}")

# ========== PRIMARY ANALYSIS: Correlation ==========
print(f"\n" + "=" * 60)
print("PRIMARY ANALYSIS")
print("=" * 60)

# For CLIP subset: DINOv2 L2 sum vs (1 - CLIP_CS(A,B))
dino_subset = dino_l2_sums[subset_indices]
semantic_drift = 1.0 - clip_cs_ab  # 1 - CLIP_CS(A,B)

r_primary, p_primary = stats.pearsonr(dino_subset, semantic_drift)
print(f"\nPrimary correlation (n={N_CLIP_SUBSET}):")
print(f"  r(DINOv2_L2_sum, 1 - CLIP_CS(A,B)) = {r_primary:.4f}")
print(f"  p-value = {p_primary:.6f}")

# Bootstrap CI for r
print(f"\nBootstrap 95% CI for r (1000 iterations)...")
n_bootstrap = 1000
r_boot = []
n_clip = len(dino_subset)
for _ in range(n_bootstrap):
    idx_boot = np.random.choice(n_clip, size=n_clip, replace=True)
    r_b, _ = stats.pearsonr(dino_subset[idx_boot], semantic_drift[idx_boot])
    r_boot.append(r_b)
r_boot = np.array(r_boot)
ci_low = np.percentile(r_boot, 2.5)
ci_high = np.percentile(r_boot, 97.5)
print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  Bootstrap mean r: {r_boot.mean():.4f}, std: {r_boot.std():.4f}")

# Also report DINOv2-only correlation (full 500 pairs)
# We need CLIP_CS(A,B) for full 500 pairs — can't compute due to RAM
# But we can report DINOv2 L2 distribution for high vs low CLIP_CS subset
print(f"\nDINOv2_L2_sum distribution by CLIP_CS(A,B) quartiles:")
q_labels = ['Q1 (low CS)', 'Q2', 'Q3', 'Q4 (high CS)']
q_bounds = [0, 0.25, 0.5, 0.75, 1.0]
for i in range(4):
    lo = np.percentile(clip_cs_ab, q_bounds[i]*100)
    hi = np.percentile(clip_cs_ab, q_bounds[i+1]*100)
    mask = (clip_cs_ab >= lo) & (clip_cs_ab <= hi)
    subset_dino = dino_subset[mask]
    print(f"  {q_labels[i]}: CS in [{lo:.3f}, {hi:.3f}], n={mask.sum()}, DINO_L2_mean={subset_dino.mean():.2f}")

# CLIP CS drop analysis
cs_drop_IA = clip_cs_IA - clip_cs_ab  # How much does I drift from A relative to A-B distance?
cs_drop_IB = clip_cs_IB - clip_cs_ab
print(f"\nSemantic drift of interpolated frame I:")
print(f"  CLIP_CS(I,A) - CLIP_CS(A,B): mean={cs_drop_IA.mean():.4f}, std={cs_drop_IA.std():.4f}")
print(f"  CLIP_CS(I,B) - CLIP_CS(A,B): mean={cs_drop_IB.mean():.4f}, std={cs_drop_IB.std():.4f}")

# DINOv2 vs CLIP CS drop correlation
r_drop_IA, p_drop_IA = stats.pearsonr(dino_subset, -cs_drop_IA)
r_drop_IB, p_drop_IB = stats.pearsonr(dino_subset, -cs_drop_IB)
print(f"\nDINOv2 L2 sum vs CLIP CS drop:")
print(f"  r(DINO_L2, -(CS(I,A)-CS(A,B))) = {r_drop_IA:.4f}, p={p_drop_IA:.6f}")
print(f"  r(DINO_L2, -(CS(I,B)-CS(A,B))) = {r_drop_IB:.4f}, p={p_drop_IB:.6f}")

# ========== FAILURE CONDITION CHECK ==========
print(f"\n" + "=" * 60)
print("FAILURE CONDITION CHECK")
print("=" * 60)
failure_condition = "r < 0.3 = FAIL; r 0.3-0.5 = INCONCLUSIVE; r > 0.5 = CONFIRM"
print(f"Threshold: {failure_condition}")
print(f"Observed r = {r_primary:.4f}")

if r_primary > 0.5:
    verdict = "CONFIRM"
    print(f">>> VERDICT: {verdict} — semantic anchor validated for interpolation detection")
elif r_primary >= 0.3:
    verdict = "INCONCLUSIVE"
    print(f">>> VERDICT: {verdict} — weak signal, needs refinement or larger n")
else:
    verdict = "FAIL"
    print(f">>> VERDICT: {verdict} — DINOv2 L2 does not detect interpolation semantic drift")

# ========== SUMMARY ==========
print(f"\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
summary = {
    "n_pairs_total": N_PAIRS,
    "n_clip_subset": N_CLIP_SUBSET,
    "r_primary": r_primary,
    "p_primary": p_primary,
    "ci_95": [ci_low, ci_high],
    "verdict": verdict,
    "dino_l2_mean": float(dino_l2_sums.mean()),
    "dino_l2_std": float(dino_l2_sums.std()),
    "clip_cs_ab_mean": float(clip_cs_ab.mean()),
    "clip_cs_IA_mean": float(clip_cs_IA.mean()),
    "clip_cs_IB_mean": float(clip_cs_IB.mean()),
}
for k, v in summary.items():
    print(f"  {k}: {v}")

print(f"\nTotal runtime: {time.strftime('%H:%M:%S')}")

# ========== SAVE RESULTS ==========
output_path = os.path.expanduser("~/data/idea_b_results_temp.json")
import json
with open(output_path, 'w') as f:
    json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
print(f"\nResults saved to {output_path}")