#!/usr/bin/env python3
"""
nova_idea_b_toy.py — Anchor-based interpolation experiment for CLIP semantic drift reduction.
Hypothesis: Selecting interpolation alpha via DINOv2 L2 anchor minimizes semantic drift vs naive midpoint.

Failure conditions:
  - r(DINOv2_L2(A,B), ΔCS) < 0.3  → FAIL
  - mean(ΔCS) ≤ 0                  → FAIL
"""

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image

# Seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cpu"
DTYPE = torch.float32
N_PAIRS = 200
ALPHA_CANDIDATES = [0.2, 0.35, 0.5, 0.65, 0.8]
ALPHA_NAIVE = 0.5

print(f"[{time.strftime('%H:%M:%S')}] === nova_idea_b_toy start ===")
t_start = time.time()

# ── 1. Load CIFAR-10 ──────────────────────────────────────────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Loading CIFAR-10 test set...")
cifar = CIFAR10(root='./data', train=False, download=True)
print(f"[{time.strftime('%H:%M:%S')}] CIFAR-10 loaded: {len(cifar)} images")

# ── 2. Sample 200 cross-class pairs ──────────────────────────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Sampling {N_PAIRS} cross-class pairs...")

class_indices = {c: [] for c in range(10)}
for idx, (_, label) in enumerate(cifar):
    class_indices[label].append(idx)

pairs = []
for _ in range(N_PAIRS):
    c1, c2 = random.sample(range(10), 2)   # different classes, no replacement
    i1 = random.choice(class_indices[c1])
    i2 = random.choice(class_indices[c2])
    img1 = cifar[i1][0]   # PIL Image
    img2 = cifar[i2][0]
    pairs.append((img1, img2, c1, c2))

print(f"[{time.strftime('%H:%M:%S')}] Sampled {len(pairs)} pairs")

# ── 3. Preprocess helpers ─────────────────────────────────────────────────────
# Standard ImageNet-style for CLIP (224×224) and DINOv2 (224×224)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def pil_to_tensor(pil_img):
    """PIL → [0,1] float tensor [3, 224, 224]"""
    return transform_224(pil_img).to(DTYPE)

def tensor_to_pil(t):
    """[3, H, W] float tensor [0,1] → PIL (uint8)"""
    t = t.cpu().float().clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

# ── 4. Load DINOv2 ───────────────────────────────────────────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Loading DINOv2 ViT-S/14...")
t_dino_load = time.time()
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2 = dinov2.to(DEVICE).eval()
print(f"[{time.strftime('%H:%M:%S')}] DINOv2 loaded in {time.time()-t_dino_load:.1f}s")

def dinov2_feat(img_tensor):
    """Extract DINOv2 feature [3,224,224] → [384]"""
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(DEVICE)        # [1,3,224,224]
        feat = dinov2(x)                               # [1,384]
    return feat.squeeze(0).cpu().to(DTYPE)             # [384]

def dinov2_l2(feat_a, feat_b):
    return (feat_a - feat_b).pow(2).sum().sqrt().item()

# ── 5. Phase 1: Anchor search via DINOv2 L2 ──────────────────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Phase 1: DINOv2 anchor search for {N_PAIRS} pairs...")
t_phase1 = time.time()

phase1_results = []  # list of dicts

for pair_idx, (img_a_pil, img_b_pil, c1, c2) in enumerate(pairs):
    # Preprocess
    tA = pil_to_tensor(img_a_pil)   # [3,224,224]
    tB = pil_to_tensor(img_b_pil)

    # Naive interpolation (alpha=0.5)
    t_naive = 0.5 * tA + 0.5 * tB

    # DINOv2 source features (once per pair)
    feat_A = dinov2_feat(tA)
    feat_B = dinov2_feat(tB)

    # Anchor search over 5 alpha candidates
    best_alpha = ALPHA_NAIVE
    best_score = float('inf')   # we minimize sum of L2 distances
    best_t = t_naive

    for alpha in ALPHA_CANDIDATES:
        t_interp = alpha * tA + (1 - alpha) * tB
        feat_interp = dinov2_feat(t_interp)
        score = dinov2_l2(feat_interp, feat_A) + dinov2_l2(feat_interp, feat_B)
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_t = t_interp

    # DINOv2 L2 distance between A and B (for correlation analysis later)
    dino_l2_AB = dinov2_l2(feat_A, feat_B)

    phase1_results.append({
        'naive': t_naive,           # [3,224,224] tensor
        'anchor': best_t,          # [3,224,224] tensor
        'anchor_alpha': best_alpha,
        'dino_l2_AB': dino_l2_AB,
        'class_pair': (c1, c2),
    })

    if (pair_idx + 1) % 50 == 0:
        elapsed = time.time() - t_phase1
        print(f"  [{pair_idx+1}/{N_PAIRS}] pairs processed ({elapsed:.1f}s elapsed)")

print(f"[{time.strftime('%H:%M:%S')}] Phase 1 done in {time.time()-t_phase1:.1f}s")

# Free DINOv2 RAM
del dinov2
import gc; gc.collect()
print(f"[{time.strftime('%H:%M:%S')}] DINOv2 unloaded, RAM freed")

# ── 6. Load CLIP ──────────────────────────────────────────────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Loading CLIP ViT-B/32...")
t_clip_load = time.time()
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
print(f"[{time.strftime('%H:%M:%S')}] CLIP loaded in {time.time()-t_clip_load:.1f}s")

def clip_encode(pil_img):
    """Encode PIL → CLIP [512] float32"""
    with torch.no_grad():
        x = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
        feat = clip_model.encode_image(x).squeeze(0).cpu().to(DTYPE)
    return feat

def clip_cosine_sim(feat_a, feat_b):
    a_n = feat_a / feat_a.norm()
    b_n = feat_b / feat_b.norm()
    return (a_n @ b_n).item()

# ── 7. Phase 2: CLIP cosine similarity evaluation ────────────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Phase 2: CLIP evaluation for {N_PAIRS} pairs...")
t_phase2 = time.time()

pairwise_results = []   # list of dicts with scalar metrics

for pair_idx, ((img_a_pil, img_b_pil, c1, c2), pr) in enumerate(zip(pairs, phase1_results)):
    # Encode all 4 images as CLIP
    feat_A     = clip_encode(img_a_pil)
    feat_B     = clip_encode(img_b_pil)
    feat_naive = clip_encode(tensor_to_pil(pr['naive']))
    feat_anchor= clip_encode(tensor_to_pil(pr['anchor']))

    # CLIP cosine similarities
    cs_naive_A  = clip_cosine_sim(feat_naive,  feat_A)
    cs_naive_B  = clip_cosine_sim(feat_naive,  feat_B)
    cs_anchor_A = clip_cosine_sim(feat_anchor, feat_A)
    cs_anchor_B = clip_cosine_sim(feat_anchor, feat_B)

    # ΔCS metrics
    delta_cs_A    = cs_anchor_A - cs_naive_A
    delta_cs_B    = cs_anchor_B - cs_naive_B
    delta_cs_total= delta_cs_A + delta_cs_B

    pairwise_results.append({
        'pair_idx': pair_idx,
        'class_pair': (c1, c2),
        'anchor_alpha': pr['anchor_alpha'],
        'dino_l2_AB': pr['dino_l2_AB'],
        'cs_naive_A': cs_naive_A,
        'cs_naive_B': cs_naive_B,
        'cs_anchor_A': cs_anchor_A,
        'cs_anchor_B': cs_anchor_B,
        'delta_cs_A': delta_cs_A,
        'delta_cs_B': delta_cs_B,
        'delta_cs_total': delta_cs_total,
    })

    if (pair_idx + 1) % 50 == 0:
        elapsed = time.time() - t_phase2
        print(f"  [{pair_idx+1}/{N_PAIRS}] CLIP evals done ({elapsed:.1f}s elapsed)")

print(f"[{time.strftime('%H:%M:%S')}] Phase 2 done in {time.time()-t_phase2:.1f}s")

# Free CLIP RAM
del clip_model
import gc; gc.collect()
print(f"[{time.strftime('%H:%M:%S')}] CLIP unloaded, RAM freed")

# ── 8. Phase 3: Analysis ──────────────────────────────────────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Phase 3: Analysis...")

from scipy.stats import pearsonr, ttest_1samp

dino_l2_arr   = np.array([r['dino_l2_AB']    for r in pairwise_results])
delta_cs_arr  = np.array([r['delta_cs_total'] for r in pairwise_results])
delta_cs_A_arr= np.array([r['delta_cs_A']    for r in pairwise_results])
delta_cs_B_arr= np.array([r['delta_cs_B']    for r in pairwise_results])

mean_delta_cs  = float(np.mean(delta_cs_arr))
std_delta_cs   = float(np.std(delta_cs_arr))
mean_delta_A   = float(np.mean(delta_cs_A_arr))
mean_delta_B   = float(np.mean(delta_cs_B_arr))

# Pearson correlation
r_val, r_pval = pearsonr(dino_l2_arr, delta_cs_arr)

# Paired t-test vs 0
t_stat, p_val = ttest_1samp(delta_cs_arr, 0)

n_positive = int(np.sum(delta_cs_arr > 0))
n_negative = int(np.sum(delta_cs_arr < 0))
n_zero     = int(np.sum(delta_cs_arr == 0))

# Failure conditions
fail_cond1 = r_val < 0.3          # correlation too weak
fail_cond2 = mean_delta_cs <= 0  # no improvement

if fail_cond1:
    status = "FAIL"
    reason = f"r(DINOv2_L2,ΔCS)={r_val:.4f} < 0.3"
elif fail_cond2:
    status = "FAIL"
    reason = f"mean(ΔCS)={mean_delta_cs:.4f} ≤ 0"
elif p_val > 0.05:
    status = "INCONCLUSIVE"
    reason = f"mean(ΔCS)={mean_delta_cs:.4f}>0 but p={p_val:.4f}>0.05"
else:
    status = "CONFIRM"
    reason = f"r={r_val:.4f}>0.3, mean(ΔCS)={mean_delta_cs:.4f}>0, p={p_val:.4f}<0.05"

print(f"\n{'='*60}")
print(f"  nova_idea_b_toy RESULTS")
print(f"{'='*60}")
print(f"  N pairs:              {N_PAIRS}")
print(f"  Alpha candidates:     {ALPHA_CANDIDATES}")
print(f"  Naive alpha:          {ALPHA_NAIVE}")
print(f"")
print(f"  ΔCS per source A:    {mean_delta_A:+.4f} ± {np.std(delta_cs_A_arr):.4f}")
print(f"  ΔCS per source B:    {mean_delta_B:+.4f} ± {np.std(delta_cs_B_arr):.4f}")
print(f"  ΔCS total (primary): {mean_delta_cs:+.4f} ± {std_delta_cs:.4f}")
print(f"")
print(f"  Pearson r(DINO_L2_AB, ΔCS): {r_val:.4f}  (p={r_pval:.4e})")
print(f"  Paired t-test vs 0:         t={t_stat:.3f}  p={p_val:.4f}")
print(f"")
print(f"  Positive ΔCS pairs:  {n_positive}/{N_PAIRS}")
print(f"  Negative ΔCS pairs: {n_negative}/{N_PAIRS}")
print(f"  Zero ΔCS pairs:      {n_zero}/{N_PAIRS}")
print(f"")
print(f"  STATUS: {status}")
print(f"  REASON: {reason}")
print(f"{'='*60}")

# ── 9. Save results ──────────────────────────────────────────────────────────
results_path = '/home/kas/.openclaw/workspace-domain/research/autonomous-research-window-0422-am/nova_idea_b_results.json'

# Convert tensor values to floats for JSON serialization
serializable = {
    'experiment': 'nova_idea_b_toy',
    'hypothesis': 'Anchor-based interpolation reduces CLIP semantic drift vs naive midpoint',
    'n_pairs': N_PAIRS,
    'alpha_candidates': ALPHA_CANDIDATES,
    'alpha_naive': ALPHA_NAIVE,
    'summary': {
        'mean_delta_cs_total': mean_delta_cs,
        'std_delta_cs_total': std_delta_cs,
        'mean_delta_cs_A': mean_delta_A,
        'mean_delta_cs_B': mean_delta_B,
        'pearson_r': r_val,
        'pearson_r_pval': r_pval,
        'ttest_t': t_stat,
        'ttest_p': p_val,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_zero': n_zero,
    },
    'failure_conditions': {
        'r_threshold': 0.3,
        'mean_delta_cs_threshold': 0,
        'fail_cond1_r_too_weak': fail_cond1,
        'fail_cond2_no_improvement': fail_cond2,
    },
    'status': status,
    'reason': reason,
    'per_pair_results': [
        {
            'pair_idx': r['pair_idx'],
            'class_pair': r['class_pair'],
            'anchor_alpha': r['anchor_alpha'],
            'dino_l2_AB': r['dino_l2_AB'],
            'cs_naive_A': r['cs_naive_A'],
            'cs_naive_B': r['cs_naive_B'],
            'cs_anchor_A': r['cs_anchor_A'],
            'cs_anchor_B': r['cs_anchor_B'],
            'delta_cs_A': r['delta_cs_A'],
            'delta_cs_B': r['delta_cs_B'],
            'delta_cs_total': r['delta_cs_total'],
        }
        for r in pairwise_results
    ],
    'runtime_seconds': time.time() - t_start,
}

with open(results_path, 'w') as f:
    json.dump(serializable, f, indent=2)

print(f"\n[{time.strftime('%H:%M:%S')}] Results saved to {results_path}")
print(f"[{time.strftime('%H:%M:%S')}] Total runtime: {time.time()-t_start:.1f}s")
print(f"\n=== EXPERIMENT {'PASS' if status=='CONFIRM' else 'FAIL'} ===")
print(f"STATUS={status} | r={r_val:.4f} | mean(ΔCS)={mean_delta_cs:+.4f} | p={p_val:.4f}")
