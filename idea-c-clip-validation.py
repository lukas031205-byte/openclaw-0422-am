#!/usr/bin/env python3
"""
Idea-C: CLIP Validation for Idea-B — addressing same-model DINOv2 confound
Nova/Kernel, 0422-AM window.

Problem: Idea-B uses DINOv2 L2 on both sides — same model confound limits interpretation.
Solution: Use CLIP ViT-B/32 as ground truth semantic distance, DINOv2 as proxy.
50 pairs from CIFAR-10 different classes.

Test 1: correlation(DINOv2 L2(A,B), CLIP CS(A,B)) — validates DINOv2 as CLIP proxy
Test 2: correlation(DINOv2 L2(I,A)+LINOv2 L2(I,B), CLIP CS(A,B)) — Idea-B with CLIP ground truth

Failure: r(DINOv2, CLIP) < 0.4 → DINOv2 is NOT a valid semantic proxy
"""

import json, os, sys, time
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy import stats
from pathlib import Path

DEVICE = "cpu"
ARTIFACT_DIR = Path("/home/kas/.openclaw/workspace-domain/research/autonomous-research-window-0422-am")
ARTIFACT_DIR.mkdir(exist_ok=True)
os.chdir(ARTIFACT_DIR)

t_start = time.time()
print(f"Device: {DEVICE}")
print(f"Start: {time.strftime('%H:%M:%S')}")

# ── Load CIFAR-10 ──────────────────────────────────────────────────────────
print("\n[1/6] Loading CIFAR-10...")
dataset = torchvision.datasets.CIFAR10(
    root=str(ARTIFACT_DIR / "data"), train=False, download=False
)
print(f"  {len(dataset)} images")

# ── Load DINOv2 ─────────────────────────────────────────────────────────────
print("\n[2/6] Loading DINOv2 ViT-S/14...")
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cpu().eval()
print(f"  {sum(p.numel() for p in dinov2.parameters())/1e6:.1f}M params")

# ── Load CLIP ───────────────────────────────────────────────────────────────
print("\n[3/6] Loading CLIP ViT-B/32...")
import clip
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
print("  CLIP loaded OK")

def resize_224(img_pil):
    return img_pil.resize((224, 224), Image.BILINEAR)

def to_dino_tensor(img_pil):
    tensor = torchvision.transforms.ToTensor()(resize_224(img_pil))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std

def to_clip_tensor(img_pil):
    return clip_preprocess(resize_224(img_pil)).unsqueeze(0).to(DEVICE)

def dino_l2(img_a_pil, img_b_pil):
    with torch.no_grad():
        f_a = dinov2(to_dino_tensor(img_a_pil).unsqueeze(0))
        f_b = dinov2(to_dino_tensor(img_b_pil).unsqueeze(0))
    return torch.norm(f_a - f_b, p=2).item()

def clip_cs(img_a_pil, img_b_pil):
    with torch.no_grad():
        f_a = clip_model.encode_image(to_clip_tensor(img_a_pil))
        f_b = clip_model.encode_image(to_clip_tensor(img_b_pil))
        f_a = f_a / f_a.norm(dim=-1, keepdim=True)
        f_b = f_b / f_b.norm(dim=-1, keepdim=True)
    return (f_a @ f_b.T).item()

# ── Sample 50 pairs from DIFFERENT classes ─────────────────────────────────
N_PAIRS = 50
print(f"\n[4/6] Sampling {N_PAIRS} pairs from different classes...")
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
print(f"\n[5/6] Processing {N_PAIRS} pairs (DINOv2 + CLIP for all)...")

results = []
for i, (idx_a, idx_b, c_a, c_b) in enumerate(pairs):
    if i % 10 == 0:
        elapsed = time.time() - t_start
        print(f"  Pair {i}/{len(pairs)}... ({elapsed:.0f}s)")

    img_a = dataset[idx_a][0]
    img_b = dataset[idx_b][0]

    # Bilinear interpolation at α=0.5
    arr_a = np.array(img_a).astype(np.float32) / 255.0
    arr_b = np.array(img_b).astype(np.float32) / 255.0
    arr_i = 0.5 * arr_a + 0.5 * arr_b
    img_i = Image.fromarray((np.clip(arr_i, 0, 1) * 255).astype(np.uint8))

    # DINOv2 L2
    l2_ia = dino_l2(img_i, img_a)
    l2_ib = dino_l2(img_i, img_b)
    l2_ab = dino_l2(img_a, img_b)

    # CLIP CS (ground truth semantic similarity)
    clip_cs_ab = clip_cs(img_a, img_b)
    clip_cs_ia = clip_cs(img_i, img_a)
    clip_cs_ib = clip_cs(img_i, img_b)

    results.append({
        "pair_id": i,
        "class_a": int(c_a), "class_b": int(c_b),
        "dino_l2_ia": float(l2_ia),
        "dino_l2_ib": float(l2_ib),
        "dino_l2_ab": float(l2_ab),
        "dino_sum":   float(l2_ia + l2_ib),
        "clip_cs_ab": float(clip_cs_ab),
        "clip_cs_ia": float(clip_cs_ia),
        "clip_cs_ib": float(clip_cs_ib),
    })

# ── Compute correlations ────────────────────────────────────────────────────
print("\n[6/6] Computing correlations...")

dino_l2_ab_arr = np.array([r["dino_l2_ab"] for r in results])
dino_sum_arr   = np.array([r["dino_sum"] for r in results])
clip_cs_ab_arr = np.array([r["clip_cs_ab"] for r in results])
clip_cs_ia_arr = np.array([r["clip_cs_ia"] for r in results])
clip_cs_ib_arr = np.array([r["clip_cs_ib"] for r in results])
sem_dist_arr   = 1 - clip_cs_ab_arr  # 1 - CLIP CS = semantic distance

# Test 1: DINOv2 L2(A,B) vs CLIP CS(A,B) — validates DINOv2 as semantic proxy
r_dino_clip, p_dc = stats.pearsonr(dino_l2_ab_arr, clip_cs_ab_arr)
r_sp_dc, p_sp_dc = stats.spearmanr(dino_l2_ab_arr, clip_cs_ab_arr)
print(f"\nTest 1: DINOv2 L2(A,B) vs CLIP CS(A,B) [DINOv2 as semantic proxy]")
print(f"  Pearson r  = {r_dino_clip:.4f}, p = {p_dc:.2e}")
print(f"  Spearman ρ = {r_sp_dc:.4f}, p = {p_sp_dc:.2e}")

# Test 2: DINOv2_sum vs CLIP CS(A,B) — Idea-B with CLIP ground truth
r_sum_clip, p_sc = stats.pearsonr(dino_sum_arr, clip_cs_ab_arr)
r_sp_sc, p_sp_sc = stats.spearmanr(dino_sum_arr, clip_cs_ab_arr)
print(f"\nTest 2: DINOv2_sum vs CLIP CS(A,B) [Idea-B with CLIP ground truth]")
print(f"  Pearson r  = {r_sum_clip:.4f}, p = {p_sc:.2e}")
print(f"  Spearman ρ = {r_sp_sc:.4f}, p = {p_sp_sc:.2e}")

# Test 3: DINOv2_sum vs semantic distance (1 - CLIP CS)
r_sum_sem, p_ss = stats.pearsonr(dino_sum_arr, sem_dist_arr)
r_sp_ss, p_sp_ss = stats.spearmanr(dino_sum_arr, sem_dist_arr)
print(f"\nTest 3: DINOv2_sum vs semantic_distance [1 - CLIP CS(A,B)]")
print(f"  Pearson r  = {r_sum_sem:.4f}, p = {p_ss:.2e}")
print(f"  Spearman ρ = {r_sp_ss:.4f}, p = {p_sp_ss:.2e}")

# Interpolation quality: CLIP CS(I,A) and CLIP CS(I,B) vs source CLIP CS(A,B)
r_ia, p_ia = stats.pearsonr(clip_cs_ia_arr, clip_cs_ab_arr)
r_ib, p_ib = stats.pearsonr(clip_cs_ib_arr, clip_cs_ab_arr)
print(f"\nInterpolation quality: CLIP CS(I,A) vs CLIP CS(A,B)")
print(f"  Pearson r  = {r_ia:.4f}, p = {p_ia:.2e}")
print(f"\nInterpolation quality: CLIP CS(I,B) vs CLIP CS(A,B)")
print(f"  Pearson r  = {r_ib:.4f}, p = {p_ib:.2e}")

# CLIP CS drop
clip_cs_drop = clip_cs_ab_arr - (clip_cs_ia_arr + clip_cs_ib_arr) / 2
print(f"\nCLIP CS drop (source - interpolated avg): mean={clip_cs_drop.mean():.4f}, std={clip_cs_drop.std():.4f}")

# ── Decision ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
# Test 1: does DINOv2 L2 predict CLIP CS?
if r_dino_clip > 0.5:
    dino_valid = "VALIDATED"
    print(f"  DINOv2 semantic proxy: [{dino_valid}] r={r_dino_clip:.4f} > 0.5")
elif r_dino_clip < 0.3:
    dino_valid = "FAIL"
    print(f"  DINOv2 semantic proxy: [{dino_valid}] r={r_dino_clip:.4f} < 0.3 — DINOv2 NOT a valid CLIP proxy")
else:
    dino_valid = "WEAK"
    print(f"  DINOv2 semantic proxy: [{dino_valid}] r={r_dino_clip:.4f} in [0.3, 0.5]")

# Test 2: does Idea-B hold with CLIP ground truth?
if r_sum_clip > 0.5:
    idea_b_clip = "CONFIRM"
    print(f"  Idea-B with CLIP: [{idea_b_clip}] r={r_sum_clip:.4f} > 0.5")
elif r_sum_clip < 0.3:
    idea_b_clip = "FAIL"
    print(f"  Idea-B with CLIP: [{idea_b_clip}] r={r_sum_clip:.4f} < 0.3")
else:
    idea_b_clip = "INCONCLUSIVE"
    print(f"  Idea-B with CLIP: [{idea_b_clip}] r={r_sum_clip:.4f} in [0.3, 0.5]")

print("="*60)

t_total = time.time() - t_start
print(f"\nTotal runtime: {t_total/60:.1f} min ({t_total:.0f}s)")

# ── Save results ───────────────────────────────────────────────────────────
output = {
    "decision": idea_b_clip,
    "dino_semantic_proxy": dino_valid,
    "r_dino_clip": float(r_dino_clip),
    "p_dino_clip": float(p_dc),
    "r_spearman_dino_clip": float(r_sp_dc),
    "r_idea_b_clip": float(r_sum_clip),
    "p_idea_b_clip": float(p_sc),
    "r_spearman_idea_b_clip": float(r_sp_sc),
    "r_dino_sum_semantic_dist": float(r_sum_sem),
    "p_semantic_dist": float(p_ss),
    "r_ia": float(r_ia),
    "r_ib": float(r_ib),
    "n_pairs": N_PAIRS,
    "runtime_min": round(t_total/60, 1),
    "dino_sum_mean": float(dino_sum_arr.mean()),
    "clip_cs_ab_mean": float(clip_cs_ab_arr.mean()),
}
with open(ARTIFACT_DIR / "idea-c-results.json", "w") as f:
    json.dump(output, f, indent=2)

import csv
with open(ARTIFACT_DIR / "idea-c-results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print(f"Results → {ARTIFACT_DIR / 'idea-c-results.json'}")