#!/usr/bin/env python3
"""
Idea-B: Video Interpolation Semantic Anchor
Kernel implementation for 0422-AM window.
Hypothesis: DINOv2 L2 on interpolated frames predicts CLIP semantic drift.
Failure: r < 0.3 | Inconclusive: r ∈ [0.3, 0.5] | Confirm: r > 0.5
CPU-feasible: DINOv2 ViT-S/14 (22M) + CLIP ViT-B/32, batch=1, RAM < 1.5GB
"""

import json, os, sys
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

print(f"Device: {DEVICE}")
print(f"CWD: {os.getcwd()}")

# ── Load CIFAR-10 ──────────────────────────────────────────────────────────
print("\n[1/6] Loading CIFAR-10 test set...")
dataset = torchvision.datasets.CIFAR10(
    root=str(ARTIFACT_DIR / "data"), train=False, download=True
)
print(f"  {len(dataset)} images loaded")

# ── Load DINOv2 ─────────────────────────────────────────────────────────────
print("\n[2/6] Loading DINOv2 ViT-S/14...")
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEVICE).eval()
print(f"  {sum(p.numel() for p in dinov2.parameters())/1e6:.1f}M params")

# ── Load CLIP ───────────────────────────────────────────────────────────────
print("\n[3/6] Loading CLIP ViT-B/32...")
try:
    import clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    print("  CLIP loaded OK")
    HAS_CLIP = True
except Exception as e:
    print(f"  CLIP load failed: {e}")
    HAS_CLIP = False

# ── Helper: to_tensor for DINOv2 ───────────────────────────────────────────
def resize_to_dino(img_pil):
    """Resize PIL Image to 224x224 (DINOv2 ViT-S/14 requires H,W multiple of 14)."""
    return img_pil.resize((224, 224), Image.BILINEAR)

def dinov2_tensor(pil_or_arr):
    """Convert PIL Image or np.array [0,255] → DINOv2 input tensor (224x224)."""
    if isinstance(pil_or_arr, np.ndarray):
        pil = Image.fromarray(pil_or_arr.astype(np.uint8))
    else:
        pil = pil_or_arr
    pil = resize_to_dino(pil)
    tensor = torchvision.transforms.ToTensor()(pil)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)

def clip_tensor(pil_or_arr):
    """Convert PIL Image or np.array → CLIP preprocessed tensor (224x224)."""
    if isinstance(pil_or_arr, np.ndarray):
        pil = Image.fromarray(pil_or_arr.astype(np.uint8))
    else:
        pil = pil_or_arr
    pil = resize_to_dino(pil)
    return clip_preprocess(pil).unsqueeze(0).to(DEVICE)

# ── DINOv2 L2 distance ──────────────────────────────────────────────────────
def dino_l2(img_a, img_b):
    with torch.no_grad():
        feat_a = dinov2(dinov2_tensor(img_a))
        feat_b = dinov2(dinov2_tensor(img_b))
    return torch.norm(feat_a - feat_b, p=2).item()

# ── CLIP cosine similarity ──────────────────────────────────────────────────
def clip_cs(img_a, img_b):
    if not HAS_CLIP:
        return None
    with torch.no_grad():
        feat_a = clip_model.encode_image(clip_tensor(img_a))
        feat_b = clip_model.encode_image(clip_tensor(img_b))
        feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
        feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
    return (feat_a @ feat_b.T).item()

# ── Sample 500 pairs from DIFFERENT classes ─────────────────────────────────
print("\n[4/6] Sampling 500 pairs from different classes...")
np.random.seed(42)
torch.manual_seed(42)

class_images = {c: [] for c in range(10)}
for idx, (img, label) in enumerate(dataset):
    class_images[label].append((idx, img))

pairs = []
attempts = 0
while len(pairs) < 500 and attempts < 100000:
    attempts += 1
    c1, c2 = np.random.choice(10, 2, replace=False)
    if len(class_images[c1]) == 0 or len(class_images[c2]) == 0:
        continue
    idx1 = np.random.randint(len(class_images[c1]))
    idx2 = np.random.randint(len(class_images[c2]))
    pairs.append((class_images[c1][idx1][0], class_images[c2][idx2][0], c1, c2))

print(f"  Sampled {len(pairs)} pairs")

# ── Run experiment ──────────────────────────────────────────────────────────
print("\n[5/6] Processing pairs (DINOv2 all 500, CLIP subset every 5th)...")

results = []
for i, (idx_a, idx_b, c_a, c_b) in enumerate(pairs):
    if i % 100 == 0:
        print(f"  Pair {i}/{len(pairs)}...")

    img_a = dataset[idx_a][0]  # PIL
    img_b = dataset[idx_b][0]  # PIL

    # Bilinear interpolation at α=0.5 → resize to 224x224 for model input
    arr_a = np.array(img_a).astype(np.float32) / 255.0
    arr_b = np.array(img_b).astype(np.float32) / 255.0
    arr_i = 0.5 * arr_a + 0.5 * arr_b
    img_i = Image.fromarray((np.clip(arr_i, 0, 1) * 255).astype(np.uint8))
    # Resize to 224x224 (DINOv2 requires H,W multiple of 14; CLIP uses 224x224)
    img_a_r = resize_to_dino(img_a)
    img_b_r = resize_to_dino(img_b)
    img_i_r = resize_to_dino(img_i)

    # DINOv2 distances (all 500) — use resized 224x224 images
    l2_ia = dino_l2(img_i_r, img_a_r)
    l2_ib = dino_l2(img_i_r, img_b_r)
    l2_ab = dino_l2(img_a_r, img_b_r)
    l2_sum = l2_ia + l2_ib

    # CLIP metrics (every 5th pair → ~100 pairs) — use resized images
    if HAS_CLIP and i % 5 == 0:
        cs_ab = clip_cs(img_a_r, img_b_r)
        cs_ia = clip_cs(img_i_r, img_a_r)
        cs_ib = clip_cs(img_i_r, img_b_r)
        sem_dist = 1 - cs_ab if cs_ab is not None else None
    else:
        cs_ab = cs_ia = cs_ib = sem_dist = None

    results.append({
        "pair_id": i,
        "class_a": int(c_a), "class_b": int(c_b),
        "dino_l2_ia": float(l2_ia),
        "dino_l2_ib": float(l2_ib),
        "dino_l2_ab": float(l2_ab),
        "dino_sum":    float(l2_sum),
        "clip_cs_ab":  float(cs_ab) if cs_ab is not None else None,
        "clip_cs_ia":  float(cs_ia) if cs_ia is not None else None,
        "clip_cs_ib":  float(cs_ib) if cs_ib is not None else None,
        "sem_dist":    float(sem_dist) if sem_dist is not None else None,
    })

print(f"  Done. {len(results)} pairs processed.")

# ── Compute correlations ────────────────────────────────────────────────────
print("\n[6/6] Computing correlations...")

all_dino_sum   = np.array([r["dino_sum"] for r in results])
all_dino_l2_ab = np.array([r["dino_l2_ab"] for r in results])

clip_results = [(r["dino_sum"], r["sem_dist"]) for r in results if r["sem_dist"] is not None]
n_clip = len(clip_results)
print(f"  CLIP subset: {n_clip} pairs")

if n_clip >= 30:
    d_arr = np.array([p[0] for p in clip_results])
    s_arr = np.array([p[1] for p in clip_results])

    r_pearson, p_pearson = stats.pearsonr(d_arr, s_arr)
    r_spearman, p_spearman = stats.spearmanr(d_arr, s_arr)

    print(f"\n  PRIMARY: DINOv2_sum vs semantic_distance (n={n_clip})")
    print(f"    Pearson r  = {r_pearson:.4f}, p = {p_pearson:.2e}")
    print(f"    Spearman ρ = {r_spearman:.4f}, p = {p_spearman:.2e}")

    # DINOv2 L2(A,B) vs semantic distance
    l2_ab_clip = np.array([results[i*5]["dino_l2_ab"] for i in range(n_clip)])
    r2, p2 = stats.pearsonr(l2_ab_clip, s_arr)
    print(f"\n  SECONDARY: DINOv2 L2(A,B) vs semantic_distance (n={n_clip})")
    print(f"    Pearson r = {r2:.4f}, p = {p2:.2e}")

    # DINOv2 sum vs DINOv2 L2(A,B) correlation (all 500)
    r_all, p_all = stats.pearsonr(all_dino_sum, all_dino_l2_ab)
    print(f"\n  SANITY: DINOv2_sum vs L2(A,B) (n=500)")
    print(f"    Pearson r = {r_all:.4f}, p = {p_all:.2e}")

    print(f"\n  DINOv2_sum stats (n=500): mean={all_dino_sum.mean():.2f}, std={all_dino_sum.std():.2f}")
else:
    r_pearson = p_pearson = r_spearman = p_spearman = None
    r2 = p2 = r_all = p_all = None
    print(f"  CLIP subset too small ({n_clip} < 30), skipping correlation")

# ── Decision ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if r_pearson is not None:
    if r_pearson > 0.5:
        decision = "CONFIRM"
        print(f"  [{decision}] r={r_pearson:.4f} > 0.5 — semantic anchor validated")
    elif r_pearson < 0.3:
        decision = "FAIL"
        print(f"  [{decision}] r={r_pearson:.4f} < 0.3 — DINOv2 does NOT detect interpolation drift")
    else:
        decision = "INCONCLUSIVE"
        print(f"  [{decision}] r={r_pearson:.4f} in [0.3, 0.5] — weak signal, needs refinement")
else:
    decision = "CLIP_UNAVAILABLE"
    print(f"  [{decision}] CLIP not available, using DINOv2 proxy only")

print("="*60)

# ── Save results ───────────────────────────────────────────────────────────
output = {
    "decision": decision,
    "r_pearson": float(r_pearson) if r_pearson is not None else None,
    "p_pearson": float(p_pearson) if p_pearson is not None else None,
    "r_spearman": float(r_spearman) if r_spearman is not None else None,
    "p_spearman": float(p_spearman) if p_spearman is not None else None,
    "r_secondary": float(r2) if r2 is not None else None,
    "p_secondary": float(p2) if p2 is not None else None,
    "r_all": float(r_all) if r_all is not None else None,
    "p_all": float(p_all) if p_all is not None else None,
    "n_clip_pairs": n_clip,
    "n_total_pairs": len(results),
    "dino_sum_mean": float(all_dino_sum.mean()),
    "dino_sum_std": float(all_dino_sum.std()),
}
with open(ARTIFACT_DIR / "idea-b-results.json", "w") as f:
    json.dump(output, f, indent=2)

import csv
with open(ARTIFACT_DIR / "idea-b-results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults → {ARTIFACT_DIR / 'idea-b-results.json'}")
print(f"CSV     → {ARTIFACT_DIR / 'idea-b-results.csv'}")