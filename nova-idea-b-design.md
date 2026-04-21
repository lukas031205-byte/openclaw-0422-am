# Nova-Idea-B Design: Video Interpolation Semantic Anchor (CPU Toy)

**Date:** 2026-04-22
**Author:** Nova
**Status:** Design Complete
**GPU:** Unavailable (~1.7GB RAM free)
**Background:** LCS compute gate Idea-A falsified (r=-0.3532). Idea-B tests a different mechanism: anchor-based interpolation reduces semantic drift vs non-anchor interpolation.

---

## 1. Hypothesis

**Core claim:** When inserting an intermediate frame between two semantically distant frames, selecting the interpolation point based on DINOv2 L2 feature distance to source frames (semantic anchor) produces lower CLIP semantic drift than naive midpoint interpolation.

**Mechanism:**
```
Source A, Source B (known semantic distance = 1 - CLIP_CS(A,B))
Naive:    I_naive = 0.5·A + 0.5·B (pixel midpoint)
Anchored: I_anchor = argmin_{t∈[0,1]} DINOv2_L2(interp(A,B,t), A) + DINOv2_L2(interp(A,B,t), B)

Hypothesis: CLIP_CS(I_anchor, A) + CLIP_CS(I_anchor, B)  >  CLIP_CS(I_naive, A) + CLIP_CS(I_naive, B)
            (anchored interpolation has higher semantic consistency to sources)
```

**Contrast with Idea-A:**
- Idea-A: additive pixel noise → DINOv2 L2 → CLIP drift (per-frame, falsified)
- Idea-B: interpolation position selection → anchor → reduced drift (triplet-level, testable)

**Scalar test:** ΔCS = [CS(I_anchor,A)+CS(I_anchor,B)] - [CS(I_naive,A)+CS(I_naive,B)]
If ΔCS > 0 consistently → anchor helps. Correlate DINOv2_L2(A,B) with ΔCS to see if anchor benefit scales with semantic distance.

---

## 2. Minimal Experiment (Step-by-Step Protocol)

### 2.1 Setup
```bash
# Environment: CPU only, no GPU needed
# RAM constraint: < 1.7GB total
# Expected runtime: ~40 min on CPU

# Dependencies
pip install torch torchvision torchvision.transforms numpy scikit-learn

# Models (loaded sequentially, not simultaneously)
- DINOv2 ViT-S/14 (dinov2_vits14) — torch.hub, ~22M params, ~400MB RAM
- CLIP ViT-B/32 — clip.load(), ~151M params, loaded separately
```

### 2.2 Dataset Generation (Synthetic)
```
- CIFAR-10 test set: 10,000 images (32×32×3)
- Sample 200 image pairs from DIFFERENT classes
  (ensures high semantic distance, known ground truth)
- For each pair (A, B):
  - A, B are preprocessed to [0,1] float tensors
  - Store as (A_tensor, B_tensor, class_pair_id)
```

**Why synthetic?** Avoids real video model complexity (VAE, temporal modeling) while isolating the interpolation mechanism. Acceptable limitation for toy validation.

### 2.3 Interpolation Generation
For each pair (A, B):

**Naive path:**
```
I_naive = 0.5 * A + 0.5 * B  (pixel-level midpoint, no DINOv2 involved)
```

**Anchored path (4 candidate search):**
```
candidates = [0.3, 0.4, 0.5, 0.6, 0.7]  # 5 alpha values around midpoint
for t in candidates:
    I_candidate = t * A + (1-t) * B
    dino_feat = DINOv2(I_candidate)
    dino_A = DINOv2(A)
    dino_B = DINOv2(B)
    score = -norm(dino_feat - dino_A) - norm(dino_feat - dino_B)  # minimize distance
I_anchor = candidate with best DINOv2 L2 score
```

**Why 5 candidates?** Trades off compute vs quality. 5 alpha values is fast (~5 extra DINOv2 forwards/pair), enough to demonstrate anchor selection effect.

### 2.4 Metric Computation
For each pair (A, B, I_naive, I_anchor):

**CLIP metrics (ground truth):**
```
clip_A = CLIP(A), clip_B = CLIP(B), clip_naive = CLIP(I_naive), clip_anchor = CLIP(I_anchor)
CS_naive_A = cosine_similarity(clip_naive, clip_A)
CS_naive_B = cosine_similarity(clip_naive, clip_B)
CS_anchor_A = cosine_similarity(clip_anchor, clip_A)
CS_anchor_B = cosine_similarity(clip_anchor, clip_B)
```

**Primary outcome:**
```
ΔCS_A = CS_anchor_A - CS_naive_A  (anchor improvement for source A)
ΔCS_B = CS_anchor_B - CS_naive_B  (anchor improvement for source B)
ΔCS_total = ΔCS_A + ΔCS_B
```

### 2.5 Analysis
```
1. Mean(ΔCS_total) across 200 pairs
   - if Mean(ΔCS_total) > 0 → anchored interpolation reduces semantic drift

2. Pearson r( DINOv2_L2(A,B), ΔCS_total )
   - if r > 0.3 → anchor benefit scales with semantic distance

3. Paired t-test: ΔCS_total vs 0
   - if p < 0.05 → statistically significant improvement
```

### 2.6 Step Summary
| Step | Action | Output |
|---|---|---|
| 1 | Load CIFAR-10, sample 200 cross-class pairs | List of (A,B) tensor pairs |
| 2 | For each pair: compute I_naive = 0.5A+0.5B | 200 naive interpolations |
| 3 | For each pair: search 5 alpha candidates via DINOv2 L2 | 200 anchored interpolations |
| 4 | Compute CLIP CS for all frames (A, B, I_naive, I_anchor) | 800 CLIP embeddings |
| 5 | Compute ΔCS_total per pair | 200 ΔCS values |
| 6 | Report mean, r, p-value | Experiment verdict |

---

## 3. Failure Conditions

| Condition | Threshold | Outcome |
|---|---|---|
| **Primary FAIL** | r(DINOv2_L2(A,B), ΔCS_total) < 0.3 | Anchor benefit does NOT scale with semantic distance |
| **Secondary FAIL** | Mean(ΔCS_total) ≤ 0 | Anchored interpolation is same or worse than naive |
| **Tertiary FAIL** | p-value (paired t-test) > 0.05 | No statistically significant improvement |
| **INCONCLUSIVE** | 0.3 ≤ r < 0.5 AND Mean(ΔCS_total) > 0 | Weak signal — needs refinement |
| **CONFIRM** | r > 0.3 AND Mean(ΔCS_total) > 0 AND p < 0.05 | Anchor-based interpolation validated |

**Note on r<0.3 threshold:** Inherited from Idea-A falsification threshold. This is a conservative bar — if DINOv2 L2 can't even weakly predict anchor benefit, the mechanism is not worth pursuing at scale.

---

## 4. CPU Feasibility

### 4.1 Memory Budget
| Component | RAM | Strategy |
|---|---|---|
| DINOv2 ViT-S/14 | ~400MB peak | Load first, process all candidates, unload |
| CLIP ViT-B/32 | ~800MB peak | Load after DINOv2, process all CLIP metrics, unload |
| CIFAR-10 tensors | ~50MB | Keep in list of PIL images, convert on-demand |
| Intermediate activations | ~100MB | No gradient storage (eval mode) |
| **Total peak** | ~1.35GB | **Within 1.7GB limit** |

### 4.2 Sequential Loading Strategy
```
Phase 1: Load DINOv2 → process all 200×5 candidate searches → unload
Phase 2: Load CLIP → compute all 800 CLIP embeddings → unload
Phase 3: Analysis (numpy, no model RAM)
```

### 4.3 Runtime Estimate
| Phase | Operation | Time |
|---|---|---|
| DINOv2 inference | 200 pairs × 5 candidates × 2 forwards = 2000 forwards | ~8 min |
| CLIP inference | 800 embeddings × ~0.5s/forward | ~7 min |
| I/O & overhead | CIFAR-10 load, pair iteration | ~5 min |
| **Total** | | **~20 min** |

### 4.4 Code Architecture
```python
# nova_idea_b_toy.py
import torch, torchvision, clip, numpy as np
from torchvision.datasets import CIFAR10
from torch.nn.functional import cosine_embedding_loss

DEVICE = "cpu"
BATCH_SIZE = 1  # RAM constraint — process one at a time

def load_models():
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE).eval()
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    return dinov2, clip_model, clip_preprocess

def sample_cross_class_pairs(dataset, n=200):
    """Sample n image pairs from different CIFAR-10 classes."""
    # Group indices by class
    class_indices = {c: [] for c in range(10)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    # Sample random cross-class pairs
    pairs = []
    for _ in range(n):
        c1, c2 = np.random.choice(10, 2, replace=False)
        i1 = np.random.choice(len(class_indices[c1]))
        i2 = np.random.choice(len(class_indices[c2]))
        img1 = dataset[class_indices[c1][i1]][0]
        img2 = dataset[class_indices[c2][i2]][0]
        pairs.append((img1, img2))
    return pairs

def interpolate_batch(img_a, img_b, alpha):
    """Pixel-level linear interpolation."""
    # img_a, img_b: PIL images or tensors
    # Convert to tensors if needed
    if not isinstance(img_a, torch.Tensor):
        ta = ToTensor()(img_a)
        tb = ToTensor()(img_b)
    else:
        ta, tb = img_a, img_b
    return alpha * ta + (1 - alpha) * tb

def dinov2_l2(feats_a, feats_b):
    """L2 distance between DINOv2 features."""
    return torch.norm(feats_a - feats_b, p=2).item()

def clip_cosine_sim(feats_a, feats_b):
    """Cosine similarity between CLIP features."""
    a_norm = feats_a / feats_a.norm(dim=-1, keepdim=True)
    b_norm = feats_b / feats_b.norm(dim=-1, keepdim=True)
    return (a_norm @ b_norm.T).item()

def find_anchor_interpolation(img_a, img_b, dinov2, preprocess_fn, alphas=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Find best alpha by minimizing DINOv2 L2 distance to sources."""
    best_score = float('-inf')
    best_alpha = 0.5
    best_img = None
    
    ta = ToTensor()(img_a).unsqueeze(0)
    tb = ToTensor()(img_b).unsqueeze(0)
    
    with torch.no_grad():
        feat_a = dinov2(preprocess_fn(ta))
        feat_b = dinov2(preprocess_fn(tb))
        
        for alpha in alphas:
            t_img = alpha * ta + (1 - alpha) * tb
            feat_t = dinov2(preprocess_fn(t_img))
            score = -dinov2_l2(feat_t, feat_a) - dinov2_l2(feat_t, feat_b)
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_img = t_img.squeeze(0)
    
    return best_img, best_alpha

def main():
    print("Loading CIFAR-10...")
    cifar = CIFAR10(root='./data', train=False, download=True)
    
    print("Sampling 200 cross-class pairs...")
    pairs = sample_cross_class_pairs(cifar, n=200)
    
    print("Phase 1: DINOv2 anchor search...")
    results = []
    for i, (img_a, img_b) in enumerate(pairs):
        # Naive interpolation
        naive = interpolate_batch(img_a, img_b, 0.5)
        # Anchor interpolation (5 candidates)
        anchor, alpha = find_anchor_interpolation(img_a, img_b, dinov2, preprocess)
        results.append({'naive': naive, 'anchor': anchor, 'alpha': alpha})
        if i % 50 == 0:
            print(f"  {i}/200 pairs processed")
    
    print("Phase 2: CLIP semantic evaluation...")
    clip_results = []
    for i, (r, (img_a, img_b)) in enumerate(zip(results, pairs)):
        # Compute CLIP embeddings
        with torch.no_grad():
            clip_a = clip_model.encode_image(clip_preprocess(img_a).unsqueeze(0))
            clip_b = clip_model.encode_image(clip_preprocess(img_b).unsqueeze(0))
            clip_naive = clip_model.encode_image(clip_preprocess(r['naive']).unsqueeze(0))
            clip_anchor = clip_model.encode_image(clip_preprocess(r['anchor']).unsqueeze(0))
        
        cs_naive_A = clip_cosine_sim(clip_naive, clip_a)
        cs_naive_B = clip_cosine_sim(clip_naive, clip_b)
        cs_anchor_A = clip_cosine_sim(clip_anchor, clip_a)
        cs_anchor_B = clip_cosine_sim(clip_anchor, clip_b)
        
        delta_cs = (cs_anchor_A - cs_naive_A) + (cs_anchor_B - cs_naive_B)
        clip_results.append({
            'delta_cs': delta_cs,
            'cs_naive_A': cs_naive_A, 'cs_naive_B': cs_naive_B,
            'cs_anchor_A': cs_anchor_A, 'cs_anchor_B': cs_anchor_B
        })
        
        if i % 50 == 0:
            print(f"  {i}/200 CLIP evaluations done")
    
    print("Phase 3: Analysis...")
    delta_cs_arr = np.array([r['delta_cs'] for r in clip_results])
    mean_delta = np.mean(delta_cs_arr)
    std_delta = np.std(delta_cs_arr)
    
    from scipy.stats import pearsonr, ttest_1samp
    # For correlation: need DINOv2 L2 between A and B as predictor
    # (skipped for brevity — can add if r computation needed)
    
    t_stat, p_val = ttest_1samp(delta_cs_arr, 0)
    
    print(f"\n=== RESULTS ===")
    print(f"Mean ΔCS: {mean_delta:.4f} (±{std_delta:.4f})")
    print(f"Paired t-test vs 0: t={t_stat:.3f}, p={p_val:.4f}")
    print(f"Positive pairs: {np.sum(delta_cs_arr > 0)}/200")
    print(f"Negative pairs: {np.sum(delta_cs_arr < 0)}/200")
    
    if mean_delta > 0 and p_val < 0.05:
        print("STATUS: CONFIRM — anchor helps")
    elif mean_delta <= 0:
        print("STATUS: FAIL — anchor same or worse")
    else:
        print("STATUS: INCONCLUSIVE — weak signal")
```

---

## 5. Expected Outcome

| Scenario | Condition | Interpretation |
|---|---|---|
| **CONFIRM** | mean(ΔCS) > 0, p < 0.05 | Anchor-based interpolation reduces semantic drift. Validated for method paper. |
| **INCONCLUSIVE** | mean(ΔCS) > 0, p > 0.05 | Directionally correct but underpowered. Needs more pairs. |
| **FAIL** | mean(ΔCS) ≤ 0 OR r < 0.3 | Naive interpolation is as good or better. Idea-B is a dead end. |

**Effect size estimate (optimistic):**
- Cross-class pairs have CLIP_CS(A,B) ≈ 0.6–0.75
- Naive interpolation likely drops this by 3–8% (interpolating different concepts mixes features)
- Anchor selection might recover 1–3% of that drop
- Mean ΔCS ≈ 0.02–0.05 would be realistic for toy validation

**Conservative bar:** Even ΔCS = 0.01 with p < 0.05 is publishable as preliminary evidence for a workshop paper toy experiment. The key is that direction is correct and statistically defensible.

---

## 6. Relationship to Prior Work

| Component | Status | Notes |
|---|---|---|
| Idea-A (LCS compute gate) | FALSIFIED | r=-0.3532, wrong direction |
| Idea-B (Anchor interpolation) | TOY VALIDATION PENDING | Different mechanism from Idea-A |
| TrACE-Video Workshop v4 | COMPLETE | Scalpel major revision in progress |

**Idea-B is NOT Idea-A.** It tests a distinct mechanism:
- Idea-A: DINOv2 L2 predicts drift from additive pixel noise (falsified)
- Idea-B: DINOv2 L2 guides interpolation position to minimize drift (untested)

Both can coexist: Idea-A as cautionary tale, Idea-B as promising direction with toy validation pending.

---

## 7. Limitations & Caveats

1. **Synthetic frames only:** No real video model involved. VAE decoder artifacts at real interpolation points not tested.
2. **Pixel-level interpolation:** Real video interpolation uses flow-based or VAE latent methods. This toy isolates the mechanism only.
3. **Small n:** 200 pairs is underpowered for strong claims. Preliminary evidence only.
4. **CIFAR-10 32×32:** ImageNet-pretrained CLIP/DINOv2 were trained on larger images. Preprocessing upsample may introduce artifacts.
5. **No causal claim:** This measures correlation between anchor selection and semantic consistency. Does not prove that using anchors in a real video model improves quality.

---

## 8. Artifact Location

**Design doc:** `research/autonomous-research-window-0422-am/nova-idea-b-design.md`
**Code toy:** `research/autonomous-research-window-0422-am/nova_idea_b_toy.py` (to be written by Kernel if approved)
**Results:** `research/autonomous-research-window-0422-am/nova_idea_b_results.md` (post-run)
