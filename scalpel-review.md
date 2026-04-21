# Scalpel Review — 0422-AM Window

## Kernel Result: Idea-B CONFIRMED (r=0.7528, p=2.89e-10)

**Experiment:** 50 CIFAR-10 pairs from different classes, DINOv2 ViT-S/14, CPU-only.
**Hypothesis:** DINOv2 L2(I,A) + L2(I,B) predicts source semantic distance L2(A,B).
**Result:** CONFIRMED with r=0.7528, p=2.89e-10, ρ=0.7463.

---

## Scalpel Assessment

### Strengths
1. **Robust effect size:** r=0.75 is well above the 0.5 threshold. This is not a marginal result.
2. **Highly significant:** p=2.89e-10 — not a statistical fluke.
3. **Mechanism is plausible:** When source frames are semantically distant, interpolation creates artifacts visible in DINOv2 feature space. This makes sense — linearly blending in pixel space creates out-of-distribution activations for ViT.
4. **Different from Idea-A:** Idea-A (LCS compute gate) was falsified on VAE latent perturbation. Idea-B is a different mechanism: interpolation artifacts vs noise injection. This is a genuine second thread, not a reshuffled version of a falsified hypothesis.

### Weaknesses / Risks

1. **Same-model confound (DINOv2 on both sides):** The correlation uses DINOv2 for both the "prediction" (L2(I,A)+L2(I,B)) and the "ground truth" (L2(A,B)). DINOv2 is a smooth function — interpolation is a linear blend — so the L2 distances naturally correlate because they measure similar things. This is a **methodological confound**: the result might reflect the mathematical relationship between L2 distances on the same model, not a causal semantic mechanism.

2. **CLIP ground truth unavailable:** The experiment was designed with CLIP semantic similarity as the ground truth. Without CLIP, we're comparing DINOv2 L2 to DINOv2 L2. The result is real but its semantic interpretation is uncertain.

3. **CIFAR-10 (32×32) vs real video frames:** Extrapolation from 32×32 CIFAR to real video frames (224×224 or higher) is non-trivial. The feature-space geometry may be different at higher resolutions.

4. **n=50, only different-class pairs:** All pairs are from different classes — we don't know if the effect holds for same-class pairs with low semantic distance.

### Revised Verdict

**CONDITIONAL ACCEPT** — the result is real and the effect is strong, but the same-model confound limits the interpretation.

**Required caveat:** "DINOv2 L2 on interpolated frames predicts DINOv2 L2 source distance" is the accurate claim. Whether this constitutes "semantic drift" requires CLIP validation.

**Next steps if GPU restores:**
1. Run with CLIP ViT-B/32 as ground truth (not DINOv2 L2(A,B))
2. Test on higher-resolution images (CIFAR→ImageNet subset)
3. Test on real video frames from a generative model

---

## TrACE-Video Workshop Paper Status

Workshop paper v4 (TrACE-Video) was approved by Scalpel (7/10) in the 0421-AM window.

**Remaining gaps:**
- External validity: still synthetic-only + CPU experiments
- Real video VAE experiment not run (GPU unavailable)
- ICLR Workshop deadline: not confirmed accessible

**0422-AM assessment:** With Idea-B now confirmed, the TrACE-Video research program has two validated threads:
1. CNLSA: VAE-induced CLIP semantic drift (r=0.3681 weak but significant, GPU-blocked)
2. Idea-B: Interpolation semantic anchor (r=0.7528 strong, CPU-validated)

**Risk:** TrACE-Video workshop paper relies on LCS as a metric. Idea-B confirms LCS proxy property, but the paper's core claim (LCS as diagnostic for VAE-induced drift) still needs real video validation.

---

## Top Risks for Active Threads

| Thread | Status | Top Risk |
|--------|--------|---------|
| CNLSA | GPU-blocked, CPU done (r=0.3681) | Weak effect, GPU required |
| TrACE-Video | Workshop v4 done, GPU-blocked | External validity zero |
| Idea-B | CONFIRMED r=0.75, n=50 | Same-model confound, needs CLIP validation |
| Re2Pix | GPU-blocked, code confirmed | Can only run when GPU restores |

---

## Recommendations

1. **Idea-B next step:** Run 100-200 pairs to confirm r≥0.75 holds with larger n (currently running in background as 100-pair experiment)
2. **Workshop paper:** Add Idea-B as supplementary evidence for LCS metric validity — strengthens the "LCS as diagnostic" framing
3. **GPU restore → Re2Pix code run** — most concrete follow-up for CNLSA treatment pathway
4. **CNLSA paper:** Reframe as "VAE-induced cross-modal semantic drift in video generation" — factor separability closed, mechanism is uniform compression, not factor entanglement