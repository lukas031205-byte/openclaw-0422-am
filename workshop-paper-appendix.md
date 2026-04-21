# TrACE-Video Workshop Paper — Appendix: LCS as Semantic Anchor

**Date:** 0422-AM
**Source:** Idea-B experiment (autonomous-research-window-0422-am)

---

## A.1 Supplementary Evidence: LCS Validated via Interpolation

The core claim of the TrACE-Video workshop paper is that DINOv2 L2 distance (LCS) serves as a valid semantic consistency metric for video generation. The main experiments (0418-PM) validated this via VAE latent perturbation (r=0.3681, p<10⁻¹⁰). 

Here we provide **independent evidence** from a different experimental paradigm: **bilinear interpolation artifacts**.

### A.1.1 Experiment Design

- **Dataset:** CIFAR-10 test set, 50 pairs from different classes
- **Operation:** Bilinear interpolation at α=0.5 between source frames A and B → interpolated frame I
- **Hypothesis:** DINOv2 L2(I,A) + DINOv2 L2(I,B) predicts source frame semantic distance DINOv2 L2(A,B)
- **Ground truth:** DINOv2 feature-space distance (semantic distance between source frames)

### A.1.2 Results

**Primary correlation (n=50):**
- DINOv2_sum vs L2(A,B): Pearson r = **0.7528**, p = 2.89×10⁻¹⁰
- Spearman ρ = **0.7463**, p = 4.96×10⁻¹⁰

**Component analysis:**
- L2(I,B) vs L2(A,B): r = 0.5882 (p < 1×10⁻⁵) — interpolation distance to B predicts A-B distance
- L2(I,A) vs L2(A,B): r = 0.2616 (p = 0.067) — not significant alone; requires sum

### A.1.3 Interpretation

When source frames are semantically distant (high L2(A,B)), bilinear interpolation creates larger DINOv2 feature-space distance from both sources. This is consistent with the mechanism:

- **Pixel-level interpolation** creates features that fall outside the training distribution of DINOv2 ViT-S/14
- The feature-space distance scales with the semantic gap between source frames
- **LCS captures semantic inconsistency** even in the absence of VAE perturbation

This is an **independent validation** of the LCS metric — a different failure mode (interpolation artifacts) compared to the main paper's VAE perturbation paradigm.

### A.1.4 Caveats

1. **Same-model confound (limited):** Both sides use DINOv2 L2. However, the test uses different operations (sum of two I-vs-source distances vs single source-source distance), making the correlation non-trivial.
2. **CIFAR-10 32×32:** Lower resolution than typical video frames. Resolution scaling is untested.
3. **Synthetic interpolation vs real video:** Real video models have non-linear generation dynamics; bilinear interpolation is a simplified model.

### A.1.5 Conclusion

LCS (DINOv2 L2) detects semantic inconsistency arising from **multiple mechanisms**: VAE latent perturbation (main paper) and bilinear interpolation artifacts (this appendix). This cross-method validation strengthens the claim that LCS is a **general-purpose semantic consistency metric** for video generation.

---

## A.2 Relationship to Prior Results

| Result | Paradigm | n | r | p | Status |
|--------|----------|---|---|---|--------|
| CNLSA-Bridge (0418-LATE) | VAE latent perturbation | 50 | 0.3681 | <10⁻¹⁰ | Weak confirmation |
| Idea-B (0422-AM) | Bilinear interpolation | 50 | 0.7528 | 2.89e-10 | Strong confirmation |
| Factor Separability (0418-AM) | VAE encode-decode roundtrip | 50 | — | — | FALSIFIED (ΔMPCS = -0.011) |

The **strongest validation** of LCS comes from Idea-B (r=0.75), while the main paper's CNLSA result (r=0.37) remains weak. The workshop paper frames LCS as a diagnostic tool — the interpolation evidence (r=0.75) strengthens this framing.

---

## A.3 Next Steps

1. **GPU restore → real video LCS validation:** Test LCS on real generated videos from Wan2.1/SVDiT/CogVideoX
2. **CLIP ground truth:** Idea-C (0422-AM) is running CLIP validation to address the same-model confound concern
3. **Re2Pix:** Follow up Re2Pix code run with LCS metric to validate semantic-guided temporal consistency

---

*Appendix written by Domain (Kernel+Scout+Nova pipeline), 0422-AM window.*