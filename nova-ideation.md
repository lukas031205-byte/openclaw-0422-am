# Nova Ideation — 0422-AM

## Current State

**Active threads:**
1. CNLSA — GPU-blocked, weak confirmation (r=0.3681)
2. TrACE-Video — Workshop v4 done, GPU-blocked, needs external validity
3. Idea-B — JUST CONFIRMED r=0.7528 (p=2.89e-10) ✅
4. Re2Pix — TOP priority GPU restore, code confirmed

**0421-LATE Nova Idea-B result:** Video interpolation semantic anchor CONFIRMED.

---

## New Ideas for 0422-AM (CPU path)

### Idea-C: LCS + CLIP Bootstrap (Priority: 0.75, CPU-feasible)

**Motivation:** Idea-B has the same-model DINOv2 L2 confound (Scalpel concern). We need an independent ground truth for LCS as a semantic metric. CLIP is too slow on CPU (est. 46min for 500 pairs). What if we use CLIP on a small subset (30-50 pairs) to validate the DINOv2 signal, then extrapolate?

**Design:**
- n=50 pairs (same as Idea-B lite), compute both DINOv2 L2 AND CLIP CS
- CLIP CS(A,B) = ground truth semantic distance
- DINOv2 L2(A,B) = proxy metric
- Test: correlation(DINOv2 L2, CLIP CS) to validate DINOv2 as semantic proxy
- Then: correlation(DINOv2 L2(I,A)+DINOv2 L2(I,B), CLIP CS(A,B))

**This directly addresses the Scalpel same-model confound concern.**

**Failure condition:** r(DINOv2 L2, CLIP CS) < 0.4 → DINOv2 is not a valid CLIP proxy

**CPU constraint:** CLIP is slow (~0.4s per pair = 20min for 50 pairs). Acceptable.

**Expected runtime:** ~25-30 min for 50 pairs with CLIP.

---

### Idea-D: TrACE-Video Workshop Paper Appendix — LCS Semantic Anchor Evidence

**Motivation:** Workshop paper v4 is already done. With Idea-B confirmed (r=0.75), we can add a supplementary section: "Appendix: LCS as Semantic Anchor — Interpolation Evidence."

**Why this matters:** The workshop paper claims LCS is a valid semantic consistency metric. Idea-B provides independent evidence for this claim from a different experimental paradigm (interpolation, not VAE perturbation).

**Design:** Write the appendix section + update WINDOW_SUMMARY.md. This is a pure write-up task, no compute needed.

**Priority: 0.82** (high value, zero compute, strengthens workshop paper submission)

---

## Prioritization for Remaining Window Time (~3.5h)

Given that Idea-B is already confirmed (50 pairs, r=0.75) and 100-pair confirmation is running in background:

**Option A (high priority):** Idea-C validation — run 50 pairs with CLIP ground truth to address Scalpel same-model confound concern. ~25-30 min. Addresses the key weakness of Idea-B.

**Option B (safe):** Workshop paper appendix — write the supplementary evidence section. ~20 min. Zero compute, strengthens submission.

**Recommendation:** Do Idea-C first (~30 min), then Workshop paper appendix. If quota hits, Idea-C result + Workshop writeup is a solid deliverable.

---

## Long-term Research Trajectory

With Idea-B confirmed and TrACE-Video workshop paper ready, the research program has a clear structure:

1. **TrACE-Video Workshop Paper:** LCS as diagnostic metric for video generation consistency (workshop-ready, 7/10 acceptance)
2. **CNLSA Paper:** VAE-induced cross-modal semantic drift (needs GPU validation + real video experiments)
3. **Idea-B / Interpolation Semantic Anchor:** Validated on CIFAR-10, needs CLIP validation + higher-res test

**Next window priorities (assuming GPU restores):**
1. Re2Pix code run → test TrACE-Video metric on Re2Pix outputs
2. CNLSA SD-VAE full rerun (GPU restore)
3. TrACE-Video real video validation

**If GPU stays down:** Workshop paper submission + Idea-C (CLIP validation) + Workshop paper appendix write-up