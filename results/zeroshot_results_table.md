Measured on **Tesla T4**, Transformers 5.20.2, image tower **float16**, 10000 CIFAR-10 test images, prompt `a photo of a {}.`, run 2026-11-08T15:00:33Z.

Validation (FP32, 32 images × 10 prompts): max |Δ logit| = 1.91e-06 vs `SiglipModel(...).logits_per_image` (|logit| ≤ 18.1, tolerance atol=0.0001, rtol=1e-05), argmax agreement 100% — **passed**.
float16 vs FP32 prediction agreement at 196 tokens: 100.00% on 32 images.

| Tokens | Top-1 acc. | 95% CI | Δ vs 196 (pp) | 95% CI of Δ (paired) | 196✓→N✗ | 196✗→N✓ | Mean cos to 196 |
|---:|---:|---|---:|---|---:|---:|---:|
| 196 | 92.59% | [92.06, 93.11] | +0.00 | [+0.00, +0.00] | 0 | 0 | 1.0000 |
| 144 | 84.84% | [84.15, 85.54] | -7.75 | [-8.40, -7.11] | 962 | 187 | 0.9105 |
| 100 | 78.28% | [77.46, 79.10] | -14.31 | [-15.08, -13.56] | 1582 | 151 | 0.9146 |
| 49 | 31.28% | [30.37, 32.20] | -61.31 | [-62.31, -60.30] | 6235 | 104 | 0.7247 |

Bootstrap: 10000 paired resamples of the 10000 images, seed 42, percentile intervals.

**End-to-end zero-shot classification latency** (GPU: pixels → patch-embed all 196 → select → encoder → head → logits against cached text embeddings; excludes CPU preprocessing and the text encoder). Not comparable to the encoder-only numbers above, which start from precomputed tokens.

| Tokens | Batch | Median (ms) | p90 (ms) | Throughput (img/s) | Speed-up |
|---:|---:|---:|---:|---:|---:|
| 196 | 16 | 40.37 | 41.53 | 396 | 1.00× |
| 144 | 16 | 33.98 | 35.14 | 471 | 1.19× |
| 100 | 16 | 21.96 | 23.49 | 729 | 1.84× |
| 49 | 16 | 15.07 | 15.58 | 1062 | 2.68× |
| 196 | 64 | 186.78 | 190.32 | 343 | 1.00× |
| 144 | 64 | 138.40 | 140.81 | 462 | 1.35× |
| 100 | 64 | 95.43 | 97.65 | 671 | 1.96× |
| 49 | 64 | 53.25 | 54.72 | 1202 | 3.51× |
