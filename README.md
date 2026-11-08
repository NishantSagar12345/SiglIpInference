efficient-vit-bench

How much faster does a pretrained Vision Transformer get if you feed it fewer patch tokens, and what does it cost in accuracy?

This repo takes google/siglip-base-patch16-224, drops patch tokens after the embedding layer (196 → 144 → 100 → 49), and measures latency, throughput, GPU memory and zero-shot CIFAR-10 accuracy on a Kaggle Tesla T4. There is no retraining and no change to the model's weights or layers.

Key findings
At small batches, batch size matters more than token count. At batch 1, every configuration took about 5.3–5.5 ms; cutting tokens bought nothing because the GPU wasn't busy enough. At batch 16, 100 tokens ran 1.94× faster than 196 (33.88 → 17.43 ms, encoder only).
Speed-ups stay below the FLOP ratio. 100 tokens is half the FLOPs (2.00×) and got 1.94× at batch 16. 49 tokens is 4.12× fewer FLOPs and got 3.05×.
Attention isn't the bottleneck at this length. The quadratic attention term is only about 4% of the compute at 196 tokens. Almost all the work is in the linear layers, so reduction saves roughly linearly, not quadratically.
Accuracy drops quickly. Zero-shot accuracy on the 10,000-image CIFAR-10 test set went from 92.59% (196 tokens) to 84.84% (144), 78.28% (100) and 31.28% (49).
Embedding similarity is a poor proxy for accuracy. At 100 tokens the pooled embeddings still had 0.915 cosine similarity to the full-token ones, yet accuracy fell by 14.31 points. 144 tokens had slightly lower similarity (0.911) but higher accuracy.
Why I did this

Vision-language and vision-language-action models turn every camera frame into hundreds of tokens. With several cameras and a history of frames, those tokens fill most of the context window, so reducing them is an obvious way to make these models cheaper. Before trying anything clever, I wanted to know what a simple, fixed reduction does on real hardware for a standard pretrained ViT, and how well FLOP counts predict it.

This is a single-image benchmark of one component. It isn't a long-context or VLA system.

How SigLIP's vision tower works

I read the installed modeling_siglip.py before changing anything. SigLIP-Base/16 at 224 px (~93M parameters) has four parts:

SiglipVisionEmbeddings: a 16×16, stride-16 convolution turns the image into a 14×14 grid of patches, flattened to (B, 196, 768). A learned position embedding is added per patch. There's no CLS token.
SiglipEncoder: 12 pre-LayerNorm transformer layers (12 heads, MLP width 3072). Images use no attention mask, and nothing depends on the sequence being 196 long.
post_layernorm: applied per token.
SiglipMultiheadAttentionPoolingHead: a learned query attends over all output tokens, then LayerNorm and an MLP, giving the (B, 768) image embedding.

The public forward always produces 196 tokens. But since positions are added inside the embedding layer, and the encoder and pooling head accept any length, you can cut the sequence between the embedding layer and the encoder without touching any module. In Transformers 4.x these modules sit under SiglipVisionModel.vision_model; in 5.x they're on the model itself. resolve_vision_core in src/model_wrapper.py handles both.

Token reduction
pixel_values (B,3,224,224)
   └─ SiglipVisionEmbeddings   → (B,196,768)  patches + position embeddings
        └─ index_select(1, idx) → (B,N,768)    N ∈ {196,144,100,49}
             └─ SiglipEncoder → post_layernorm → pooling head → (B,768)

For a k×k subset I keep rows and columns at round(linspace(0, 13, k)) and take every combination, in row-major order. The same indices are used for every image. Because the spacing is always more than 1 for k < 14, rounding can't produce duplicates; the code asserts this anyway.

Configuration	Grid	Tokens	Kept	Row/column positions (0–13)
Baseline	14×14	196	100%	0 1 2 3 4 5 6 7 8 9 10 11 12 13
Reduced	12×12	144	73.5%	0 1 2 4 5 6 7 8 9 11 12 13
Reduced	10×10	100	51.0%	0 1 3 4 6 7 9 10 12 13
Reduced	7×7	49	25.0%	0 2 4 6 9 11 13

The 49-token pattern (X = kept):

X . X . X . X . . X . X . X
. . . . . . . . . . . . . .
X . X . X . X . . X . X . X
. . . . . . . . . . . . . .
X . X . X . X . . X . X . X
. . . . . . . . . . . . . .
X . X . X . X . . X . X . X
. . . . . . . . . . . . . .
. . . . . . . . . . . . . .
X . X . X . X . . X . X . X
. . . . . . . . . . . . . .
X . X . X . X . . X . X . X
. . . . . . . . . . . . . .
X . X . X . X . . X . X . X

Selection happens after the position embedding is added, so each kept token keeps the position of the patch it came from. validate_positional_preservation checks this by recomputing conv(patch) + pos[idx] by hand. The model was never trained on sparse inputs like this, so some loss of quality is expected.

Checking the baseline

Before any reduction, the 196-token path through my wrapper is compared with the standard SiglipVisionModel(pixel_values=...) output (last_hidden_state and pooler_output), in FP32 and FP16, for both SDPA and eager attention. The run stops if they disagree. The wrapper calls the same modules in the same order, so the difference should be zero; the measured values are in results/validation.csv. For the image-text model the logits matched within 1.9×10⁻⁶.

Expected cost from FLOPs

Per encoder layer with N tokens (D = 768, MLP width I = 3072), counting matmul FLOPs:

Component	FLOPs	Scales with
Q, K, V and output projections	8·N·D²	N
MLP	4·N·D·I	N
Attention scores and weighted sum	4·N²·D	N²
Pooling head	≈ 4·N·D² + const	N
Patch-embedding conv	0.23 GFLOPs	not reduced (all 196 patches are embedded first)
Tokens	Encoder + head GFLOPs/image	Quadratic share	FLOP ratio vs 196
196	35.19	4.0%	1.00×
144	25.58	3.0%	1.38×
100	17.60	2.1%	2.00×
49	8.54	1.0%	4.12×

These match PyTorch's FlopCounterMode. The quadratic term would only dominate at around 2·D + I ≈ 4,600 tokens, which is the range of high-resolution images, video and multi-camera inputs. At ViT-B/16 lengths it's small, and the FLOP ratio is an upper bound on real speed-up: small batches don't fill the GPU, and kernel launches, LayerNorm, the pooling head and the patch embedding don't shrink with N.

Setup
Setting	Value
Model	google/siglip-base-patch16-224, pretrained, eval()
Hardware	Kaggle Tesla T4 (GPU name, driver and clocks saved in results/environment.json)
Precision	FP16 for all timing and memory runs; FP32 for validation and reference embeddings
Attention	SDPA (main); eager attention as a comparison
Data	First 256 CIFAR-10 test images for timing; all 10,000 for accuracy
Batch sizes	1, 8, 16
Timing	10 warm-up runs, then 50 timed runs per setting; torch.cuda.synchronize() before and after each run, inside torch.inference_mode()
Ordering	Configurations interleaved within each repetition, so GPU clock drift doesn't favour one
Seed	42

What each timing mode covers (inputs already on the GPU):

Mode	Timed	Notes
encoder_only	precomputed tokens → 12 layers → post-LN → pooling head	main number; excludes preprocessing, patch embedding and selection
selection_only	index_select from 196 to N tokens	cost of the reduction itself
full_pipeline	pixels → patch embedding (all 196) → selection → encoder → head	realistic GPU cost
CPU preprocessing	SigLIP image processor	timed separately, never included above

Latency is reported as median and p90. Throughput is batch size divided by median latency. For memory, I reset the peak counter, run one forward pass and read torch.cuda.max_memory_allocated(). The baseline is the memory allocated once the weights and inputs are loaded, and the tables report the increase over it (activations, attention workspace, outputs). Weights, reserved memory and total device memory are logged separately.

Results

Filled in from results/*.csv by python src/benchmark.py --fill-readme. Figures are in figures/.

<!-- RESULTS_START -->

Measured on Tesla T4, PyTorch 2.11.0+cu130, Transformers 5.20.2, CUDA 13.0, precision float16, 50 timed reps after 10 warm-up iterations, run 2026-11-08T14:59:07Z.

Encoder + pooling head, attention = sdpa (primary)

Tokens	Batch	Median (ms)	p90 (ms)	Throughput (img/s)	Speed-up	Incr. peak mem (MiB)
196	1	5.29	5.60	189	1.00×	4.0
144	1	5.38	5.68	186	0.98×	2.3
100	1	5.39	5.84	186	0.98×	1.6
49	1	5.49	5.92	182	0.96×	0.8
196	8	15.67	16.41	510	1.00×	26.0
144	8	13.35	13.72	599	1.17×	19.4
100	8	10.37	10.62	771	1.51×	14.3
49	8	6.52	6.75	1228	2.40×	7.1
196	16	33.88	35.01	472	1.00×	50.5
144	16	29.40	29.70	544	1.15×	39.7
100	16	17.43	17.96	918	1.94×	26.4
49	16	11.11	11.35	1440	3.05×	14.2

Encoder + pooling head, attention = eager (ablation)

Tokens	Batch	Median (ms)	p90 (ms)	Throughput (img/s)	Speed-up	Incr. peak mem (MiB)
196	1	6.98	7.41	143	1.00×	5.0
144	1	6.80	7.38	147	1.03×	3.0
100	1	6.97	7.45	143	1.00×	1.8
49	1	7.00	7.38	143	1.00×	0.8
196	8	21.56	21.90	371	1.00×	40.4
144	8	17.62	18.20	454	1.22×	24.5
100	8	12.45	12.84	643	1.73×	16.0
49	8	7.94	8.16	1007	2.71×	7.6
196	16	48.15	49.77	332	1.00×	79.2
144	16	39.85	40.98	401	1.21×	48.8
100	16	22.81	23.72	701	2.11×	30.7
49	16	14.21	14.80	1126	3.39×	15.1

Full GPU pipeline (pixels → patch-embed all 196 → select → encoder → head), sdpa

Tokens	Batch	Median (ms)	p90 (ms)	Throughput (img/s)	Speed-up
196	1	5.60	6.30	179	1.00×
144	1	5.60	5.99	179	1.00×
100	1	5.71	6.16	175	0.98×
49	1	5.68	6.09	176	0.99×
196	8	17.04	17.56	469	1.00×
144	8	14.54	14.87	550	1.17×
100	8	11.43	11.78	700	1.49×
49	8	7.52	7.80	1064	2.27×
196	16	39.20	39.98	408	1.00×
144	16	30.75	31.33	520	1.27×
100	16	20.02	20.39	799	1.96×
49	16	13.18	13.58	1214	2.97×

Theoretical matmul FLOPs (encoder + head, per image)

Tokens	GFLOPs (analytic)	GFLOPs (torch counter)	Quadratic attention share	Theoretical speed-up
196	35.19	35.19	4.0%	1.00×
144	25.58	25.58	3.0%	1.38×
100	17.60	17.60	2.1%	2.00×
49	8.54	8.54	1.0%	4.12×

Pooled-embedding cosine similarity to the 196-token baseline (not accuracy)

Tokens	dtype	Mean	Std	Min	n
196	float16	1.0000	0.0000	1.0000	256
144	float16	0.9111	0.0312	0.7777	256
100	float16	0.9150	0.0365	0.7774	256
49	float16	0.7266	0.0607	0.5594	256
196	float32	1.0000	0.0000	1.0000	256
144	float32	0.9111	0.0312	0.7779	256
100	float32	0.9150	0.0365	0.7783	256
49	float32	0.7265	0.0607	0.5596	256

CPU preprocessing (SigLIP image processor, excluded from GPU timings): batch 1: 0.8 ms, batch 8: 4.4 ms, batch 16: 8.7 ms

<!-- RESULTS_END -->
Figure	File
Median latency vs tokens (shaded to p90)	figures/fig1_latency.png
Throughput vs tokens	figures/fig2_throughput.png
Extra peak memory vs tokens	figures/fig3_memory.png
Embedding cosine similarity vs tokens	figures/fig4_similarity.png
Measured vs theoretical speed-up	figures/fig5_speedup_vs_theory.png

A few things stand out. Batch 1 shows no gain at all. Eager attention is slower than SDPA and benefits a bit more from reduction, since it builds the full N×N attention matrix. At batch 8 the full pipeline gains slightly less than encoder-only, because the patch embedding still runs on all 196 patches.

Zero-shot classification

Cosine similarity says how far embeddings move, not whether predictions change. So I also measured zero-shot top-1 accuracy on the full CIFAR-10 test set (10,000 images) with the complete SigLIP image-text model, for the same four token counts.

Scoring follows SiglipModel.forward:

image: pixels → token-reduced vision tower → pooled embedding → L2-normalise
text:  "a photo of a {class}." for 10 classes → text tower → L2-normalise (computed once)
logits = image · textᵀ · exp(logit_scale) + logit_bias      prediction = argmax
Every configuration sees the same 10,000 images, the same preprocessing and the same 10 prompts. Each batch is embedded once and all four configurations select from those same tokens.
Before evaluating any reduced setting, the 196-token pipeline is checked against SiglipModel(input_ids, pixel_values).logits_per_image in FP32 on 32 images × 10 prompts (atol 1e-4, rtol 1e-5, and every prediction must agree). If it fails, the run stops. The result is in results/zeroshot_validation.json.
The image tower runs in FP16; text embeddings and logits stay in FP32. FP16 and FP32 predictions agreed on all 32 validation images.
Confidence intervals come from a paired bootstrap: 10,000 resamples of the 10,000 images (seed 42), using the same resampled images for every configuration.
Two latencies are reported and kept apart. Encoder-only comes from the benchmark above. End-to-end classification is measured here: pixels on the GPU → patch embedding → selection → encoder → head → logits → argmax, at batch 16 and 64. Neither includes CPU preprocessing or the text encoder.
<!-- ZEROSHOT_START -->

Measured on Tesla T4, Transformers 5.20.2, image tower float16, 10000 CIFAR-10 test images, prompt a photo of a {}., run 2026-11-08T15:00:33Z.

Validation (FP32, 32 images × 10 prompts): max |Δ logit| = 1.91e-06 vs SiglipModel(...).logits_per_image (|logit| ≤ 18.1, tolerance atol=0.0001, rtol=1e-05), argmax agreement 100% — passed. float16 vs FP32 prediction agreement at 196 tokens: 100.00% on 32 images.

Tokens	Top-1 acc.	95% CI	Δ vs 196 (pp)	95% CI of Δ (paired)	196✓→N✗	196✗→N✓	Mean cos to 196
196	92.59%	[92.06, 93.11]	+0.00	[+0.00, +0.00]	0	0	1.0000
144	84.84%	[84.15, 85.54]	-7.75	[-8.40, -7.11]	962	187	0.9105
100	78.28%	[77.46, 79.10]	-14.31	[-15.08, -13.56]	1582	151	0.9146
49	31.28%	[30.37, 32.20]	-61.31	[-62.31, -60.30]	6235	104	0.7247

Bootstrap: 10000 paired resamples of the 10000 images, seed 42, percentile intervals.

End-to-end zero-shot classification latency (GPU: pixels → patch-embed all 196 → select → encoder → head → logits against cached text embeddings; excludes CPU preprocessing and the text encoder). Not comparable to the encoder-only numbers above, which start from precomputed tokens.

Tokens	Batch	Median (ms)	p90 (ms)	Throughput (img/s)	Speed-up
196	16	40.37	41.53	396	1.00×
144	16	33.98	35.14	471	1.19×
100	16	21.96	23.49	729	1.84×
49	16	15.07	15.58	1062	2.68×
196	64	186.78	190.32	343	1.00×
144	64	138.40	140.81	462	1.35×
100	64	95.43	97.65	671	1.96×
49	64	53.25	54.72	1202	3.51×
<!-- ZEROSHOT_END -->
Figure	File
Accuracy vs tokens, with 95% CI	figures/fig6_zeroshot_accuracy_vs_tokens.png
Accuracy vs latency (encoder-only and end-to-end)	figures/fig7_zeroshot_accuracy_vs_latency.png
Embedding similarity vs accuracy	figures/fig8_similarity_vs_accuracy.png
Per-class accuracy	figures/fig9_zeroshot_per_class.png

Every reduced configuration lost accuracy, and every confidence interval for the difference excludes zero. At 100 tokens the 1.96× end-to-end speed-up (batch 64) came with a 14.31-point drop. CIFAR-10 images are upsampled from 32×32, which makes neighbouring patches very similar, so the loss on real high-resolution images is probably larger.

Limitations
Low-resolution data. Each 16×16 patch covers about 2×2 original CIFAR pixels, so the images are highly redundant. That probably flatters the reduced configurations. Latency and memory don't depend on image content.
One dataset, one prompt. Zero-shot accuracy is sensitive to prompt wording. I used a single template (a photo of a {}.) on purpose so that only the image side changes.
Simple selection. Uniform subsampling ignores image content, and the model was never trained on sparse inputs. Smarter methods should keep more information for the same budget.
Short sequences. At 196 tokens or fewer, attention is a few percent of compute. These results say nothing about quadratic savings at long context.
One model, one GPU, one precision. Kaggle T4s are shared and can throttle. Clocks are logged but not locked, and results from other GPUs aren't directly comparable.
Patch embedding isn't reduced. All 196 patches are still embedded before selection. Skipping them would mean changing the convolution.
Inference-only memory. Under inference_mode, activations are freed layer by layer, so peak memory reflects roughly one layer's working set, not training memory.
What I'd try next
Content-aware reduction (attention-based pruning, ToMe-style token merging, learned selectors) against this uniform baseline at the same budgets.
Random selection as a control, to separate "fewer tokens" from "which tokens".
Native-resolution datasets such as ImageNet-1k validation, prompt ensembles and linear probes.
Longer sequences (higher resolution, video, multiple cameras), where attention and kernels like FlashAttention start to matter.
Measuring how vision-encoder token reduction affects language-model prefill time and KV-cache size in a full VLM or VLA.
Light fine-tuning on sparse inputs to recover accuracy at low token counts.
Running it on Kaggle
Create a Kaggle notebook and import kaggle_notebook.ipynb (File → Import Notebook).
Set Accelerator to GPU T4 x2 (only cuda:0 is used) and turn Internet on, which is needed for the checkpoint and CIFAR-10.
Run All. The benchmark takes about 5–10 minutes on a T4 and the zero-shot evaluation another 5–10, mostly CPU preprocessing.
Outputs go to /kaggle/working/efficient-vit-bench/: results/, figures/, an updated README.md and efficient-vit-bench-results.zip.

The notebook writes the four source files in src/ with %%writefile, so nothing needs installing. On any CUDA machine:

bash
pip install -r requirements.txt
python src/benchmark.py --out .                 # latency, memory, similarity
python src/benchmark.py --out . --fill-readme   # update the Results section
python src/zeroshot.py --out .                  # zero-shot accuracy
python src/zeroshot.py --out . --fill-readme    # update the zero-shot section

Useful flags: --batch-sizes 1 8 16 32, --reps 100, --attn-impls sdpa (skip eager), --dtype float32.

Repository layout
efficient-vit-bench/
├── README.md
├── requirements.txt
├── kaggle_notebook.ipynb      # writes src/ and runs everything
├── src/
│   ├── token_selection.py     # k×k token subsets and checks
│   ├── model_wrapper.py       # token-reduced SigLIP forward, validation, FLOP counts
│   ├── benchmark.py           # latency, memory, similarity, plots
│   └── zeroshot.py            # zero-shot accuracy, bootstrap CIs, end-to-end latency
├── results/                   # CSV/JSON from the run
└── figures/                   # PNG/PDF from the run

results/ contains the environment and settings (environment.json, experiment_params.json, model_config.json, token_indices.json), validation (validation.csv, zeroshot_validation.json), raw and summarised measurements (latency_*.csv, memory_summary.csv, flops.csv, similarity_*.csv, preprocessing_*.csv), and the zero-shot outputs (zeroshot_predictions.csv with one row per image and configuration, zeroshot_summary.csv, zeroshot_per_class.csv, zeroshot_latency_*.csv, zeroshot_text_embeddings.pt).
