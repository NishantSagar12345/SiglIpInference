"""Experiment E: zero-shot CIFAR-10 classification under visual token reduction.

Uses the full pretrained SigLIP image-text model. Reduced-token image
embeddings come from the existing ``TokenReducedSiglipVision`` wrapper and are
scored exactly as in ``SiglipModel.forward`` (checked in transformers 4.46 and 5.x):

    image_embeds = vision pooler_output (MAP head)       text_embeds = text pooler_output
    both L2-normalised
    logits_per_image = image_embeds @ text_embeds.T * exp(logit_scale) + logit_bias

Prediction = argmax over the 10 class prompts. Text embeddings are computed once
(FP32) and cached; only the image side changes between configurations.

Run:
    python src/zeroshot.py --out .              # after (or independently of) src/benchmark.py
    python src/zeroshot.py --out . --fill-readme

Order of work inside ``main``:
  1. FP32: check that our 196-token pipeline reproduces SiglipModel(...).logits_per_image.
     If it does not, the run stops (RuntimeError) before any reduced configuration is evaluated.
  2. Evaluate all 10,000 test images for 196/144/100/49 tokens: identical images,
     preprocessing and prompts. Each preprocessed batch is embedded once and every
     configuration is selected from the same (B,196,768) tokens.
  3. Top-1, per-class accuracy, paired bootstrap CIs, end-to-end classification latency.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B  # noqa: E402  (reuses data loading, preprocessing, env capture, sync)
from model_wrapper import TokenReducedSiglipVision  # noqa: E402
from token_selection import build_configs  # noqa: E402

# Label order of the CIFAR-10 dataset (torchvision `classes` and HF `uoft-cs/cifar10`).
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
PROMPT_TEMPLATE = "a photo of a {}."
TEXT_MAX_LENGTH = 64  # SigLIP text tower was trained with padding="max_length", 64 tokens


# ----------------------------------------------------------------------------- model / text
def load_siglip(model_id: str, device: torch.device, attn_impl: str = "sdpa"):
    from transformers import SiglipModel
    model = SiglipModel.from_pretrained(model_id, attn_implementation=attn_impl).float().eval().to(device)
    return model


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def tokenize_prompts(tokenizer, classes=CIFAR10_CLASSES, template=PROMPT_TEMPLATE) -> tuple[list[str], torch.Tensor]:
    prompts = [template.format(c) for c in classes]
    ids = tokenizer(prompts, padding="max_length", max_length=TEXT_MAX_LENGTH,
                    truncation=True, return_tensors="pt")["input_ids"]
    assert ids.shape == (len(classes), TEXT_MAX_LENGTH), ids.shape
    return prompts, ids


@torch.inference_mode()
def text_embeddings(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Normalised FP32 text embeddings, computed once (same call as SiglipModel.forward)."""
    out = model.text_model(input_ids=input_ids)
    emb = out.pooler_output.float()
    return emb / emb.norm(p=2, dim=-1, keepdim=True)


def score(image_embeds: torch.Tensor, text_norm: torch.Tensor, logit_scale: float, logit_bias: float) -> torch.Tensor:
    """SigLIP logits_per_image, computed in FP32 regardless of the image-tower precision."""
    img = image_embeds.float()
    img = img / img.norm(p=2, dim=-1, keepdim=True)
    return img @ text_norm.t() * logit_scale + logit_bias


# ----------------------------------------------------------------------------- validation
@torch.inference_mode()
def validate_image_text_logits(model, wrapper, pixel_values, input_ids, text_norm,
                               logit_scale, logit_bias, atol: float = 1e-4, rtol: float = 1e-5) -> dict:
    """196-token pipeline vs the standard SiglipModel(input_ids, pixel_values) forward (FP32)."""
    ref = model(input_ids=input_ids, pixel_values=pixel_values)
    identity = torch.arange(wrapper.num_patches, device=pixel_values.device)
    _, pooled = wrapper(pixel_values, identity)
    ours = score(pooled, text_norm, logit_scale, logit_bias)
    ref_logits = ref.logits_per_image.float()
    img_norm = pooled.float() / pooled.float().norm(dim=-1, keepdim=True)
    res = {
        "n_images": int(pixel_values.shape[0]), "n_prompts": int(input_ids.shape[0]),
        "dtype": str(pixel_values.dtype),
        "logits_shape": list(ours.shape),
        "max_abs_diff_logits": float((ours - ref_logits).abs().max()),
        "max_abs_ref_logit": float(ref_logits.abs().max()),
        "max_abs_diff_image_embeds": float((img_norm - ref.image_embeds.float()).abs().max()),
        "max_abs_diff_text_embeds": float((text_norm - ref.text_embeds.float()).abs().max()),
        "argmax_agreement": float((ours.argmax(-1) == ref_logits.argmax(-1)).float().mean()),
        "atol": atol, "rtol": rtol,
    }
    res["passed"] = bool(ours.shape == ref_logits.shape
                         and torch.allclose(ours, ref_logits, atol=atol, rtol=rtol)
                         and res["argmax_agreement"] == 1.0)
    return res


# ----------------------------------------------------------------------------- evaluation
@torch.inference_mode()
def evaluate(wrapper, processor, images, labels, configs, text_norm, logit_scale, logit_bias,
             batch_size: int, dtype: torch.dtype, device: torch.device) -> list[dict]:
    """Per-image predictions for every configuration on identical preprocessed batches."""
    rows = []
    n = len(images)
    for start in range(0, n, batch_size):
        px = B.preprocess(processor, images[start:start + batch_size]).to(device=device, dtype=dtype)
        tokens = wrapper.embed(px)                                   # (b,196,D) once per batch
        ref_emb = None
        for name, idx in configs.items():                            # "196" first
            _, pooled = wrapper.encode(wrapper.select(tokens, idx))
            logits = score(pooled, text_norm, logit_scale, logit_bias)
            emb = pooled.float() / pooled.float().norm(dim=-1, keepdim=True)
            if ref_emb is None:
                assert name == "196", "196-token configuration must be evaluated first"
                ref_emb = emb
            cos = (emb * ref_emb).sum(-1)
            pred = logits.argmax(-1).cpu().tolist()
            true_logit = logits.gather(1, torch.as_tensor(labels[start:start + len(pred)], device=device)[:, None])
            for j, p in enumerate(pred):
                i = start + j
                rows.append({"image": i, "label": labels[i], "num_tokens": int(name), "pred": p,
                             "correct": int(p == labels[i]), "true_class_logit": float(true_logit[j, 0]),
                             "max_logit": float(logits[j].max()), "cosine_to_196": float(cos[j])})
        if (start // batch_size) % 10 == 0:
            print(f"  evaluated {min(start + batch_size, n)}/{n}")
    return rows


def paired_bootstrap(correct: dict[int, np.ndarray], baseline: int = 196, n_boot: int = 10000,
                     seed: int = 42, chunk: int = 200) -> dict:
    """Percentile 95% CIs from one shared set of image resamples (paired across configs)."""
    keys = sorted(correct, reverse=True)
    C = np.stack([correct[k].astype(np.float32) for k in keys])      # (n_cfg, n)
    n = C.shape[1]
    rng = np.random.default_rng(seed)
    boots = []
    for s in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - s), n))
        boots.append(C[:, idx].mean(-1))                              # (n_cfg, chunk)
    Bm = np.concatenate(boots, axis=1)                                # (n_cfg, n_boot)
    b0 = Bm[keys.index(baseline)]
    out = {}
    for i, k in enumerate(keys):
        diff = Bm[i] - b0
        base_c, k_c = correct[baseline].astype(bool), correct[k].astype(bool)
        out[k] = {
            "accuracy": float(C[i].mean()),
            "accuracy_ci_low": float(np.percentile(Bm[i], 2.5)),
            "accuracy_ci_high": float(np.percentile(Bm[i], 97.5)),
            "delta_vs_196": float(C[i].mean() - C[keys.index(baseline)].mean()),
            "delta_ci_low": float(np.percentile(diff, 2.5)),
            "delta_ci_high": float(np.percentile(diff, 97.5)),
            "baseline_right_reduced_wrong": int((base_c & ~k_c).sum()),
            "baseline_wrong_reduced_right": int((~base_c & k_c).sum()),
        }
    return out


# ----------------------------------------------------------------------------- latency
def classification_latency(wrapper, px_all, configs, text_norm, logit_scale, logit_bias,
                           batch_sizes, warmup, reps, tag) -> list[dict]:
    """End-to-end GPU classification: pixels -> embed(196) -> select -> encoder -> head
    -> normalise -> logits vs cached text embeddings -> argmax. Interleaved like run_latency.
    Excludes CPU preprocessing and the (cached) text encoder."""
    device = px_all.device
    rows = []
    for bs in batch_sizes:
        px = px_all[:bs].contiguous()

        def make(idx):
            def fn():
                _, pooled = wrapper(px, idx)
                return score(pooled, text_norm, logit_scale, logit_bias).argmax(-1)
            return fn

        runners = {name: make(idx) for name, idx in configs.items()}
        keys = list(runners)
        with torch.inference_mode():
            for k in keys:
                for _ in range(warmup):
                    runners[k]()
            B.sync(device)
            for r in range(reps):
                for k in keys[r % len(keys):] + keys[:r % len(keys)]:
                    B.sync(device)
                    t0 = time.perf_counter()
                    runners[k]()
                    B.sync(device)
                    rows.append({**tag, "mode": "zeroshot_end_to_end", "batch": bs, "num_tokens": int(k),
                                 "rep": r, "latency_ms": (time.perf_counter() - t0) * 1e3})
    return rows


# ----------------------------------------------------------------------------- plots
def make_plots(results_dir: Path, figures_dir: Path, encoder_batch: int, e2e_batch: int) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 120})
    params = json.loads((results_dir / "zeroshot_params.json").read_text())
    gpu, prec, n = params.get("gpu_name") or "cpu", params["eval_dtype"], params["n_images"]
    summ = pd.read_csv(results_dir / "zeroshot_summary.csv").sort_values("num_tokens")
    acc = summ.accuracy * 100
    err = np.vstack([acc - summ.accuracy_ci_low * 100, summ.accuracy_ci_high * 100 - acc])
    paths = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figures_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(figures_dir / f"{name}.png")

    def label_points(ax, xs, ys):
        for x, y, t in zip(xs, ys, summ.num_tokens):
            ax.annotate(f"{t}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    # fig6: accuracy vs token count
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.errorbar(summ.num_tokens, acc, yerr=err, marker="o", capsize=3, label="top-1 (95% bootstrap CI)")
    ax.axhline(acc[summ.num_tokens == 196].iloc[0], color="grey", ls=":", lw=1, label="196-token baseline")
    ax.set_xticks(summ.num_tokens)
    ax.set_xlabel("Visual tokens per image (N)"); ax.set_ylabel("Zero-shot top-1 accuracy (%)")
    ax.set_title(f"CIFAR-10 test zero-shot accuracy (n={n})\n{gpu} · {prec} · prompt “{params['prompt_template']}”",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig6_zeroshot_accuracy_vs_tokens")

    # fig7: accuracy vs measured latency (two different latency definitions, two panels)
    lat_p = results_dir / "latency_summary.csv"
    e2e = pd.read_csv(results_dir / "zeroshot_latency_summary.csv")
    e2e = e2e[e2e.batch == e2e_batch].set_index("num_tokens").loc[summ.num_tokens]
    panels = []
    if lat_p.exists():
        enc = pd.read_csv(lat_p)
        enc = enc[(enc.attn_impl == "sdpa") & (enc["mode"] == "encoder_only") & (enc.batch == encoder_batch)]
        if len(enc) == len(summ):
            enc = enc.set_index("num_tokens").loc[summ.num_tokens]
            panels.append((enc.median_ms.values, f"Encoder-only latency, batch {encoder_batch} (ms)\n"
                                                 "(Exp. A: precomputed tokens → encoder + head)"))
    panels.append((e2e.median_ms.values, f"End-to-end classification latency, batch {e2e_batch} (ms)\n"
                                         "(pixels on GPU → embed → select → encoder → head → logits)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 3.8), sharey=True, squeeze=False)
    for ax, (x, xlabel) in zip(axes[0], panels):
        ax.errorbar(x, acc, yerr=err, marker="o", capsize=3)
        label_points(ax, x, acc)
        ax.set_xlabel(xlabel, fontsize=9)
    axes[0][0].set_ylabel("Zero-shot top-1 accuracy (%)")
    fig.suptitle(f"Accuracy vs measured latency · {gpu} · {prec} · sdpa · median of timed reps", fontsize=10)
    save(fig, "fig7_zeroshot_accuracy_vs_latency")

    # fig8: embedding similarity vs accuracy
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.errorbar(summ.cosine_mean, acc, yerr=err, xerr=summ.cosine_std, marker="o", capsize=3, ls="-")
    label_points(ax, summ.cosine_mean, acc)
    ax.set_xlabel("Mean cosine similarity to 196-token embedding (± std)")
    ax.set_ylabel("Zero-shot top-1 accuracy (%)")
    ax.set_title(f"Embedding drift vs accuracy (same {n} images)\n{gpu} · {prec}", fontsize=10)
    save(fig, "fig8_similarity_vs_accuracy")

    # fig9: per-class accuracy
    pc = pd.read_csv(results_dir / "zeroshot_per_class.csv")
    tokens = sorted(pc.num_tokens.unique(), reverse=True)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.8 / len(tokens)
    x = np.arange(len(CIFAR10_CLASSES))
    for i, t in enumerate(tokens):
        d = pc[pc.num_tokens == t].set_index("class_name").reindex(CIFAR10_CLASSES)
        ax.bar(x + (i - (len(tokens) - 1) / 2) * w, d.accuracy * 100, w, label=f"{t} tokens")
    ax.set_xticks(x, CIFAR10_CLASSES, rotation=30, ha="right")
    ax.set_ylabel("Per-class accuracy (%)")
    ax.set_title(f"Per-class zero-shot accuracy (1,000 test images per class) · {prec}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=len(tokens))
    save(fig, "fig9_zeroshot_per_class")
    return paths


# ----------------------------------------------------------------------------- README
def results_markdown(results_dir: Path) -> str:
    import pandas as pd
    p = json.loads((results_dir / "zeroshot_params.json").read_text())
    v = json.loads((results_dir / "zeroshot_validation.json").read_text())
    s = pd.read_csv(results_dir / "zeroshot_summary.csv").sort_values("num_tokens", ascending=False)
    e2e = pd.read_csv(results_dir / "zeroshot_latency_summary.csv")
    lines = [f"Measured on **{p.get('gpu_name') or 'cpu'}**, Transformers {p['transformers']}, image tower "
             f"**{p['eval_dtype']}**, {p['n_images']} CIFAR-10 test images, prompt `{p['prompt_template']}`, "
             f"run {p['timestamp_utc']}.", "",
             f"Validation (FP32, {v['logits']['n_images']} images × {v['logits']['n_prompts']} prompts): "
             f"max |Δ logit| = {v['logits']['max_abs_diff_logits']:.2e} vs `SiglipModel(...).logits_per_image` "
             f"(|logit| ≤ {v['logits']['max_abs_ref_logit']:.1f}, tolerance atol={v['logits']['atol']}, "
             f"rtol={v['logits']['rtol']}), argmax agreement {v['logits']['argmax_agreement']:.0%} — "
             f"{'**passed**' if v['logits']['passed'] else '**FAILED**'}."]
    if "precision_check" in v:
        pc = v["precision_check"]
        lines.append(f"{p['eval_dtype']} vs FP32 prediction agreement at 196 tokens: "
                     f"{pc['prediction_agreement']:.2%} on {pc['n_images']} images.")
    lines += ["", "| Tokens | Top-1 acc. | 95% CI | Δ vs 196 (pp) | 95% CI of Δ (paired) | 196✓→N✗ | 196✗→N✓ | Mean cos to 196 |",
              "|---:|---:|---|---:|---|---:|---:|---:|"]
    for r in s.itertuples():
        lines.append(f"| {r.num_tokens} | {r.accuracy * 100:.2f}% | [{r.accuracy_ci_low * 100:.2f}, "
                     f"{r.accuracy_ci_high * 100:.2f}] | {r.delta_vs_196 * 100:+.2f} | "
                     f"[{r.delta_ci_low * 100:+.2f}, {r.delta_ci_high * 100:+.2f}] | "
                     f"{r.baseline_right_reduced_wrong} | {r.baseline_wrong_reduced_right} | {r.cosine_mean:.4f} |")
    lines += ["", f"Bootstrap: {p['n_boot']} paired resamples of the {p['n_images']} images, seed {p['seed']}, "
                  "percentile intervals.", "",
              "**End-to-end zero-shot classification latency** (GPU: pixels → patch-embed all 196 → select → "
              "encoder → head → logits against cached text embeddings; excludes CPU preprocessing and the text "
              "encoder). Not comparable to the encoder-only numbers above, which start from precomputed tokens.", "",
              "| Tokens | Batch | Median (ms) | p90 (ms) | Throughput (img/s) | Speed-up |", "|---:|---:|---:|---:|---:|---:|"]
    for r in e2e.sort_values(["batch", "num_tokens"], ascending=[True, False]).itertuples():
        lines.append(f"| {r.num_tokens} | {r.batch} | {r.median_ms:.2f} | {r.p90_ms:.2f} | "
                     f"{r.throughput_img_per_s:.0f} | {r.speedup_vs_196:.2f}× |")
    return "\n".join(lines) + "\n"


def fill_readme(readme: Path, results_dir: Path) -> None:
    start, end = "<!-- ZEROSHOT_START -->", "<!-- ZEROSHOT_END -->"
    text = readme.read_text()
    assert start in text and end in text, "README zero-shot markers missing"
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(f"{head}{start}\n{results_markdown(results_dir)}{end}{tail}")
    print(f"README zero-shot section updated from {results_dir}")


# ----------------------------------------------------------------------------- main
def main(args) -> None:
    import pandas as pd
    import transformers

    out = Path(args.out)
    results_dir, figures_dir = out / "results", out / "figures"
    results_dir.mkdir(parents=True, exist_ok=True); figures_dir.mkdir(parents=True, exist_ok=True)
    B.set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_dtype = B.DTYPES[args.dtype]
    env = B.collect_environment(device)

    configs = {k: v.to(device) for k, v in build_configs(14, (14, 12, 10, 7)).items()}
    assert list(configs)[0] == "196"

    images, labels, source = B.load_cifar10_test(args.n_images, root=args.data_root)
    processor = B.load_processor(args.model_id)
    tokenizer = load_tokenizer(args.model_id)
    prompts, input_ids = tokenize_prompts(tokenizer)
    input_ids = input_ids.to(device)

    # 1. FP32 validation of the image-text pathway ---------------------------------
    model = load_siglip(args.model_id, device, args.attn_impl)
    wrapper = TokenReducedSiglipVision(model)
    logit_scale = float(model.logit_scale.float().exp())
    logit_bias = float(model.logit_bias.float())
    text_norm = text_embeddings(model, input_ids)                     # cached, FP32
    assert text_norm.shape == (len(CIFAR10_CLASSES), wrapper.hidden_size), text_norm.shape
    torch.save({"prompts": prompts, "text_embeds_normalized": text_norm.cpu(),
                "logit_scale_exp": logit_scale, "logit_bias": logit_bias},
               results_dir / "zeroshot_text_embeddings.pt")

    px_val = B.preprocess(processor, images[:args.val_images]).to(device)
    val = {"logits": validate_image_text_logits(model, wrapper, px_val, input_ids, text_norm,
                                                logit_scale, logit_bias, atol=args.atol, rtol=args.rtol)}
    (results_dir / "zeroshot_validation.json").write_text(json.dumps(val, indent=2))
    print("FP32 image-text logit validation:", json.dumps(val["logits"], indent=2))
    if not val["logits"]["passed"]:
        raise RuntimeError("196-token pipeline does NOT reproduce SiglipModel logits_per_image within tolerance; "
                           "stopping before evaluating reduced-token configurations. See "
                           f"{results_dir / 'zeroshot_validation.json'}")
    with torch.inference_mode():
        _, pooled32 = wrapper(px_val, configs["196"])
        pred32 = score(pooled32, text_norm, logit_scale, logit_bias).argmax(-1)

    # 2. Evaluation in the benchmark precision ----------------------------------------
    model.to(eval_dtype)                                              # vision tower used from here on
    if eval_dtype != torch.float32:
        with torch.inference_mode():
            _, pooled16 = wrapper(px_val.to(eval_dtype), configs["196"])
            pred16 = score(pooled16, text_norm, logit_scale, logit_bias).argmax(-1)
        val["precision_check"] = {"n_images": int(px_val.shape[0]),
                                  "prediction_agreement": float((pred16 == pred32).float().mean())}
        (results_dir / "zeroshot_validation.json").write_text(json.dumps(val, indent=2))
    del px_val; gc.collect()

    t0 = time.perf_counter()
    rows = evaluate(wrapper, processor, images, labels, configs, text_norm, logit_scale, logit_bias,
                    args.batch_size, eval_dtype, device)
    eval_wall_s = time.perf_counter() - t0
    pred = pd.DataFrame(rows)
    pred.to_csv(results_dir / "zeroshot_predictions.csv", index=False)
    for k in configs:
        assert (pred.num_tokens == int(k)).sum() == len(images)

    # 3. Metrics ----------------------------------------------------------------------------
    correct = {int(k): pred[pred.num_tokens == int(k)].sort_values("image").correct.values for k in configs}
    boot = paired_bootstrap(correct, n_boot=args.n_boot, seed=args.seed)
    cos = pred.groupby("num_tokens").cosine_to_196.agg(cosine_mean="mean", cosine_std="std")
    summ = pd.DataFrame([{"num_tokens": k, "n_images": len(images), **v,
                          "cosine_mean": cos.loc[k, "cosine_mean"], "cosine_std": cos.loc[k, "cosine_std"]}
                         for k, v in boot.items()])
    summ.to_csv(results_dir / "zeroshot_summary.csv", index=False)
    pc = pred.groupby(["num_tokens", "label"]).correct.agg(n="count", accuracy="mean").reset_index()
    pc["class_name"] = pc.label.map(dict(enumerate(CIFAR10_CLASSES)))
    base_pc = pc[pc.num_tokens == 196].set_index("label").accuracy
    pc["delta_vs_196"] = pc.accuracy - pc.label.map(base_pc)
    pc.to_csv(results_dir / "zeroshot_per_class.csv", index=False)

    # 4. End-to-end classification latency ---------------------------------------------
    px_lat = B.preprocess(processor, images[:max(args.latency_batch_sizes)]).to(device=device, dtype=eval_dtype)
    lat_rows = classification_latency(wrapper, px_lat, configs, text_norm, logit_scale, logit_bias,
                                      args.latency_batch_sizes, args.warmup, args.reps,
                                      {"attn_impl": args.attn_impl, "dtype": args.dtype})
    lat = pd.DataFrame(lat_rows)
    lat.to_csv(results_dir / "zeroshot_latency_raw.csv", index=False)
    B.aggregate_latency(lat).to_csv(results_dir / "zeroshot_latency_summary.csv", index=False)

    params = {"model_id": args.model_id, "model_class": type(model).__name__, "dataset": source,
              "n_images": len(images), "classes": CIFAR10_CLASSES, "prompt_template": PROMPT_TEMPLATE,
              "prompts": prompts, "text_max_length": TEXT_MAX_LENGTH, "eval_dtype": args.dtype,
              "logits_dtype": "float32", "attn_impl": args.attn_impl, "eval_batch_size": args.batch_size,
              "eval_wall_seconds_all_configs_incl_cpu_preprocessing": eval_wall_s,
              "latency_batch_sizes": args.latency_batch_sizes, "warmup_iters": args.warmup, "timed_reps": args.reps,
              "n_boot": args.n_boot, "seed": args.seed, "validation_images": args.val_images,
              "logit_scale_exp": logit_scale, "logit_bias": logit_bias,
              "token_configs": {k: v.numel() for k, v in configs.items()},
              "transformers": transformers.__version__, "torch": torch.__version__,
              "gpu_name": env.get("gpu_name"), "timestamp_utc": env["timestamp_utc"]}
    (results_dir / "zeroshot_params.json").write_text(json.dumps(params, indent=2))
    del model, wrapper, px_lat; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    figs = make_plots(results_dir, figures_dir, args.encoder_latency_batch, args.e2e_plot_batch)
    md = results_markdown(results_dir)
    (results_dir / "zeroshot_results_table.md").write_text(md)
    print("\n" + md)
    print("Figures:", *figs, sep="\n  ")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=".")
    p.add_argument("--model-id", default=B.MODEL_ID)
    p.add_argument("--n-images", type=int, default=10000)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--batch-size", type=int, default=128, help="evaluation batch (fits a 16 GB T4 in FP16)")
    p.add_argument("--dtype", default="float16", choices=list(B.DTYPES))
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--val-images", type=int, default=32)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--latency-batch-sizes", type=int, nargs="+", default=[16, 64])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--encoder-latency-batch", type=int, default=16,
                   help="batch of the Experiment A encoder-only latency used in fig7")
    p.add_argument("--e2e-plot-batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fill-readme", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    if a.fill_readme:
        fill_readme(Path(a.out) / "README.md", Path(a.out) / "results")
    else:
        main(a)
