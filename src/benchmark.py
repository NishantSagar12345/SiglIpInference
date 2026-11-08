"""Benchmark: visual token count vs latency, throughput, memory and representation.

Run (Kaggle GPU or any CUDA machine):

    python src/benchmark.py --out .            # full run, writes results/ and figures/
    python src/benchmark.py --fill-readme      # re-render README results from results/*.csv

Timing modes (all on GPU, model already loaded, inputs already on device):
  encoder_only   precomputed (B,N,768) tokens -> 12 encoder layers -> post-LN -> MAP head.
                 PRIMARY measurement. Excludes CPU preprocessing, patch embedding
                 and token selection.
  selection_only (B,196,768) tokens -> index_select -> (B,N,768). Cost of selection alone.
  full_pipeline  (B,3,224,224) pixels -> patch embed + pos (all 196) -> select -> encoder -> head.
CPU image preprocessing (SigLIP processor) is timed separately.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_selection import build_configs, describe, selection_table  # noqa: E402
from model_wrapper import (TokenReducedSiglipVision, analytic_flops,  # noqa: E402
                           validate_baseline, validate_positional_preservation)

MODEL_ID = "google/siglip-base-patch16-224"
DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


# ----------------------------------------------------------------------------- setup
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Keep FP32 validation in true FP32 (no effect on T4, which has no TF32).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def collect_environment(device: torch.device) -> dict:
    import transformers
    env = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device": str(device),
    }
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        env.update({
            "gpu_name": p.name,
            "gpu_total_memory_mib": p.total_memory / 2**20,
            "gpu_compute_capability": f"{p.major}.{p.minor}",
            "gpu_multiprocessors": p.multi_processor_count,
            "gpu_count_visible": torch.cuda.device_count(),
        })
        try:
            q = "name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.limit,temperature.gpu"
            env["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", f"--query-gpu={q}", "--format=csv"],
                capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:  # nvidia-smi missing is not fatal
            env["nvidia_smi"] = f"unavailable: {e}"
    return env


# ----------------------------------------------------------------------------- data
def load_cifar10_test(n: int, root: str = "./data"):
    """First ``n`` CIFAR-10 test images (deterministic). Returns PIL images, labels."""
    try:
        from torchvision.datasets import CIFAR10
        ds = CIFAR10(root=root, train=False, download=True)
        imgs = [ds[i][0].convert("RGB") for i in range(n)]
        labels = [int(ds[i][1]) for i in range(n)]
        source = "torchvision.datasets.CIFAR10(train=False)"
    except Exception as e:
        print(f"torchvision CIFAR-10 download failed ({e}); falling back to HF datasets")
        from datasets import load_dataset
        ds = load_dataset("uoft-cs/cifar10", split="test")
        imgs = [ds[i]["img"].convert("RGB") for i in range(n)]
        labels = [int(ds[i]["label"]) for i in range(n)]
        source = "huggingface datasets uoft-cs/cifar10 (test)"
    assert len(imgs) == n and imgs[0].size == (32, 32), imgs[0].size
    return imgs, labels, source


def load_processor(model_id: str = MODEL_ID):
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(model_id)


def preprocess(processor, images) -> torch.Tensor:
    px = processor(images=images, return_tensors="pt")["pixel_values"]
    assert px.ndim == 4 and px.shape[1:] == (3, 224, 224), px.shape
    return px


def time_preprocessing(processor, images, batch_sizes, reps: int = 10) -> list[dict]:
    rows = []
    for bs in batch_sizes:
        batch = images[:bs]
        preprocess(processor, batch)  # warm-up
        for r in range(reps):
            t0 = time.perf_counter()
            preprocess(processor, batch)
            rows.append({"stage": "cpu_preprocessing", "batch": bs, "rep": r,
                         "latency_ms": (time.perf_counter() - t0) * 1e3})
    return rows


# ----------------------------------------------------------------------------- model
def load_model(model_id: str, attn_impl: str, dtype: torch.dtype, device: torch.device):
    """Load SiglipVisionModel; returns (model, weight_bytes_on_device, attn_impl_used)."""
    from transformers import SiglipVisionModel
    try:
        model = SiglipVisionModel.from_pretrained(model_id, attn_implementation=attn_impl)
    except (ValueError, TypeError, ImportError) as e:
        raise RuntimeError(f"attn_implementation={attn_impl!r} not supported here: {e}") from e
    model = model.to(dtype).eval()
    before = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    model = model.to(device)
    sync(device)
    after = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    used = getattr(model.config, "_attn_implementation", attn_impl)
    assert used == attn_impl, f"requested {attn_impl}, model uses {used}"
    return model, after - before, used


# ----------------------------------------------------------------------------- timing
def make_runner(wrapper: TokenReducedSiglipVision, mode: str, px: torch.Tensor, idx: torch.Tensor):
    """Return a zero-arg callable for one (mode, config, batch) with inputs on device."""
    with torch.inference_mode():
        if mode == "encoder_only":
            tokens = wrapper.select(wrapper.embed(px), idx).contiguous()
            assert tokens.shape[1] == idx.numel()
            return lambda: wrapper.encode(tokens)
        if mode == "selection_only":
            full = wrapper.embed(px).contiguous()
            return lambda: wrapper.select(full, idx)
        if mode == "full_pipeline":
            return lambda: wrapper(px, idx)
    raise ValueError(mode)


def run_latency(wrapper, px_all, configs, batch_sizes, modes, warmup, reps, tag: dict) -> list[dict]:
    """Interleaved timing: every repetition visits every (config, mode) once.

    Interleaving spreads clock/thermal drift (T4s throttle) across configs
    instead of biasing whichever config runs last.
    """
    device = px_all.device
    rows = []
    for bs in batch_sizes:
        px = px_all[:bs].contiguous()
        runners = {}
        for name, idx in configs.items():
            for mode in modes:
                runners[(name, mode)] = make_runner(wrapper, mode, px, idx)
        keys = list(runners)
        with torch.inference_mode():
            for k in keys:                       # warm-up (kernel selection, caches)
                for _ in range(warmup):
                    runners[k]()
            sync(device)
            for r in range(reps):
                order = keys[r % len(keys):] + keys[:r % len(keys)]   # rotate start
                for k in order:
                    fn = runners[k]
                    sync(device)
                    t0 = time.perf_counter()
                    fn()
                    sync(device)
                    dt = time.perf_counter() - t0
                    rows.append({**tag, "mode": k[1], "batch": bs, "num_tokens": int(k[0]),
                                 "rep": r, "latency_ms": dt * 1e3})
        del runners
    return rows


def run_memory(wrapper, px_all, configs, batch_sizes, modes, weight_bytes: int, tag: dict) -> list[dict]:
    """Peak allocated memory per forward, relative to a defined baseline.

    baseline_allocated = torch.cuda.memory_allocated() after the model weights
    and that run's inputs are resident and the cache has been emptied.
    incremental_peak  = max_memory_allocated() during one forward - baseline
                       (activations, workspace, outputs).
    """
    device = px_all.device
    if device.type != "cuda":
        return []
    total = torch.cuda.get_device_properties(device).total_memory
    rows = []
    for bs in batch_sizes:
        for name, idx in configs.items():
            for mode in modes:
                gc.collect(); torch.cuda.empty_cache()
                px = px_all[:bs].contiguous()
                fn = make_runner(wrapper, mode, px, idx)
                with torch.inference_mode():
                    fn(); sync(device)                       # warm-up allocation
                    gc.collect(); torch.cuda.empty_cache(); sync(device)
                    base = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    try:
                        out = fn(); sync(device)
                        peak = torch.cuda.max_memory_allocated(device)
                        reserved = torch.cuda.max_memory_reserved(device)
                        oom = False
                        del out
                    except torch.cuda.OutOfMemoryError:
                        peak = reserved = float("nan"); oom = True
                rows.append({**tag, "mode": mode, "batch": bs, "num_tokens": int(name),
                             "weights_mib": weight_bytes / 2**20,
                             "baseline_allocated_mib": base / 2**20,
                             "peak_allocated_mib": peak / 2**20,
                             "incremental_peak_mib": (peak - base) / 2**20,
                             "peak_reserved_mib": reserved / 2**20,
                             "device_total_mib": total / 2**20, "oom": oom})
                del fn, px
    gc.collect(); torch.cuda.empty_cache()
    return rows


def counted_flops(wrapper, px_all, configs, cfg) -> list[dict]:
    """FLOPs counted by torch.utils.flop_counter for the encoder_only path, batch 1."""
    rows = []
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        FlopCounterMode = None
    for name, idx in configs.items():
        a = analytic_flops(int(name), cfg)
        counted = float("nan")
        if FlopCounterMode is not None:
            try:
                fn = make_runner(wrapper, "encoder_only", px_all[:1], idx)
                with torch.inference_mode(), FlopCounterMode(display=False) as fc:
                    fn()
                counted = float(fc.get_total_flops())
            except Exception as e:
                print(f"FlopCounterMode failed for {name}: {e}")
        rows.append({**a, "flops_counted_torch": counted})
    base = rows[0]["flops_encoder_plus_head"]
    for r in rows:
        r["theoretical_speedup_vs_196"] = base / r["flops_encoder_plus_head"]
    return rows


# ----------------------------------------------------------------------------- similarity
@torch.inference_mode()
def pooled_embeddings(wrapper, px_all, idx, chunk: int = 32) -> torch.Tensor:
    outs = []
    for i in range(0, px_all.shape[0], chunk):
        _, pooled = wrapper(px_all[i:i + chunk], idx)
        outs.append(pooled.float().cpu())
    return torch.cat(outs)


def run_similarity(emb_by_cfg: dict[str, torch.Tensor], tag: dict, reference: torch.Tensor | None = None):
    """Per-image cosine similarity of pooled embeddings vs the 196-token baseline."""
    ref = emb_by_cfg["196"] if reference is None else reference
    raw, agg = [], []
    for name, e in emb_by_cfg.items():
        cos = torch.nn.functional.cosine_similarity(e, ref, dim=-1)
        for i, c in enumerate(cos.tolist()):
            raw.append({**tag, "num_tokens": int(name), "image": i, "cosine": c})
        agg.append({**tag, "num_tokens": int(name), "n_images": len(cos),
                    "cosine_mean": float(cos.mean()), "cosine_std": float(cos.std(unbiased=True)),
                    "cosine_min": float(cos.min()), "cosine_median": float(cos.median())})
    return raw, agg


# ----------------------------------------------------------------------------- aggregation
def aggregate_latency(df):
    g = df.groupby(["attn_impl", "dtype", "mode", "batch", "num_tokens"])["latency_ms"]
    out = g.agg(n="count", median_ms="median", mean_ms="mean", std_ms="std", min_ms="min",
                p90_ms=lambda s: float(np.percentile(s, 90))).reset_index()
    out["throughput_img_per_s"] = out["batch"] / (out["median_ms"] / 1e3)
    base = out[out.num_tokens == 196].set_index(["attn_impl", "dtype", "mode", "batch"])["median_ms"]
    out["speedup_vs_196"] = [base.loc[(r.attn_impl, r.dtype, r.mode, r.batch)] / r.median_ms
                             for r in out.itertuples()]
    return out.sort_values(["attn_impl", "mode", "batch", "num_tokens"], ascending=[True, True, True, False])


# ----------------------------------------------------------------------------- plots
def make_plots(results_dir: Path, figures_dir: Path, primary_impl: str = "sdpa") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 120})
    env = json.loads((results_dir / "environment.json").read_text())
    params = json.loads((results_dir / "experiment_params.json").read_text())
    gpu, prec = env.get("gpu_name", env["device"]), params["benchmark_dtype"]
    lat = pd.read_csv(results_dir / "latency_summary.csv")
    lat = lat[(lat.attn_impl == primary_impl) & (lat["mode"] == "encoder_only")]
    tokens = sorted(lat.num_tokens.unique())
    marks = ["o", "s", "^", "D"]
    paths = []

    def finish(ax, ylabel, title, fname):
        ax.set_xlabel("Visual tokens per image (N)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(tokens)
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        fig = ax.figure
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = figures_dir / f"{fname}.{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(figures_dir / f"{fname}.png")
        plt.close(fig)

    sub = f"{gpu} · {prec} · attn={primary_impl} · encoder+head only"
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m, (bs, d) in zip(marks, lat.groupby("batch")):
        d = d.sort_values("num_tokens")
        ax.plot(d.num_tokens, d.median_ms, marker=m, label=f"batch {bs} (median)")
        ax.fill_between(d.num_tokens, d.median_ms, d.p90_ms, alpha=0.15)
    finish(ax, "Latency per forward pass (ms)", f"Median latency (band: median→p90)\n{sub}", "fig1_latency")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m, (bs, d) in zip(marks, lat.groupby("batch")):
        d = d.sort_values("num_tokens")
        ax.plot(d.num_tokens, d.throughput_img_per_s, marker=m, label=f"batch {bs}")
    finish(ax, "Throughput (images / s)", f"Throughput = batch / median latency\n{sub}", "fig2_throughput")

    mp = results_dir / "memory_summary.csv"
    if mp.exists():
        mem = pd.read_csv(mp)
        mem = mem[(mem.attn_impl == primary_impl) & (mem["mode"] == "encoder_only")]
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for m, (bs, d) in zip(marks, mem.groupby("batch")):
            d = d.sort_values("num_tokens")
            ax.plot(d.num_tokens, d.incremental_peak_mib, marker=m, label=f"batch {bs}")
        w = mem.weights_mib.iloc[0]
        finish(ax, "Incremental peak allocated (MiB)",
               f"Peak allocated − baseline (weights {w:.0f} MiB + inputs excluded)\n{sub}", "fig3_memory")

    sp = results_dir / "similarity_summary.csv"
    if sp.exists():
        sim = pd.read_csv(sp)
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for m, (dt, d) in zip(marks, sim.groupby("dtype")):
            d = d.sort_values("num_tokens")
            ax.errorbar(d.num_tokens, d.cosine_mean, yerr=d.cosine_std, marker=m, capsize=3,
                        label=f"{dt} (mean ± std, n={int(d.n_images.iloc[0])})")
        finish(ax, "Cosine similarity to 196-token embedding",
               "Pooled-embedding similarity (not accuracy)\nCIFAR-10 test, upsampled 32→224 px",
               "fig4_similarity")

    fp = results_dir / "flops.csv"
    if fp.exists():
        fl = pd.read_csv(fp).sort_values("num_tokens")
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(fl.num_tokens, fl.theoretical_speedup_vs_196, "k--", marker="x",
                label="theoretical (matmul FLOP ratio)")
        for m, (bs, d) in zip(marks, lat.groupby("batch")):
            d = d.sort_values("num_tokens")
            ax.plot(d.num_tokens, d.speedup_vs_196, marker=m, label=f"measured, batch {bs}")
        finish(ax, "Speed-up vs 196 tokens (×)", f"Measured vs theoretical speed-up\n{sub}",
               "fig5_speedup_vs_theory")
    return paths


# ----------------------------------------------------------------------------- README table
def results_markdown(results_dir: Path, primary_impl: str = "sdpa") -> str:
    import pandas as pd
    env = json.loads((results_dir / "environment.json").read_text())
    params = json.loads((results_dir / "experiment_params.json").read_text())
    lat = pd.read_csv(results_dir / "latency_summary.csv")
    mem = pd.read_csv(results_dir / "memory_summary.csv") if (results_dir / "memory_summary.csv").exists() else None
    sim = pd.read_csv(results_dir / "similarity_summary.csv") if (results_dir / "similarity_summary.csv").exists() else None
    fl = pd.read_csv(results_dir / "flops.csv")
    lines = [f"Measured on **{env.get('gpu_name', env['device'])}**, PyTorch {env['torch']}, "
             f"Transformers {env['transformers']}, CUDA {env.get('torch_cuda_build')}, "
             f"precision **{params['benchmark_dtype']}**, {params['timed_reps']} timed reps after "
             f"{params['warmup_iters']} warm-up iterations, run {env['timestamp_utc']}.", ""]
    impls = sorted(lat.attn_impl.unique(), key=lambda s: (s != primary_impl, s))
    for impl in impls:
        d = lat[(lat.attn_impl == impl) & (lat["mode"] == "encoder_only")]
        label = "primary" if impl == primary_impl else "ablation"
        lines += [f"**Encoder + pooling head, attention = `{impl}` ({label})**", "",
                  "| Tokens | Batch | Median (ms) | p90 (ms) | Throughput (img/s) | Speed-up | Incr. peak mem (MiB) |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in d.sort_values(["batch", "num_tokens"], ascending=[True, False]).itertuples():
            m = "–"
            if mem is not None:
                mm = mem[(mem.attn_impl == impl) & (mem["mode"] == "encoder_only") &
                         (mem.batch == r.batch) & (mem.num_tokens == r.num_tokens)]
                if len(mm):
                    m = f"{mm.incremental_peak_mib.iloc[0]:.1f}"
            lines.append(f"| {r.num_tokens} | {r.batch} | {r.median_ms:.2f} | {r.p90_ms:.2f} | "
                         f"{r.throughput_img_per_s:.0f} | {r.speedup_vs_196:.2f}× | {m} |")
        lines.append("")
    d = lat[(lat.attn_impl == primary_impl) & (lat["mode"] == "full_pipeline")]
    if len(d):
        lines += [f"**Full GPU pipeline (pixels → patch-embed all 196 → select → encoder → head), `{primary_impl}`**", "",
                  "| Tokens | Batch | Median (ms) | p90 (ms) | Throughput (img/s) | Speed-up |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for r in d.sort_values(["batch", "num_tokens"], ascending=[True, False]).itertuples():
            lines.append(f"| {r.num_tokens} | {r.batch} | {r.median_ms:.2f} | {r.p90_ms:.2f} | "
                         f"{r.throughput_img_per_s:.0f} | {r.speedup_vs_196:.2f}× |")
        lines.append("")
    lines += ["**Theoretical matmul FLOPs (encoder + head, per image)**", "",
              "| Tokens | GFLOPs (analytic) | GFLOPs (torch counter) | Quadratic attention share | Theoretical speed-up |",
              "|---:|---:|---:|---:|---:|"]
    for r in fl.sort_values("num_tokens", ascending=False).itertuples():
        c = "–" if np.isnan(r.flops_counted_torch) else f"{r.flops_counted_torch / 1e9:.2f}"
        lines.append(f"| {r.num_tokens} | {r.flops_encoder_plus_head / 1e9:.2f} | {c} | "
                     f"{r.quadratic_share * 100:.1f}% | {r.theoretical_speedup_vs_196:.2f}× |")
    lines.append("")
    if sim is not None:
        lines += ["**Pooled-embedding cosine similarity to the 196-token baseline** (not accuracy)", "",
                  "| Tokens | dtype | Mean | Std | Min | n |", "|---:|---|---:|---:|---:|---:|"]
        for r in sim.sort_values(["dtype", "num_tokens"], ascending=[True, False]).itertuples():
            lines.append(f"| {r.num_tokens} | {r.dtype} | {r.cosine_mean:.4f} | {r.cosine_std:.4f} | "
                         f"{r.cosine_min:.4f} | {r.n_images} |")
        lines.append("")
    pre = results_dir / "preprocessing_summary.csv"
    if pre.exists():
        p = pd.read_csv(pre)
        lines += ["**CPU preprocessing (SigLIP image processor, excluded from GPU timings)**: " +
                  ", ".join(f"batch {r.batch}: {r.median_ms:.1f} ms" for r in p.itertuples()), ""]
    return "\n".join(lines)


def fill_readme(readme: Path, results_dir: Path) -> None:
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    text = readme.read_text()
    assert start in text and end in text, "README results markers missing"
    table = results_markdown(results_dir)
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(f"{head}{start}\n{table}\n{end}{tail}")
    print(f"README results section updated from {results_dir}")


# ----------------------------------------------------------------------------- main
def main(args) -> None:
    import pandas as pd

    out = Path(args.out)
    results_dir, figures_dir = out / "results", out / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available - GPU latency/memory results will not be meaningful.")
    bench_dtype = DTYPES[args.dtype]
    batch_sizes = [int(b) for b in args.batch_sizes]
    modes = ["encoder_only", "selection_only", "full_pipeline"]

    env = collect_environment(device)
    print(json.dumps({k: v for k, v in env.items() if k != "nvidia_smi"}, indent=2))

    # Token configurations --------------------------------------------------
    configs = build_configs(grid_size=14, sides=(14, 12, 10, 7))
    for name, idx in configs.items():
        print(f"\n{name} tokens:\n{describe(idx)}")
    (results_dir / "token_indices.json").write_text(json.dumps(selection_table(configs), indent=1))

    # Data --------------------------------------------------------------------
    images, labels, source = load_cifar10_test(args.n_images, root=args.data_root)
    processor = load_processor(args.model_id)
    px_cpu = preprocess(processor, images)
    pre_rows = time_preprocessing(processor, images, batch_sizes)
    pre = pd.DataFrame(pre_rows)
    pre.to_csv(results_dir / "preprocessing_raw.csv", index=False)
    pre.groupby("batch")["latency_ms"].median().rename("median_ms").reset_index() \
       .to_csv(results_dir / "preprocessing_summary.csv", index=False)

    params = {
        "model_id": args.model_id, "seed": args.seed, "n_images": args.n_images,
        "dataset": source, "image_indices": f"0..{args.n_images - 1} of the test split",
        "original_resolution": "32x32 (upsampled to 224x224 by the SigLIP processor)",
        "processor": processor.to_dict() if hasattr(processor, "to_dict") else str(processor),
        "batch_sizes": batch_sizes, "warmup_iters": args.warmup, "timed_reps": args.reps,
        "benchmark_dtype": args.dtype, "attn_impls": args.attn_impls, "primary_attn_impl": args.attn_impls[0],
        "timing_modes": modes, "timer": "time.perf_counter with torch.cuda.synchronize before and after",
        "token_configs": {k: v.numel() for k, v in configs.items()},
        "selection_rule": "rows=cols=round(linspace(0,13,k)); keep Cartesian product, row-major",
    }

    lat_rows, mem_rows, sim_raw, sim_agg, val_rows = [], [], [], [], []
    flops_rows = None
    primary = args.attn_impls[0]

    # Phase 1: FP32 validation + reference embeddings (primary attention impl)
    model, _, used = load_model(args.model_id, primary, torch.float32, device)
    (results_dir / "model_config.json").write_text(model.config.to_json_string())
    print(model)
    wrapper = TokenReducedSiglipVision(model)
    px32 = px_cpu.to(device)
    v = validate_baseline(model, wrapper, px32[:8], atol=args.atol_fp32)
    val_rows.append({"check": "baseline_vs_hf_forward", "attn_impl": used, **v})
    for name, idx in configs.items():
        idx = idx.to(device)
        val_rows.append({"check": "positional_preservation", "attn_impl": used, "dtype": "torch.float32",
                         **validate_positional_preservation(wrapper, px32[:4], idx, atol=1e-5)})
    print("FP32 baseline validation passed:", val_rows[0])
    emb32 = {n: pooled_embeddings(wrapper, px32, idx.to(device)) for n, idx in configs.items()}
    r, a = run_similarity(emb32, {"dtype": "float32", "attn_impl": used})
    sim_raw += r; sim_agg += a
    del model, wrapper, px32; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Phase 2: benchmark in the benchmark dtype for each attention implementation
    for impl in args.attn_impls:
        try:
            model, wbytes, used = load_model(args.model_id, impl, bench_dtype, device)
        except RuntimeError as e:
            print(f"Skipping attn_impl={impl}: {e}")
            continue
        wrapper = TokenReducedSiglipVision(model)
        px = px_cpu.to(device=device, dtype=bench_dtype)
        dev_cfgs = {n: idx.to(device) for n, idx in configs.items()}
        tol = args.atol_fp16 if bench_dtype != torch.float32 else args.atol_fp32
        v = validate_baseline(model, wrapper, px[:8], atol=tol)
        val_rows.append({"check": "baseline_vs_hf_forward", "attn_impl": used, **v})
        print(f"{args.dtype} baseline validation ({used}) passed:", v)
        tag = {"attn_impl": used, "dtype": args.dtype}

        if impl == primary:
            emb = {n: pooled_embeddings(wrapper, px, idx) for n, idx in dev_cfgs.items()}
            if bench_dtype != torch.float32:
                r, a = run_similarity(emb, {"dtype": args.dtype, "attn_impl": used})
                sim_raw += r; sim_agg += a
            # precision noise floor: benchmark-dtype 196 vs fp32 196
            nf = torch.nn.functional.cosine_similarity(emb["196"], emb32["196"], dim=-1)
            val_rows.append({"check": f"precision_noise_floor_{args.dtype}_vs_fp32_196", "attn_impl": used,
                             "cosine_mean": float(nf.mean()), "cosine_min": float(nf.min())})
            flops_rows = counted_flops(wrapper, px, dev_cfgs, model.config)

        lat_rows += run_latency(wrapper, px, dev_cfgs, batch_sizes, modes, args.warmup, args.reps, tag)
        mem_rows += run_memory(wrapper, px, dev_cfgs, batch_sizes, ["encoder_only", "full_pipeline"], wbytes, tag)
        del model, wrapper, px; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save -------------------------------------------------------------------------
    env_after = collect_environment(device)
    env["nvidia_smi_after"] = env_after.get("nvidia_smi")
    (results_dir / "environment.json").write_text(json.dumps(env, indent=2))
    (results_dir / "experiment_params.json").write_text(json.dumps(params, indent=2, default=str))
    pd.DataFrame(val_rows).to_csv(results_dir / "validation.csv", index=False)
    lat_df = pd.DataFrame(lat_rows)
    lat_df.to_csv(results_dir / "latency_raw.csv", index=False)
    aggregate_latency(lat_df).to_csv(results_dir / "latency_summary.csv", index=False)
    if mem_rows:
        pd.DataFrame(mem_rows).to_csv(results_dir / "memory_summary.csv", index=False)
    pd.DataFrame(sim_raw).to_csv(results_dir / "similarity_raw.csv", index=False)
    pd.DataFrame(sim_agg).to_csv(results_dir / "similarity_summary.csv", index=False)
    pd.DataFrame(flops_rows).to_csv(results_dir / "flops.csv", index=False)
    figs = make_plots(results_dir, figures_dir, primary_impl=primary)
    (results_dir / "results_table.md").write_text(results_markdown(results_dir, primary_impl=primary))
    print("\n" + (results_dir / "results_table.md").read_text())
    print("Figures:", *figs, sep="\n  ")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=".")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--n-images", type=int, default=256)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--batch-sizes", nargs="+", default=[1, 8, 16])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--dtype", default="float16", choices=list(DTYPES))
    p.add_argument("--attn-impls", nargs="+", default=["sdpa", "eager"],
                   help="first entry is the primary implementation; others are labelled ablations")
    p.add_argument("--atol-fp32", type=float, default=1e-4)
    p.add_argument("--atol-fp16", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fill-readme", action="store_true", help="only re-render README results from results/")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    if a.fill_readme:
        fill_readme(Path(a.out) / "README.md", Path(a.out) / "results")
    else:
        main(a)
