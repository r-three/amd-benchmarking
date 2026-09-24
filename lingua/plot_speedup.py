"""Heatmaps of lingua throughput (wps at step 10) of one run relative to a reference run.

Usage: python plot_speedup.py [cmp_dir] [ref_dir ...]
Writes speedup_<cmp>_vs_<ref>.png per reference. One panel per model size:
x = sequences per GPU (bs), y = grad accumulation steps (gas), cell = cmp/ref
wps (>1: cmp faster). White cell = reference has no data (e.g. OOM); a green
check on it means the comparison run did complete there.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

cmp_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "8xmi350x-cs-amdnode3")
refs = [Path(p) for p in (sys.argv[2:] or ["8xh100", "8xmi300x"])]
STEP = 10  # compare wps at this global step on both sides (early steps still warming up)
BS, GAS = [1, 2, 4, 8], [1, 2, 4, 8]


def wps(path):
    if not path.exists():
        return None
    for l in path.read_text().splitlines():
        if l.strip() and (r := json.loads(l))["global_step"] == STEP:
            return r["speed/wps"]  # first occurrence, in case the file holds several runs
    return None


def plot(ref):
    sizes = sorted({re.match(r"llama_(\d+B)", d.name)[1] for d in cmp_dir.iterdir()})
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 4.4), squeeze=False)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("white")
    for ax, size in zip(axes[0], sizes):
        grid = np.full((len(GAS), len(BS)), np.nan)
        ok = np.zeros_like(grid, dtype=bool)
        for i, g in enumerate(GAS):
            for j, b in enumerate(BS):
                name = f"llama_{size}_bs{b}_gas{g}"
                a = wps(cmp_dir / name / "metrics.jsonl")
                r = wps(ref / name / "metrics.jsonl")
                if r and a:
                    grid[i, j] = a / r
                elif a:
                    ok[i, j] = True  # ref missing/OOM, cmp ran
        im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", cmap=cmap,
                       norm=LogNorm(0.25, 4.0))
        for i in range(len(GAS)):
            for j in range(len(BS)):
                if ok[i, j]:
                    ax.text(j, i, "✓", ha="center", va="center", color="green", fontsize=22, weight="bold")
                elif not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}x", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(BS)), BS)
        ax.set_yticks(range(len(GAS)), GAS)
        ax.set_xlabel("sequences per GPU")
        ax.set_ylabel("grad accumulation steps")
        ax.set_title(f"llama {size}")
        for k in range(len(BS) + 1):  # thin grid so white cells stay visible
            ax.axvline(k - 0.5, color="lightgray", lw=0.5)
            ax.axhline(k - 0.5, color="lightgray", lw=0.5)
    fig.suptitle(f"{cmp_dir.name} wps relative to {ref.name}  (white+✓: ref has no data, cmp ran)")
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="wps ratio")
    cb.set_ticks([0.25, 0.5, 1, 2, 4])
    cb.set_ticklabels(["0.25x", "0.5x", "1x", "2x", "4x"])
    out = f"speedup_{cmp_dir.name}_vs_{ref.name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


for ref in refs:
    plot(ref)
