"""Plot lingua throughput (wps) of an AMD run relative to 8xh100, per config.

Usage: python plot_speedup.py [amd_dir] [baseline_dir] [out.png]
Ratio > 1 means the AMD node is faster than 8xh100. Configs missing from
either side (e.g. OOM / empty metrics.jsonl on H100) are skipped.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt

amd = Path(sys.argv[1] if len(sys.argv) > 1 else "8xmi350x-cs-amdnode3")
base = Path(sys.argv[2] if len(sys.argv) > 2 else "8xh100")
out = sys.argv[3] if len(sys.argv) > 3 else "speedup_vs_8xh100.png"
WARMUP, LAST = 2, 10  # average steps 3-10 (skip warmup); same window on both sides


def wps(path):
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = rows[WARMUP:LAST]
    return st.mean(r["speed/wps"] for r in rows) if rows else None


ratios = {}  # size -> {(bs, gas): ratio}
for d in sorted(amd.iterdir()):
    m = re.fullmatch(r"llama_(\d+B)_bs(\d+)_gas(\d+)", d.name)
    if not m:
        continue
    a, b = wps(d / "metrics.jsonl"), wps(base / d.name / "metrics.jsonl")
    if a and b:
        ratios.setdefault(m[1], {})[(int(m[2]), int(m[3]))] = a / b

fig, axes = plt.subplots(1, len(ratios), figsize=(6 * len(ratios), 4), squeeze=False)
for ax, (size, r) in zip(axes[0], sorted(ratios.items())):
    keys = sorted(r)
    ax.bar(range(len(keys)), [r[k] for k in keys],
           color=["tab:green" if r[k] >= 1 else "tab:red" for k in keys])
    ax.axhline(1, color="k", lw=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"bs{b}\ngas{g}" for b, g in keys], fontsize=7)
    ax.set_ylabel(f"wps vs {base.name} (x)")
    ax.set_title(f"llama {size}: {amd.name}")
fig.tight_layout()
fig.savefig(out, dpi=150)
for size, r in sorted(ratios.items()):
    print(size, {f"bs{b}_gas{g}": round(v, 2) for (b, g), v in sorted(r.items())})
