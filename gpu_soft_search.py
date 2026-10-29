"""Run the batched soft packing kernel on a Kaggle GPU and check its output.

``build`` writes a private script kernel into gpu_kernel/build/<slug> with one
job per target n (box sides and incumbent layout) embedded in
gpu_kernel/soft_gpu.py. Box sides are given directly with --targets for one n
or as offsets below each incumbent side with --deltas.
``push`` builds and uploads it with GPU enabled; ``fetch`` downloads
candidates.json and summary.json, repairs each candidate with the fixed-angle
LP, validates it strictly and writes only improvements to a partial CSV that
packing_solution.py accepts through --portfolio; ``check`` repeats this step
on an existing download. ``local`` runs the built kernel on this machine for
a smoke test. Requires the kaggle package and an API token for push and fetch, PyTorch for the kernel and SciPy for fetch.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

from optimal_locks import proven_optimal
from projected_search import (load_selected, normalized, restore_lp,
                              strict_side, write_partial)
from validate_submission import make_square, measure_packing, validate_submission

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "gpu_kernel" / "soft_gpu.py"
BUILD = ROOT / "gpu_kernel" / "build"


def slug_for(ns: list[int], tag: str) -> str:
    stem = f"soft-gpu-{ns[0]}" if len(ns) == 1 else "soft-gpu"
    return f"{stem}-{tag}" if tag else stem


def build(ns: list[int], targets: list[float] | None, deltas: list[float] | None,
          minutes: float, batch: int, kinds: list[str], seed: int, keep: float,
          top: int, slug: str, username: str, source: Path) -> Path:
    layouts = load_selected(source, set(ns))
    jobs = []
    for n in ns:
        incumbent = normalized(layouts[n])
        # Public layouts may touch exactly, which strict_side rejects.
        side = measure_packing(n, [make_square(i, *row)
                                   for i, row in enumerate(incumbent)]).side
        boxes = targets if targets else [round(side - delta, 9) for delta in deltas]
        jobs.append({"n": n, "targets": boxes, "incumbent_side": side,
                     "incumbent": [list(map(float, row)) for row in incumbent]})
    config = {"jobs": jobs, "minutes": minutes, "batch": batch, "seed": seed,
              "kinds": kinds, "keep": keep, "top": top}
    code = SOURCE.read_text(encoding="utf-8")
    marker = code.index("CONFIG = {")
    end = code.index("}\n", marker) + 2
    code = code[:marker] + "CONFIG = " + json.dumps(config) + "\n" + code[end:]
    folder = BUILD / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "soft_gpu.py").write_text(code, encoding="utf-8")
    metadata = {"id": f"{username}/{slug}", "title": slug, "code_file": "soft_gpu.py",
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_internet": False, "dataset_sources": [],
                "competition_sources": [], "kernel_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                 encoding="utf-8")
    return folder


def check(candidates: list[dict], n: int, incumbent: float,
          seed: int) -> tuple[list[dict], tuple[float, list] | None]:
    """Polish one n's candidates with L-BFGS, LP-repair them, keep valid sides."""
    import numpy as np
    from soft_global_search import settle

    rng = random.Random(seed)
    records, best = [], None
    for item in candidates:
        vec, energy = settle(np.asarray(item["layout"], dtype=float).ravel(), item["side"])
        item = {**item, "polished": energy}
        layout = [tuple(map(float, row)) for row in vec.reshape(-1, 3)]
        repaired = restore_lp(layout, item["side"] + 1e-8, trust=.05, rng=rng,
                              time_limit=30.0)
        side = None
        if repaired is not None and len(repaired) == n:
            try:
                side = strict_side(repaired)
            except ValueError:
                side = None
        records.append({"side_target": item["side"], "kind": item["kind"],
                        "energy": item["energy"], "polished": item["polished"],
                        "side": side})
        if side is not None and side < incumbent - 1e-9 and (best is None or side < best[0]):
            best = (side, repaired)
    return records, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "local", "push", "fetch", "check"))
    parser.add_argument("ns", type=int, nargs="+", metavar="n")
    parser.add_argument("--targets", type=float, nargs="+",
                        help="box sides, only with a single n")
    parser.add_argument("--deltas", type=float, nargs="+",
                        help="box sides as offsets below each incumbent side")
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--kinds", nargs="+",
                        default=["incumbent", "block", "band", "mixed", "random"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", type=float, default=1e-4)
    parser.add_argument("--top", type=int, default=8,
                        help="best layouts of every batch kept for local polish")
    parser.add_argument("--tag", default="")
    parser.add_argument("--username", default="borisbezmenov")
    parser.add_argument("--input", type=Path, default=ROOT / "submission_improved.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "gpu_soft_candidates.csv")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if any(proven_optimal(n) is not None or not 1 < n <= 1000 for n in args.ns):
        parser.error("choose unlocked n")
    if len(args.ns) > 1 and not args.tag:
        parser.error("several n need a --tag for the kernel name")
    slug = slug_for(args.ns, args.tag)
    folder = BUILD / slug
    if args.action in ("build", "local", "push"):
        if bool(args.targets) == bool(args.deltas) or (args.targets and len(args.ns) > 1):
            parser.error("give --targets for one n or --deltas")
        folder = build(args.ns, args.targets, args.deltas, args.minutes, args.batch,
                       args.kinds, args.seed, args.keep, args.top, slug,
                       args.username, args.input)
        print(f"built {folder}")
        if args.action == "local":
            subprocess.run([sys.executable, "soft_gpu.py"], cwd=folder, check=True)
        elif args.action == "push":
            subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p",
                            str(folder)], check=True)
        return
    out = folder / "output"
    if args.action == "fetch":
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output",
                        f"{args.username}/{slug}", "-p", str(out), "-o"], check=True)
    sides = {r.n: r.side for r in validate_submission(args.input)}
    candidates = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    report = {"kernel": f"{args.username}/{slug}", "device": summary.get("device"),
              "gpu": summary.get("name"), "config": summary.get("config"),
              "batches": summary.get("batches"), "results": {}}
    improved = {}
    for n in args.ns:
        # Single-n kernels before the job format wrote candidates without n.
        mine = [c for c in candidates if c.get("n", args.ns[0]) == n]
        records, best = check(mine, n, sides[n], args.seed)
        report["results"][str(n)] = {
            "incumbent_side": sides[n], "checked": records,
            "best_side": min((r["side"] for r in records if r["side"]), default=None)}
        if best is not None:
            improved[n] = best[1]
            report["results"][str(n)]["score_gain"] = (sides[n] - best[0]) / math.sqrt(n)
        valid = sum(r["side"] is not None for r in records)
        print(f"n={n}: {len(records)} candidates, {valid} valid, best "
              f"{report['results'][str(n)]['best_side']}, incumbent {sides[n]:.9f}")
    if improved:
        write_partial(args.output, improved)
        report["candidate_csv"] = str(args.output)
    target = args.report or ROOT / "reports" / f"gpu_{slug}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report {target}")


if __name__ == "__main__":
    main()
