"""Import strictly better layouts from the best public Kaggle notebooks.

The script lists competition notebooks by public score, skips runs that are
not complete, downloads each notebook's submission.csv and checks it with the
local validator and an exact check of their decimal text. Partial files are accepted: Kaggle scores missing n as grids. Layouts whose side is strictly smaller than every snapshot already
used by the generator are copied with their original decimal rows into a
deterministic gzip archive in published_sources/auto. The import log
published_sources/auto/imports.json records the notebook run time, file
hashes and the selected n; packing_solution.py reads every archive listed there.

Requires the kaggle package and an API token; the generator itself does not.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from exact_overlap import overlapping, read_text
from optimal_locks import proven_optimal
from validate_submission import ValidationError, total_score, validate_submission

ROOT = Path(__file__).resolve().parent
AUTO_DIR = ROOT / "published_sources" / "auto"
IMPORT_LOG = AUTO_DIR / "imports.json"
COMPETITION = "squares-in-squares"


def load_imports(path: Path = IMPORT_LOG) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def imported_archives(path: Path = IMPORT_LOG) -> list[Path]:
    """Archives in import order; the generator adds them to its portfolio."""
    return [path.parent / entry["archive"] for entry in load_imports(path)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_sides(paths: Iterable[Path]) -> dict[int, float]:
    """Smallest validated side per n over partial or complete CSV snapshots."""
    best: dict[int, float] = {}
    for path in paths:
        for result in validate_submission(path, allow_partial=True):
            best[result.n] = min(best.get(result.n, math.inf), result.side)
    return best


def select_better(candidate: dict[int, float],
                  baseline: dict[int, float]) -> list[int]:
    """Unlocked n whose candidate beats both the grid and every snapshot."""
    return sorted(
        n for n, side in candidate.items()
        if proven_optimal(n) is None
        and side < math.isqrt(n - 1) + 1
        and side < baseline.get(n, math.inf)
    )


def write_delta(source: Path, keep: Iterable[int], target: Path) -> int:
    """Copy the original rows for the kept n into a reproducible gzip."""
    selected = set(keep)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    rows = 0
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        writer.writerow(next(reader))
        for row in reader:
            if int(row[1]) in selected:
                writer.writerow(row)
                rows += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                           compresslevel=9) as archive:
            archive.write(buffer.getvalue().encode("utf-8"))
    return rows


def run_stamp(last_run_time: datetime | None) -> str:
    """Identify a notebook run; the listing API omits version numbers."""
    if last_run_time is None:
        return "unknown"
    return last_run_time.strftime("%Y%m%dT%H%M%S")


def archive_name(ref: str, stamp: str) -> str:
    owner, slug = ref.split("/", 1)
    return f"{owner}__{slug}__{stamp}.csv.gz"


def list_notebooks(api, top: int) -> list:
    kernels = api.kernels_list(competition=COMPETITION, sort_by="scoreAscending",
                               page_size=min(max(top, 1), 100)) or []
    return [kernel for kernel in kernels if kernel is not None][:top]


def download_submission(api, ref: str, directory: Path) -> Path | None:
    """Fetch only files named submission.csv and keep the shallowest one."""
    api.kernels_output(ref, str(directory), file_pattern=r"(^|/)submission\.csv$",
                       quiet=True)
    found = sorted(directory.rglob("submission.csv"),
                   key=lambda path: (len(path.parts), str(path)))
    return found[0] if found else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=10,
                        help="number of best-scoring public notebooks to check")
    parser.add_argument("--min-gain", type=float, default=1e-9,
                        help="skip a notebook whose total score gain is smaller")
    parser.add_argument("--dry-run", action="store_true",
                        help="report gains without writing archives")
    args = parser.parse_args()
    if not 1 <= args.top <= 100:
        parser.error("--top must be in 1..100")

    from kaggle.api.kaggle_api_extended import KaggleApi
    from packing_solution import fixed_public_portfolios

    api = KaggleApi()
    api.authenticate()
    imports = load_imports()
    seen = {(entry["ref"], entry["run"]) for entry in imports}
    baseline = snapshot_sides(list(fixed_public_portfolios()) + imported_archives())
    added = 0
    for kernel in list_notebooks(api, args.top):
        ref, run = kernel.ref, run_stamp(kernel.last_run_time)
        label = f"{ref} @ {run}"
        if (ref, run) in seen:
            print(f"{label}: already imported")
            continue
        try:
            status = api.kernels_status(ref).status
        except Exception as error:  # network or permission problems
            print(f"{label}: status unavailable: {error}")
            continue
        if str(status).upper().split(".")[-1] != "COMPLETE":
            print(f"{label}: run status {status}, skipped")
            continue
        with tempfile.TemporaryDirectory() as directory:
            try:
                path = download_submission(api, ref, Path(directory))
            except Exception as error:  # network or permission problems
                print(f"{label}: download failed: {error}")
                continue
            if path is None:
                print(f"{label}: no submission.csv in output")
                continue
            try:
                results = validate_submission(path, allow_partial=True)
            except ValidationError as error:
                print(f"{label}: rejected by validator: {error}")
                continue
            sides = {result.n: result.side for result in results}
            keep = select_better(sides, baseline)
            # The validator tolerates 1e-9 overlaps; Kaggle certifies text.
            rejected = overlapping(read_text(path), only=keep)
            if rejected:
                print(f"{label}: {len(rejected)} layouts overlap in exact "
                      f"decimal arithmetic, dropped")
                keep = [n for n in keep if n not in rejected]
            gain = sum((min(baseline.get(n, math.inf), math.isqrt(n - 1) + 1)
                        - sides[n]) / math.sqrt(n) for n in keep)
            score = total_score(results)
            print(f"{label}: score {score:.12f}, "
                  f"{len(keep)} better layouts, gain {gain:.3e}")
            if not keep or gain < args.min_gain or args.dry_run:
                continue
            name = archive_name(ref, run)
            rows = write_delta(path, keep, AUTO_DIR / name)
            imports.append({
                "ref": ref,
                "run": run,
                "title": kernel.title,
                "url": f"https://www.kaggle.com/code/{ref}",
                "last_run_time": (kernel.last_run_time.isoformat()
                                  if kernel.last_run_time else None),
                "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source_sha256": sha256(path),
                "source_score": round(score, 12),
                "estimated_gain": round(gain, 12),
                "rows": rows,
                "n": keep,
                "archive": name,
                "archive_sha256": sha256(AUTO_DIR / name),
            })
            for n in keep:
                baseline[n] = sides[n]
            added += 1
    if added:
        AUTO_DIR.mkdir(parents=True, exist_ok=True)
        IMPORT_LOG.write_text(json.dumps(imports, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
        print(f"Imported {added} notebooks; rebuild with python3 packing_solution.py")
    else:
        print("No new strictly better layouts")


if __name__ == "__main__":
    main()
