"""Compare old, public and rebuilt submissions puzzle by puzzle."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from validate_submission import validate_submission


ROOT = Path(__file__).resolve().parent


def sides(path: Path, partial: bool = False) -> dict[int, float]:
    return {result.n: result.side for result in
            validate_submission(path, allow_partial=partial)}


def compare(before: dict[int, float], public: dict[int, float],
            after: dict[int, float]) -> list[dict]:
    rows = []
    for n in range(1, 1001):
        grid = math.isqrt(n - 1) + 1
        old, pub, new = before[n], public.get(n, grid), after[n]
        if new > min(old, pub) + 1e-8:
            raise ValueError(f"n={n}: rebuilt solution lost a better packing")
        scale = math.sqrt(n)
        rows.append({"n": n, "before_side": old, "public_side": pub,
                     "final_side": new,
                     "public_gain": (old - pub) / scale,
                     "total_gain": (old - new) / scale,
                     "extra_gain": (min(old, pub) - new) / scale})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value if key == "n" else f"{value:.12f}"
                             for key, value in row.items()})


def write_markdown(path: Path, rows: list[dict], label: str = "V36") -> None:
    def score(field: str) -> float:
        return math.fsum((row[field] / math.sqrt(row["n"]) - 1)
                         for row in rows)

    lines = [f"# Сравнение с публичным портфелем {label}", "",
             "Стороны измерены локальным валидатором после чтения CSV. "
             "Выигрыш считается как разность сторон, делённая на `√n`; "
             "меньший общий балл лучше.", "",
             f"- Наш файл до импорта: **{score('before_side'):.12f}**",
             f"- Публичный файл {label}: **{score('public_side'):.12f}**",
             f"- После генератора: **{score('final_side'):.12f}**",
             f"- Выигрыш нового портфеля: **{math.fsum(r['public_gain'] for r in rows):.12f}**",
             f"- Дополнительный выигрыш генератора: **{math.fsum(r['extra_gain'] for r in rows):.12f}**",
             "", "## Диапазоны n", "",
             "| n | Улучшенных публичным файлом | Выигрыш портфеля | Выигрыш сверх него |",
             "|---|---:|---:|---:|"]
    for lo in range(1, 1001, 100):
        group = [row for row in rows if lo <= row["n"] < lo + 100]
        lines.append(f"| {lo}–{lo+99} | "
                     f"{sum(row['public_gain'] > 1e-9 for row in group)} | "
                     f"{math.fsum(row['public_gain'] for row in group):.9f} | "
                     f"{math.fsum(row['extra_gain'] for row in group):.9f} |")
    lines.extend(["", "## Наибольший выигрыш от нового портфеля", "",
                  "| n | Было L | Публичное L | Итоговое L | Выигрыш балла |",
                  "|---:|---:|---:|---:|---:|"])
    for row in sorted(rows, key=lambda row: -row["public_gain"])[:30]:
        lines.append(f"| {row['n']} | {row['before_side']:.6f} | "
                     f"{row['public_side']:.6f} | {row['final_side']:.6f} | "
                     f"{row['public_gain']:.9f} |")
    lines.extend(["", "Полная таблица для всех `n=1..1000` находится в CSV. "
                  "Результат основан на локальном валидаторе; при обновлении "
                  "официальной метрики Kaggle её нужно проверить отдельно.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("public", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--csv", type=Path,
                        default=ROOT / "reports" / "portfolio_v36_comparison.csv")
    parser.add_argument("--markdown", type=Path,
                        default=ROOT / "reports" / "portfolio_v36_comparison.md")
    parser.add_argument("--label", default="V36")
    args = parser.parse_args()
    rows = compare(sides(args.before), sides(args.public, partial=True),
                   sides(args.after))
    write_csv(args.csv, rows)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.markdown, rows, args.label)
    print(f"Compared 1000 layouts; "
          f"public gain {math.fsum(r['public_gain'] for r in rows):.12f}, "
          f"extra gain {math.fsum(r['extra_gain'] for r in rows):.12f}")


if __name__ == "__main__":
    main()
