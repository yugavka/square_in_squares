"""Report which fault-line gadgets survive in the final 1000 layouts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from functools import lru_cache

from optimal_locks import fixed_packing, proven_optimal
from packing_solution import FaultTrace, improved_packings
from validate_submission import make_square, measure_packing


ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def locked_side(n: int) -> float:
    _, squares = fixed_packing(n)
    return measure_packing(n, [make_square(i, *row)
                               for i, row in enumerate(squares)]).side


def final_side(n: int, packings: dict) -> float:
    if proven_optimal(n) is not None:
        return locked_side(n)
    return packings[n][0] if n in packings else math.isqrt(n - 1) + 1


def build_report(baseline: dict, actual: dict,
                 trace: dict[int, FaultTrace]) -> dict:
    puzzles = []
    summary = defaultdict(lambda: {"uses": set(), "occurrences": 0,
                                   "allocated_gain": 0.0,
                                   "associated_gain": 0.0,
                                   "unique_bottleneck_uses": 0})
    gains = []
    for n in range(1, 1001):
        old, new = final_side(n, baseline), final_side(n, actual)
        gain = (old - new) / math.sqrt(n)
        if gain < -1e-8:
            raise AssertionError(f"DP worsened n={n}: {old} -> {new}")
        gains.append(gain)
        if n not in trace:
            if gain > 1e-8:
                raise AssertionError(f"n={n} improved without DP provenance")
            continue
        lineage = trace[n]
        signatures = [(g.w, g.h, g.surplus) for g in lineage.gadgets]
        overhangs = [g.overhang for g in lineage.gadgets]
        max_f = max(overhangs)
        bottlenecks = [i for i, f in enumerate(overhangs)
                       if f >= max_f - 1e-8]
        for i, (g, key) in enumerate(zip(lineage.gadgets, signatures)):
            item = summary[key]
            item["uses"].add(n)
            item["occurrences"] += 1
            item["allocated_gain"] += gain / len(signatures)
            if signatures.index(key) == i:
                item["associated_gain"] += gain
            if len(bottlenecks) == 1 and bottlenecks[0] == i:
                item["unique_bottleneck_uses"] += 1
            item["overhang"] = g.overhang
        puzzles.append({"n": n, "baseline_side": old, "final_side": new,
                        "score_gain": gain, "source_n": lineage.source_n,
                        "steps": list(lineage.steps),
                        "gadgets": [list(key) for key in signatures]})
    gadgets = [{"w": w, "h": h, "surplus": surplus,
                "overhang": item["overhang"],
                "uses": len(item["uses"]),
                "n": sorted(item["uses"]),
                "occurrences": item["occurrences"],
                "allocated_gain": item["allocated_gain"],
                "associated_gain": item["associated_gain"],
                "unique_bottleneck_uses": item["unique_bottleneck_uses"]}
               for (w, h, surplus), item in summary.items()]
    gadgets.sort(key=lambda row: (-row["allocated_gain"], row["w"],
                                  row["h"], row["surplus"]))
    baseline_score = math.fsum(final_side(n, baseline) / math.sqrt(n) - 1
                               for n in range(1, 1001))
    final_score = math.fsum(final_side(n, actual) / math.sqrt(n) - 1
                            for n in range(1, 1001))
    total_gain = math.fsum(gains)
    allocated = math.fsum(row["allocated_gain"] for row in gadgets)
    if not math.isclose(total_gain, baseline_score - final_score, abs_tol=1e-10):
        raise AssertionError("per-n gains do not sum to score difference")
    if not math.isclose(total_gain, allocated, abs_tol=1e-10):
        raise AssertionError("allocated gadget gains do not sum to score difference")
    return {"baseline_score": baseline_score, "final_score": final_score,
            "score_gain": total_gain, "affected_n": len(puzzles),
            "direct_n": sum(not row["steps"] for row in puzzles),
            "derived_n": sum(bool(row["steps"]) for row in puzzles),
            "gadgets": gadgets, "puzzles": puzzles}


def render_markdown(report: dict) -> str:
    lines = ["# Вклад прямоугольных фрагментов",
             "", "Сравнение полного генератора с режимом `--no-fault-dp`.",
             "Локальный балл: меньше — лучше.", "",
             f"- Без DP: **{report['baseline_score']:.12f}**",
             f"- С DP: **{report['final_score']:.12f}**",
             f"- Снижение балла: **{report['score_gain']:.12f}**",
             f"- Итоговых раскладок из DP: **{report['affected_n']}** "
             f"({report['direct_n']} прямых, {report['derived_n']} после удаления)",
             "", "## Фрагменты по распределённому выигрышу", "",
             "`w×h+d` означает размеры фрагмента и число квадратов сверх `w·h`; "
             "`f` — его выступ за целочисленную рамку.",
             "Выигрыш каждого `n` делится поровну между вхождениями фрагментов "
             "в его исходную цепочку. Это учётный вклад, а не независимый "
             "эффект замены одного фрагмента. Связанный выигрыш считает всю "
             "прибавку `n` для каждого фрагмента цепочки и поэтому не суммируется.",
             "Уникальный максимум `f` показывает, где один фрагмент ограничивает "
             "порог цепочки; это подсказка для следующего поиска, не гарантия "
             "улучшения итоговой стороны.", "",
             "| Фрагмент | f | n | Вхождений | Распределённый выигрыш | Связанный выигрыш | Уникальный максимум f | Размеры n |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for g in report["gadgets"]:
        lines.append(f"| {g['w']}×{g['h']}+{g['surplus']} | "
                     f"{g['overhang']:.6f} | {g['uses']} | {g['occurrences']} | "
                     f"{g['allocated_gain']:.9f} | {g['associated_gain']:.9f} | "
                     f"{g['unique_bottleneck_uses']} | "
                     f"{', '.join(map(str, g['n']))} |")
    bottlenecks = sorted(report["gadgets"],
                         key=lambda row: (-row["unique_bottleneck_uses"],
                                          -row["allocated_gain"]))
    lines.extend(["", "## Кандидаты для следующего поиска", "",
                  "Фрагменты ниже чаще других одни задают максимальный `f` "
                  "в используемых цепочках. Уменьшение их `f` стоит проверить "
                  "в первую очередь; итоговый выигрыш нужно измерять новым "
                  "полным прогоном.", ""])
    for g in bottlenecks[:5]:
        lines.append(f"- **{g['w']}×{g['h']}+{g['surplus']}**: "
                     f"{g['unique_bottleneck_uses']} случаев с единственным "
                     f"максимумом `f={g['overhang']:.6f}`; используется в "
                     f"{g['uses']} итоговых раскладках.")
    lines.extend(["", "## Итоговые раскладки с происхождением из DP", "",
                  "`Источник` — число квадратов при исходной сборке цепочки. "
                  "Полный путь удаления записан в JSON.", "",
                  "| n | Источник | Шагов удаления | Сторона без DP | Итоговая сторона | Выигрыш балла | Фрагменты |",
                  "|---:|---:|---:|---:|---:|---:|---|"])
    for p in report["puzzles"]:
        chain = ", ".join(f"{w}×{h}+{d}" for w, h, d in p["gadgets"])
        lines.append(f"| {p['n']} | {p['source_n']} | {len(p['steps'])} | "
                     f"{p['baseline_side']:.6f} | {p['final_side']:.6f} | "
                     f"{p['score_gain']:.9f} | {chain} |")
    lines.extend(["", "Метод: для каждого `n` сравниваются стороны итоговых "
                  "раскладок двух полных прогонов; разность баллов равна "
                  "`(L_без_DP − L_с_DP)/√n`. Отчёт использует стороны в памяти "
                  "до сериализации CSV и отдельно измеряет зафиксированные "
                  "доказанные раскладки по их координатам.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path,
                        default=ROOT / "reports" / "fault_line_report.md")
    parser.add_argument("--json", type=Path,
                        default=ROOT / "reports" / "fault_line_report.json")
    args = parser.parse_args()
    baseline = improved_packings(include_fault_dp=False)
    trace: dict[int, FaultTrace] = {}
    actual = improved_packings(trace=trace)
    report = build_report(baseline, actual, trace)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print(f"Score {report['baseline_score']:.12f} -> {report['final_score']:.12f}; "
          f"{report['affected_n']} DP-derived n, {len(report['gadgets'])} gadgets")
    print(args.markdown)
    print(args.json)


if __name__ == "__main__":
    main()
