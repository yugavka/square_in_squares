"""Import published packing coordinates after independent geometry checks.

Inputs are text extracted from Kingbird SVGs by Ellsworth's SVG parser and
Ryan Xu's square_packing_records.json. The generator itself is fully offline:
it reads the resulting record_seeds.json, not these upstream files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from optimal_locks import SAFETY_SCALE
from validate_submission import make_square, measure_packing, squares_overlap


XU_SOURCE = ('https://github.com/ry-xu/square_packing/blob/'
             '8dc415296f697f5140caea27c7a0193d52deb4e6/'
             'square_packing_records.json')
TEXT_LINE = re.compile(
    r'Square \d+: x=([^,]+), y=([^,]+), deg=([^,]+)$'
)


def checked_side(n: int, placements: list[list[float]]) -> float:
    """Check count, strict non-overlap and measured side with export clearance."""
    if len(placements) != n:
        raise ValueError(f'n={n}: expected {n} squares, got {len(placements)}')
    if not all(len(row) == 3 and all(math.isfinite(v) for v in row)
               for row in placements):
        raise ValueError(f'n={n}: each placement must have finite x, y, angle')
    geometry = [make_square(i, *row) for i, row in enumerate(placements)]
    min_x = min(square.min_x for square in geometry)
    min_y = min(square.min_y for square in geometry)
    safe = [make_square(i, (x - min_x) * SAFETY_SCALE,
                        (y - min_y) * SAFETY_SCALE, angle)
            for i, (x, y, angle) in enumerate(placements)]
    ordered = sorted(safe, key=lambda square: square.min_x)
    active = []
    for square in ordered:
        active = [other for other in active if other.max_x > square.min_x]
        for other in active:
            if other.max_y > square.min_y and square.max_y > other.min_y:
                if squares_overlap(other, square, tolerance=0):
                    raise ValueError(
                        f'n={n}: squares {other.square_id} and '
                        f'{square.square_id} overlap after safety scaling'
                    )
        active.append(square)
    return measure_packing(n, safe).side


def catalog_entry(path: Path) -> tuple[int, dict]:
    match = re.fullmatch(r'parsed-(\d+)\.txt', path.name)
    if match is None:
        raise ValueError(f'expected parsed-N.txt: {path}')
    n = int(match.group(1))
    lines = path.read_text(encoding='utf-8').splitlines()
    if not lines or not lines[0].startswith('Final s: '):
        raise ValueError(f'{path}: missing catalog side')
    side = lines[0].split(': ', 1)[1]
    placements = []
    for line in lines[1:]:
        if not line.startswith('Square '):
            continue
        found = TEXT_LINE.fullmatch(line)
        if found is None:
            raise ValueError(f'{path}: invalid placement: {line}')
        placements.append([float(found[1]), float(found[2]),
                           math.radians(float(found[3]))])
    return n, {
        'source': f'https://kingbird.myphotos.cc/packing/square-{n}.svg',
        'catalog_side': side,
        'squares': placements,
    }


def xu_entries(path: Path) -> list[tuple[int, dict]]:
    records = json.loads(path.read_text(encoding='utf-8'))['records']
    return [(item['n'], {
        'source': XU_SOURCE,
        'published_side': item['side'],
        'squares': [[float(square[key]) for key in ('x', 'y', 'angle')]
                    for square in item['squares']],
    }) for item in records]


def import_entries(destination: Path, candidates: list[tuple[int, dict]]) -> list[int]:
    data = json.loads(destination.read_text(encoding='utf-8'))
    changed = []
    for n, entry in candidates:
        proposed_side = checked_side(n, entry['squares'])
        declared_side = float(entry.get('catalog_side',
                                        entry.get('published_side')))
        if abs(proposed_side - declared_side) > 1e-4:
            raise ValueError(f'n={n}: parsed side {proposed_side} disagrees '
                             f'with source side {declared_side}')
        previous = data.get(str(n))
        if previous is not None:
            previous_side = checked_side(n, previous['squares'])
            if proposed_side >= previous_side - 1e-8:
                continue
        data[str(n)] = entry
        changed.append(n)
    ordered = {key: data[key] for key in sorted(data, key=int)}
    destination.write_text(json.dumps(ordered, indent=2, ensure_ascii=False)
                           + '\n', encoding='utf-8')
    return sorted(set(changed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog-dir', type=Path,
                        help='directory with parsed-N.txt from Ellsworth SVG parser')
    parser.add_argument('--xu-records', type=Path,
                        help='Ryan Xu square_packing_records.json')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('record_seeds.json'))
    args = parser.parse_args()
    if args.catalog_dir is None and args.xu_records is None:
        parser.error('provide --catalog-dir and/or --xu-records')
    candidates = []
    if args.catalog_dir is not None:
        candidates.extend(catalog_entry(path)
                          for path in sorted(args.catalog_dir.glob('parsed-*.txt')))
    if args.xu_records is not None:
        candidates.extend(xu_entries(args.xu_records))
    changed = import_entries(args.output, candidates)
    print(f'Imported {len(changed)} improved seed sizes: {changed}')


if __name__ == '__main__':
    main()
