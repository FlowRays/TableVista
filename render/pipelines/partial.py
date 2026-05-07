from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from rendering.parsers.latex_table import ParsedTable, normalize_rectangular


@dataclass(frozen=True)
class RenderSpec:
    tables: List[ParsedTable]


def apply_table_partial(
    spec: RenderSpec, rng: random.Random
) -> Tuple[RenderSpec, Dict[str, Any]]:
    if len(spec.tables) != 1:
        return spec, {"mode": "unchanged"}

    table = spec.tables[0]
    total_rows = len(table.matrix)
    total_cols = len(table.matrix[0]) if total_rows else 0
    header_rows = max(1, min(int(table.header_rows), total_rows or 1))
    left_cols = max(0, min(int(table.left_header_cols), total_cols))
    data_rows = max(0, total_rows - header_rows)
    data_cols = max(0, total_cols - left_cols)

    modes: List[str] = []
    weights: List[float] = []

    if data_cols >= 2:
        modes.append("split_cols_2")
        weights.append(0.55 if data_cols >= 4 else 0.40)
    if data_cols >= 3:
        modes.append("split_cols_3")
        weights.append(0.35 if data_cols >= 6 else 0.22)

    if data_rows >= 2:
        modes.append("split_rows_2")
        weights.append(0.28 if data_rows >= 6 else 0.18)
    if data_rows >= 3:
        modes.append("split_rows_3")
        weights.append(0.18 if data_rows >= 9 else 0.12)

    if data_rows >= 1 and header_rows >= 2:
        modes.append("header_body")
        weights.append(0.06)

    if not modes:
        return RenderSpec(tables=[table, table]), {
            "mode": "duplicate",
            "panel_count": 2,
            "orientation": "vertical",
        }

    mode = rng.choices(modes, weights=weights, k=1)[0]

    if mode == "split_rows_2":
        a, b, meta = _partial_split_rows(table, rng)
        return RenderSpec(tables=[a, b]), {
            "mode": mode,
            "panel_count": 2,
            "orientation": "vertical",
            **meta,
        }

    if mode == "split_rows_3":
        parts, meta = _partial_split_rows_three(table, rng)
        return RenderSpec(tables=parts), {
            "mode": mode,
            "panel_count": 3,
            "orientation": "vertical",
            **meta,
        }

    if mode == "split_cols_2":
        a, b, meta = _partial_split_cols(table, rng)
        return RenderSpec(tables=[a, b]), {
            "mode": mode,
            "panel_count": 2,
            "orientation": "horizontal",
            **meta,
        }

    if mode == "split_cols_3":
        parts, meta = _partial_split_cols_three(table, rng)
        return RenderSpec(tables=parts), {
            "mode": mode,
            "panel_count": 3,
            "orientation": "horizontal",
            **meta,
        }

    a, b, meta = _partial_split_header_body(table)
    return RenderSpec(tables=[a, b]), {
        "mode": mode,
        "panel_count": 2,
        "orientation": "vertical",
        **meta,
    }


def _partial_split_rows(
    table: ParsedTable, rng: random.Random
) -> Tuple[ParsedTable, ParsedTable, Dict[str, Any]]:
    matrix = table.matrix
    header_rows = max(1, min(table.header_rows, len(matrix) or 1))
    data = matrix[header_rows:]
    if len(data) < 2:
        raise ValueError("split_rows_2 requires at least two data rows.")

    split = 1 if len(data) == 2 else rng.randint(1, len(data) - 1)
    overlap = 1 if (len(data) >= 3 and split >= 2 and rng.random() < 0.35) else 0
    a_end = min(len(data), split + overlap)
    b_start = max(0, split - overlap)
    a_data = data[:a_end]
    b_data = data[b_start:]
    if not a_data or not b_data:
        overlap = 0
        a_data = data[:split]
        b_data = data[split:]

    a = matrix[:header_rows] + a_data
    b = matrix[:header_rows] + b_data

    t1 = ParsedTable(
        matrix=normalize_rectangular(a),
        header_rows=header_rows,
        left_header_cols=table.left_header_cols,
        format=table.format,
    )
    t2 = ParsedTable(
        matrix=normalize_rectangular(b),
        header_rows=header_rows,
        left_header_cols=table.left_header_cols,
        format=table.format,
    )
    return (
        t1,
        t2,
        {
            "split_at": split,
            "overlap_rows": overlap,
            "rows_a": [0, header_rows + len(a_data)],
            "rows_b": [0, header_rows + len(b_data)],
        },
    )


def _partial_split_cols(
    table: ParsedTable, rng: random.Random
) -> Tuple[ParsedTable, ParsedTable, Dict[str, Any]]:
    matrix = table.matrix
    total_rows = len(matrix)
    total_cols = len(matrix[0]) if total_rows else 0
    header_rows = max(1, min(int(table.header_rows), total_rows or 1))
    left_cols = max(0, min(int(table.left_header_cols), total_cols))
    remaining = max(0, total_cols - left_cols)
    if remaining < 2:
        raise ValueError("split_cols_2 requires at least two data columns.")

    split = rng.randint(1, remaining - 1)
    overlap = 1 if rng.random() < 0.30 else 0

    a_cols = list(range(0, left_cols)) + list(
        range(left_cols, left_cols + split + overlap)
    )
    b_cols = list(range(0, left_cols)) + list(
        range(left_cols + split - overlap, total_cols)
    )

    if len(set(a_cols)) <= left_cols or len(set(b_cols)) <= left_cols:
        overlap = 0
        a_cols = list(range(0, left_cols)) + list(range(left_cols, left_cols + split))
        b_cols = list(range(0, left_cols)) + list(range(left_cols + split, total_cols))

    a = [[row[c] for c in a_cols] for row in matrix]
    b = [[row[c] for c in b_cols] for row in matrix]

    t1 = ParsedTable(
        matrix=normalize_rectangular(a),
        header_rows=header_rows,
        left_header_cols=min(left_cols, len(a_cols)),
        format=table.format,
    )
    t2 = ParsedTable(
        matrix=normalize_rectangular(b),
        header_rows=header_rows,
        left_header_cols=min(left_cols, len(b_cols)),
        format=table.format,
    )
    return (
        t1,
        t2,
        {
            "split_at": left_cols + split,
            "overlap_cols": overlap,
            "cols_a": a_cols,
            "cols_b": b_cols,
        },
    )


def _partial_split_header_body(
    table: ParsedTable,
) -> Tuple[ParsedTable, ParsedTable, Dict[str, Any]]:
    matrix = table.matrix
    total_rows = len(matrix)
    header_rows = max(1, min(int(table.header_rows), total_rows or 1))
    if total_rows <= header_rows:
        raise ValueError("header_body split requires at least one body row.")

    head = matrix[:header_rows]
    body = matrix[header_rows:]

    head_table = ParsedTable(
        matrix=normalize_rectangular(head),
        header_rows=header_rows,
        left_header_cols=table.left_header_cols,
        format=table.format,
    )
    body_table = ParsedTable(
        matrix=normalize_rectangular(body),
        header_rows=0,
        left_header_cols=table.left_header_cols,
        format=table.format,
    )
    return (
        head_table,
        body_table,
        {"mode": "header_body", "header_rows": header_rows, "body_rows": len(body)},
    )


def _partial_split_rows_three(
    table: ParsedTable, rng: random.Random
) -> Tuple[List[ParsedTable], Dict[str, Any]]:
    matrix = table.matrix
    header_rows = max(1, min(table.header_rows, len(matrix) or 1))
    data = matrix[header_rows:]
    if len(data) < 3:
        raise ValueError("split_rows_3 requires at least three data rows.")

    s1 = rng.randint(1, len(data) - 2)
    s2 = rng.randint(s1 + 1, len(data) - 1)
    overlap = 1 if rng.random() < 0.25 else 0

    def seg(start: int, end: int) -> List[List[str]]:
        start = max(0, start)
        end = max(start + 1, min(len(data), end))
        return data[start:end]

    seg1 = seg(0, s1 + overlap)
    seg2 = seg(s1 - overlap, s2 + overlap)
    seg3 = seg(s2 - overlap, len(data))

    if not seg1 or not seg2 or not seg3:
        overlap = 0
        seg1 = seg(0, s1)
        seg2 = seg(s1, s2)
        seg3 = seg(s2, len(data))

    parts: List[ParsedTable] = []
    for seg_data in (seg1, seg2, seg3):
        sub = matrix[:header_rows] + seg_data
        parts.append(
            ParsedTable(
                matrix=normalize_rectangular(sub),
                header_rows=header_rows,
                left_header_cols=table.left_header_cols,
                format=table.format,
            )
        )

    meta = {
        "split_at": [s1, s2],
        "overlap_rows": overlap,
        "segments": [len(seg1), len(seg2), len(seg3)],
    }
    return parts, meta


def _partial_split_cols_three(
    table: ParsedTable, rng: random.Random
) -> Tuple[List[ParsedTable], Dict[str, Any]]:
    matrix = table.matrix
    total_rows = len(matrix)
    total_cols = len(matrix[0]) if total_rows else 0
    header_rows = max(1, min(int(table.header_rows), total_rows or 1))
    left_cols = max(0, min(int(table.left_header_cols), total_cols))
    remaining = max(0, total_cols - left_cols)

    if remaining < 3:
        raise ValueError("split_cols_3 requires at least three data columns.")

    s1 = rng.randint(1, remaining - 2)
    s2 = rng.randint(s1 + 1, remaining - 1)
    overlap = 1 if rng.random() < 0.25 else 0

    def seg(start: int, end: int) -> List[int]:
        start = max(0, start)
        end = max(start + 1, min(remaining, end))
        cols = list(range(left_cols + start, left_cols + end))
        return list(range(0, left_cols)) + cols

    seg1 = seg(0, s1 + overlap)
    seg2 = seg(s1 - overlap, s2 + overlap)
    seg3 = seg(s2 - overlap, remaining)

    if (
        len(set(seg1)) <= left_cols
        or len(set(seg2)) <= left_cols
        or len(set(seg3)) <= left_cols
    ):
        overlap = 0
        seg1 = seg(0, s1)
        seg2 = seg(s1, s2)
        seg3 = seg(s2, remaining)

    parts: List[ParsedTable] = []
    for cols in (seg1, seg2, seg3):
        sub = [[row[c] for c in cols] for row in matrix]
        parts.append(
            ParsedTable(
                matrix=normalize_rectangular(sub),
                header_rows=header_rows,
                left_header_cols=min(left_cols, len(cols)),
                format=table.format,
            )
        )

    meta = {
        "col_split_at": [left_cols + s1, left_cols + s2],
        "overlap_cols": overlap,
        "cols_parts": [seg1, seg2, seg3],
    }
    return parts, meta
