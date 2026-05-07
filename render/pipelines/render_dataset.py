from __future__ import annotations

import json
import os
import random
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

from postprocess.image_transforms import apply_noise, apply_photo, apply_structural
from rendering.screenshot import PlaywrightScreenshotSession
from rendering.table_schema import CanonicalTable, create_simple_table
from rendering.parsers.latex_table import ParsedTable, parse_latex_tables
from pipelines.io import ensure_parent, read_jsonl
from pipelines.partial import RenderSpec, apply_table_partial
from pipelines.image_specs import STYLE_IMAGE_SPECS, resolve_style_name, stable_int_seed

Matrix = List[List[str]]
RENDER_ROOT = Path(__file__).resolve().parents[1]
CODEBASE_ROOT = RENDER_ROOT.parent


PARTIAL_TABLE_CSS = (
    "th, td { white-space: pre-wrap; word-break: normal; overflow-wrap: break-word; }\n"
    ".multi-panel { display: flex; flex-direction: column; gap: 24px; align-items: flex-start; width: fit-content; }\n"
    ".panel { display: block; width: fit-content; min-width: 200px; }\n"
    ".panel-label { font-size: 12px; color: #54595d; margin: 0 0 6px 0; }\n"
)

_SCREENSHOT_EXTRA_CSS = ".multi-panel { gap: 12px; }\n"


def _sanitize_token(text: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    token = token.strip("_")
    if not token:
        raise ValueError("Record id is empty after path sanitization.")
    return token


def _matrix_to_canonical(
    matrix: Matrix,
    header_rows: int,
    *,
    source: str,
    caption: str,
    source_format: str,
    has_header: bool = True,
) -> CanonicalTable:
    if has_header:
        header_rows = max(1, min(int(header_rows), len(matrix) or 1))
    else:
        header_rows = 0

    table = create_simple_table(matrix, has_header=bool(has_header))
    table.metadata.source = source
    table.metadata.caption = caption or ""
    table.metadata.source_format = source_format

    if has_header and header_rows > 1:
        for cell in table.structure.cells:
            if cell.start_row < header_rows:
                cell.is_header = True

    return table


def _build_canonical_list(
    spec: RenderSpec,
    *,
    source: str,
    caption: str,
    source_format: str,
) -> List[CanonicalTable]:
    tables: List[CanonicalTable] = []
    for t in spec.tables:
        cap = t.table_name or caption
        tables.append(
            _matrix_to_canonical(
                t.matrix,
                t.header_rows,
                source=source,
                caption=cap,
                source_format=source_format,
            )
        )
    return tables


def _parsed_to_canonical(
    parsed: ParsedTable,
    *,
    source: str,
    caption: str,
    source_format: str,
) -> CanonicalTable:
    has_header = parsed.header_rows > 0
    return _matrix_to_canonical(
        parsed.matrix,
        parsed.header_rows,
        source=source,
        caption=caption,
        source_format=source_format,
        has_header=has_header,
    )


def _render_tables(
    renderer: PlaywrightScreenshotSession,
    *,
    parsed_tables: List[ParsedTable],
    source_id: str,
    table_title: str,
    style: str,
    output_path: Path,
    viewport: Tuple[int, int],
    extra_css: str = "",
) -> None:
    spec = RenderSpec(tables=parsed_tables)
    canonical_list = _build_canonical_list(
        spec,
        source=source_id,
        caption=table_title,
        source_format="latex_table",
    )
    if extra_css:
        _prev_extra = renderer._extra_css
        renderer._extra_css = _prev_extra + "\n" + extra_css
        renderer._css_cache.clear()
    try:
        renderer.render(
            canonical_list,
            style=style,
            output_path=output_path,
            show_panel_label=False,
            viewport=viewport,
        )
    finally:
        if extra_css:
            renderer._extra_css = _prev_extra
            renderer._css_cache.clear()


def _open_png_bytes(png_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(png_bytes)).convert("RGB")


def _compose_images(
    images: List[Image.Image],
    *,
    orientation: str,
    gap: int = 24,
    background: Tuple[int, int, int] = (255, 255, 255),
    align: str = "center",
) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), background)

    if len(images) == 1:
        return images[0].copy()

    if orientation == "vertical":
        w = max(img.width for img in images)
        h = sum(img.height for img in images) + int(gap) * (len(images) - 1)
        canvas = Image.new("RGB", (w, h), background)
        y = 0
        for img in images:
            x = 0 if align == "left" else (w - img.width) // 2
            canvas.paste(img, (x, y))
            y += img.height + int(gap)
        return canvas

    h = max(img.height for img in images)
    w = sum(img.width for img in images) + int(gap) * (len(images) - 1)
    canvas = Image.new("RGB", (w, h), background)
    x = 0
    for img in images:
        y = 0 if align == "top" else (h - img.height) // 2
        canvas.paste(img, (x, y))
        x += img.width + int(gap)
    return canvas


def _render_single_table_image(
    renderer: PlaywrightScreenshotSession,
    table: CanonicalTable,
    *,
    style: str,
    viewport: Tuple[int, int],
) -> Image.Image:
    png_bytes = renderer.screenshot_bytes(
        [table],
        style=style,
        show_panel_label=False,
        viewport=viewport,
    )
    return _open_png_bytes(png_bytes)


def _render_partial_image(
    renderer: PlaywrightScreenshotSession,
    parsed: ParsedTable,
    *,
    rng: random.Random,
    source: str,
    caption: str,
    source_format: str,
    viewport: Tuple[int, int],
) -> Tuple[Image.Image, Dict[str, Any]]:
    spec0 = RenderSpec(tables=[parsed])
    spec1, table_m = apply_table_partial(spec0, rng=rng)
    parts = spec1.tables
    images: List[Image.Image] = []
    for t in parts:
        images.append(
            _render_single_table_image(
                renderer,
                _parsed_to_canonical(
                    t, source=source, caption=caption, source_format=source_format
                ),
                style="web",
                viewport=viewport,
            )
        )
    orientation = str(table_m.get("orientation") or "vertical")
    combined = _compose_images(
        images,
        orientation=("horizontal" if orientation == "horizontal" else "vertical"),
        gap=24,
        background=(255, 255, 255),
    )
    return combined, table_m


def _render_partial_image_for_tables(
    renderer: PlaywrightScreenshotSession,
    parsed_tables: List[ParsedTable],
    *,
    rng: random.Random,
    selected_table_index: Optional[int] = None,
    source: str,
    caption: str,
    source_format: str,
    viewport: Tuple[int, int],
) -> Tuple[Image.Image, Dict[str, Any]]:
    if not parsed_tables:
        raise ValueError("partial image rendering requires at least one parsed table.")

    if len(parsed_tables) == 1:
        return _render_partial_image(
            renderer,
            parsed_tables[0],
            rng=rng,
            source=source,
            caption=caption,
            source_format=source_format,
            viewport=viewport,
        )

    if selected_table_index is not None and 0 <= int(selected_table_index) < len(
        parsed_tables
    ):
        selected_idx = int(selected_table_index)
    else:
        data_indices = [
            i
            for i, t in enumerate(parsed_tables)
            if t.table_name != "context"
            and (len(t.matrix[0]) > 1 if t.matrix else False)
        ]
        pool = data_indices if data_indices else list(range(len(parsed_tables)))
        selected_idx = rng.choice(pool)
    selected = parsed_tables[selected_idx]
    selected_img, selected_meta = _render_partial_image(
        renderer,
        selected,
        rng=rng,
        source=source,
        caption=(selected.table_name or caption),
        source_format=source_format,
        viewport=viewport,
    )

    images: List[Image.Image] = []
    for idx, t in enumerate(parsed_tables):
        if idx == selected_idx:
            images.append(selected_img)
            continue
        cap = t.table_name or caption
        canonical = _parsed_to_canonical(
            t, source=source, caption=cap, source_format=source_format
        )
        images.append(
            _render_single_table_image(
                renderer, canonical, style="web", viewport=viewport
            )
        )

    combined = _compose_images(
        images,
        orientation="vertical",
        gap=24,
        background=(255, 255, 255),
        align="left",
    )
    meta = {
        "mode": "table_stack",
        "tables_count": len(parsed_tables),
        "selected_table_index": selected_idx,
        "selected_table_name": selected.table_name,
        "selected_partial": selected_meta,
    }
    return combined, meta


def generate_visual_benchmark_dataset(
    *,
    input: str,
    output_dir: str,
    limit: int,
    start: int,
    overwrite: bool,
    base_seed: int,
    viewport_width: int,
    viewport_height: int,
    num_shards: int = 1,
    shard_index: int = 0,
    playwright_browsers_path: str,
    image_names: Optional[List[str]] = None,
) -> int:
    input_path = Path(input)
    output_root = Path(output_dir)
    images_root = output_root
    logs_root = output_root / "logs"

    style_image_names = {spec.name for spec in STYLE_IMAGE_SPECS}
    table_image_names = {"noise", "structural", "partial", "missing"}
    question_image_names = {"screenshot", "photo"}
    allowed_image_names = style_image_names | table_image_names | question_image_names
    selected_image_names = (
        set(allowed_image_names) if image_names is None else set(image_names)
    )
    unknown = sorted(selected_image_names - allowed_image_names)
    if unknown:
        raise ValueError(
            f"Unknown image_names: {unknown}. Allowed: {sorted(allowed_image_names)}"
        )

    selected_style_specs = [
        spec for spec in STYLE_IMAGE_SPECS if spec.name in selected_image_names
    ]
    selected_table_image_names = selected_image_names & table_image_names
    selected_question_image_names = selected_image_names & question_image_names

    if selected_table_image_names or selected_question_image_names:
        if not any(spec.name == "web" for spec in selected_style_specs):
            raise ValueError(
                "Derived image names require 'web' to be rendered. "
                "Please include 'web' in image_names."
            )

    num_shards = int(num_shards)
    shard_index = int(shard_index)
    if num_shards <= 1:
        num_shards = 1
        shard_index = 0

    shard_tag = f"shard{shard_index:02d}of{num_shards:02d}"
    index_path = output_root / (
        "manifest.jsonl" if num_shards == 1 else f"manifest.{shard_tag}.jsonl"
    )
    stats_path = output_root / (
        "stats.json" if num_shards == 1 else f"stats.{shard_tag}.json"
    )
    log_path = logs_root / (
        "render.log.jsonl" if num_shards == 1 else f"render.{shard_tag}.log.jsonl"
    )

    ensure_parent(index_path)
    ensure_parent(stats_path)
    ensure_parent(log_path)

    counts: Dict[str, Any] = {
        "records_total": 0,
        "records_processed": 0,
        "records_failed": 0,
        "unique_tables": 0,
        "source_id": {},
        "table_format": {},
        "rows_dist": {},
        "cols_dist": {},
        "run": {
            "input": str(input_path),
            "output_dir": str(output_root),
            "base_seed": int(base_seed),
            "viewport": [int(viewport_width), int(viewport_height)],
            "sharding": {
                "num_shards": int(num_shards),
                "shard_index": int(shard_index),
            },
            "playwright_browsers_path": "",
            "playwright_browsers_path_mode": str(playwright_browsers_path),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    records = list(read_jsonl(input_path))
    counts["records_total"] = len(records)

    start = max(0, int(start))
    end = len(records) if int(limit) <= 0 else min(len(records), start + int(limit))
    records_slice = list(enumerate(records))[start:end]
    if num_shards > 1:
        records_slice = [
            (idx, rec)
            for (idx, rec) in records_slice
            if (idx % num_shards) == shard_index
        ]
    unique_tables: set[str] = set()

    with index_path.open("w", encoding="utf-8") as index_f, log_path.open(
        "w", encoding="utf-8"
    ) as log_f:
        effective_browsers_path = ""
        if str(playwright_browsers_path).strip().lower() != "auto":
            effective_browsers_path = str(Path(playwright_browsers_path).resolve())
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = effective_browsers_path
        else:
            repo_browsers = CODEBASE_ROOT / "artifacts" / "_playwright_browsers"
            if repo_browsers.exists():
                effective_browsers_path = str(repo_browsers.resolve())
                os.environ.setdefault(
                    "PLAYWRIGHT_BROWSERS_PATH", effective_browsers_path
                )

        counts["run"]["playwright_browsers_path"] = os.environ.get(
            "PLAYWRIGHT_BROWSERS_PATH", effective_browsers_path
        )
        browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
        install_cmd = (
            f'PLAYWRIGHT_BROWSERS_PATH="{browsers_path}" python -m playwright install chromium'
            if browsers_path
            else "python -m playwright install chromium"
        )

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to launch browser. Please install: {install_cmd}"
                ) from exc

            page = browser.new_page(
                viewport={"width": int(viewport_width), "height": int(viewport_height)}
            )
            renderer = PlaywrightScreenshotSession(
                page,
                template_path=RENDER_ROOT / "templates" / "base.html",
                extra_css=PARTIAL_TABLE_CSS,
            )

            run_t0 = time.time()
            total_to_process = len(records_slice)

            for record_idx, record in records_slice:
                if "id" not in record:
                    raise ValueError(
                        f"Missing required field 'id' at input row {record_idx}."
                    )
                source_id = str(record["id"])
                record_id = _sanitize_token(source_id)
                table_title = ""
                table_field = record["table"]

                question = str(record.get("question") or "").strip()

                counts["source_id"][source_id] = (
                    counts["source_id"].get(source_id, 0) + 1
                )
                unique_tables.add(source_id)

                parsed_tables = parse_latex_tables(table_field)
                parsed = parsed_tables[0]
                counts["table_format"][parsed.format] = (
                    counts["table_format"].get(parsed.format, 0) + 1
                )
                counts["rows_dist"][len(parsed.matrix)] = (
                    counts["rows_dist"].get(len(parsed.matrix), 0) + 1
                )
                cols = len(parsed.matrix[0]) if parsed.matrix else 0
                counts["cols_dist"][cols] = counts["cols_dist"].get(cols, 0) + 1

                record_dir = images_root / record_id
                record_dir.mkdir(parents=True, exist_ok=True)
                record_image_dir = record_dir

                def rel(pth: Path) -> str:
                    return str(pth.relative_to(output_root))

                try:
                    viewport_tuple = (int(viewport_width), int(viewport_height))
                    style_image_paths: Dict[str, Path] = {}
                    web_table_img: Optional[Image.Image] = None
                    for spec in selected_style_specs:
                        style_image_path = record_image_dir / f"{spec.name}.png"
                        style_name = resolve_style_name(
                            spec,
                            record_id=record_id,
                            base_seed=int(base_seed),
                        )
                        if overwrite or not style_image_path.exists():
                            _render_tables(
                                renderer,
                                parsed_tables=parsed_tables,
                                source_id=source_id,
                                table_title=table_title,
                                style=style_name,
                                output_path=style_image_path,
                                viewport=viewport_tuple,
                            )
                        style_image_paths[spec.name] = style_image_path
                        if spec.name == "web" and web_table_img is None:
                            web_table_img = Image.open(style_image_path).convert("RGB")

                    web_path = style_image_paths.get("web")
                    if (
                        selected_table_image_names or selected_question_image_names
                    ) and web_path is None:
                        raise RuntimeError(
                            "Required image name 'web' was not rendered for this record."
                        )

                    table_image_index: Dict[str, Any] = {}
                    question_image_index: Dict[str, Any] = {}

                    if selected_table_image_names:
                        assert web_table_img is not None

                        if "noise" in selected_table_image_names:
                            noise_path = record_image_dir / "noise.png"
                            noise_meta: Dict[str, Any] = {}
                            noise_seed = stable_int_seed(
                                [record_id, "noise"], base_seed=int(base_seed)
                            )
                            if overwrite or not noise_path.exists():
                                base_img = web_table_img.copy()
                                noise_rng = random.Random(noise_seed)
                                noise_np_rng = np.random.default_rng(noise_seed)
                                noisy_img, noise_meta = apply_noise(
                                    base_img, rng=noise_rng, np_rng=noise_np_rng
                                )
                                noisy_img.save(noise_path)
                            table_image_index["noise"] = {
                                "seed": noise_seed,
                                "path": rel(noise_path),
                                "meta": noise_meta,
                            }

                        if "partial" in selected_table_image_names:
                            partial_path = record_image_dir / "partial.png"
                            partial_meta: Dict[str, Any] = {}
                            partial_seed = stable_int_seed(
                                [record_id, "partial"], base_seed=int(base_seed)
                            )
                            if overwrite or not partial_path.exists():
                                partial_rng = random.Random(partial_seed)
                                table_img, table_m = _render_partial_image_for_tables(
                                    renderer,
                                    parsed_tables,
                                    rng=partial_rng,
                                    source=source_id,
                                    caption=table_title,
                                    source_format="latex_table",
                                    viewport=viewport_tuple,
                                )
                                table_img.save(partial_path)
                                partial_meta = {"table": table_m}
                            table_image_index["partial"] = {
                                "seed": partial_seed,
                                "path": rel(partial_path),
                                "meta": partial_meta,
                            }

                        if "missing" in selected_table_image_names:
                            _missing_parsed_tables = parse_latex_tables(
                                record["table_missing"]
                            )
                            missing_path = record_image_dir / "missing.png"
                            missing_meta: Dict[str, Any] = {}
                            missing_seed = stable_int_seed(
                                [record_id, "missing"], base_seed=int(base_seed)
                            )
                            if overwrite or not missing_path.exists():
                                _render_tables(
                                    renderer,
                                    parsed_tables=_missing_parsed_tables,
                                    source_id=source_id,
                                    table_title=table_title,
                                    style="web",
                                    output_path=missing_path,
                                    viewport=viewport_tuple,
                                )
                                missing_meta = {"source": "table_missing_field"}
                            table_image_index["missing"] = {
                                "seed": missing_seed,
                                "path": rel(missing_path),
                                "meta": missing_meta,
                            }

                        if "structural" in selected_table_image_names:
                            sn_path = record_image_dir / "structural.png"
                            sn_meta: Dict[str, Any] = {}
                            sn_seed = stable_int_seed(
                                [record_id, "structural"], base_seed=int(base_seed)
                            )
                            if overwrite or not sn_path.exists():
                                base_img = web_table_img.copy()
                                sn_rng = random.Random(sn_seed)
                                sn_np_rng = np.random.default_rng(sn_seed)
                                sn_img, sn_meta = apply_structural(
                                    base_img, rng=sn_rng, np_rng=sn_np_rng
                                )
                                sn_img.save(sn_path)
                            table_image_index["structural"] = {
                                "seed": sn_seed,
                                "path": rel(sn_path),
                                "meta": sn_meta,
                            }

                    if selected_question_image_names:
                        assert web_table_img is not None
                        record_image_dir = images_root / record_id
                        record_image_dir.mkdir(parents=True, exist_ok=True)

                        screenshot_path = record_image_dir / "screenshot.png"
                        needs_screenshot = bool(
                            selected_question_image_names & {"screenshot", "photo"}
                        )
                        if needs_screenshot:
                            if overwrite or not screenshot_path.exists():
                                screenshot_tables: List[ParsedTable] = []
                                if question:
                                    screenshot_tables.append(
                                        ParsedTable(
                                            matrix=[["Question"], [question]],
                                            header_rows=1,
                                            left_header_cols=0,
                                            format="latex",
                                            table_name="question",
                                        )
                                    )
                                screenshot_tables.extend(parsed_tables)
                                _render_tables(
                                    renderer,
                                    parsed_tables=screenshot_tables,
                                    source_id=source_id,
                                    table_title=table_title,
                                    style="web",
                                    output_path=screenshot_path,
                                    viewport=viewport_tuple,
                                    extra_css=_SCREENSHOT_EXTRA_CSS,
                                )
                        if "screenshot" in selected_question_image_names:
                            question_image_index["screenshot"] = {
                                "path": rel(screenshot_path),
                                "meta": {"question_added": bool(question)},
                            }

                        if "photo" in selected_question_image_names:
                            photo_path = record_image_dir / "photo.png"
                            photo_seed = stable_int_seed(
                                [record_id, "photo"], base_seed=int(base_seed)
                            )
                            photo_meta: Dict[str, Any] = {}
                            if overwrite or not photo_path.exists():
                                base_img = Image.open(screenshot_path).convert("RGB")
                                photo_rng = random.Random(photo_seed)
                                photo_np_rng = np.random.default_rng(photo_seed)
                                out_img, photo_meta = apply_photo(
                                    base_img, rng=photo_rng, np_rng=photo_np_rng
                                )
                                out_img.save(photo_path)
                            question_image_index["photo"] = {
                                "seed": photo_seed,
                                "path": rel(photo_path),
                                "meta": photo_meta,
                            }

                    index_line: Dict[str, Any] = {"id": record_id}
                    index_line.update(
                        {
                            f"{spec.name}_path": rel(style_image_paths[spec.name])
                            for spec in selected_style_specs
                        }
                    )
                    for key, field_name in {
                        "noise": "noise_path",
                        "structural": "structural_path",
                        "partial": "partial_path",
                        "missing": "missing_path",
                    }.items():
                        if key in table_image_index:
                            index_line[field_name] = table_image_index[key]["path"]
                    if "screenshot" in question_image_index:
                        index_line["screenshot_path"] = question_image_index[
                            "screenshot"
                        ]["path"]
                    if "photo" in question_image_index:
                        index_line["photo_path"] = question_image_index["photo"]["path"]
                    index_f.write(json.dumps(index_line, ensure_ascii=False) + "\n")
                    counts["records_processed"] += 1
                    log_f.write(
                        json.dumps(
                            {"id": record_id, "status": "ok"}, ensure_ascii=False
                        )
                        + "\n"
                    )

                except Exception as e:
                    counts["records_failed"] += 1
                    log_f.write(
                        json.dumps(
                            {"id": record_id, "status": "error", "error": str(e)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    raise RuntimeError(f"Failed to render record {record_id}") from e

                done = counts["records_processed"] + counts["records_failed"]
                if done % 20 == 0 or done == total_to_process:
                    elapsed = max(0.001, time.time() - run_t0)
                    rate = done / elapsed
                    eta = (total_to_process - done) / max(1e-6, rate)
                    print(
                        f"[progress] {done}/{total_to_process} "
                        f"processed={counts['records_processed']} failed={counts['records_failed']} "
                        f"elapsed={elapsed:.1f}s eta={eta/60:.1f}min"
                    )

            browser.close()

    counts["run"]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    counts["unique_tables"] = len(unique_tables)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)

    print(
        f"[render] done: processed={counts['records_processed']} failed={counts['records_failed']} output_dir={output_root}"
    )
    print(f"- manifest: {index_path}")
    print(f"- stats: {stats_path}")
    print(f"- log:   {log_path}")
    return 0
