import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

CODEBASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CODEBASE_ROOT / "configs" / "render.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(value: str | Path, *, base: Path = CODEBASE_ROOT) -> Path:
    text = os.path.expandvars(str(value))
    path = Path(text).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def merged_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    cfg.setdefault("data", {})
    cfg.setdefault("render", {})
    cfg.setdefault("parallel", {})

    if args.input:
        cfg["data"]["input"] = args.input
    if args.output:
        cfg["data"]["output"] = args.output
    if args.limit is not None:
        cfg["render"]["limit"] = args.limit
    if args.overwrite:
        cfg["render"]["overwrite"] = True
    if args.num_shards is not None:
        cfg["parallel"]["num_shards"] = args.num_shards
    if args.shard_index is not None:
        cfg["parallel"]["shard_index"] = args.shard_index
    return cfg


def run_one(cfg: dict[str, Any], *, shard_index: int | None = None) -> int:
    render_dir = CODEBASE_ROOT / "render"
    sys.path.insert(0, str(render_dir))

    from pipelines.render_dataset import generate_visual_benchmark_dataset

    data_cfg = cfg["data"]
    render_cfg = cfg["render"]
    parallel_cfg = cfg.get("parallel", {})

    input_path = resolve_path(data_cfg["input"])
    output_dir = resolve_path(data_cfg["output"])
    num_shards = int(parallel_cfg.get("num_shards") or 1)
    actual_shard_index = shard_index
    if actual_shard_index is None:
        actual_shard_index = parallel_cfg.get("shard_index")
    actual_shard_index = 0 if actual_shard_index is None else int(actual_shard_index)

    return generate_visual_benchmark_dataset(
        input=str(input_path),
        output_dir=str(output_dir),
        limit=int(render_cfg.get("limit", 0)),
        start=int(render_cfg.get("start", 0)),
        overwrite=bool(render_cfg.get("overwrite", False)),
        base_seed=int(render_cfg.get("base_seed", 42)),
        viewport_width=int(render_cfg.get("viewport_width", 1000)),
        viewport_height=int(render_cfg.get("viewport_height", 800)),
        num_shards=num_shards,
        shard_index=actual_shard_index,
        playwright_browsers_path=str(
            render_cfg.get("playwright_browsers_path", "auto")
        ),
        image_names=render_cfg.get("image_names"),
    )


def run_parallel(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    num_shards = int(cfg.get("parallel", {}).get("num_shards") or 1)
    shard_index = cfg.get("parallel", {}).get("shard_index")
    if num_shards <= 1 or shard_index is not None:
        return run_one(cfg)

    procs: list[subprocess.Popen] = []
    for i in range(num_shards):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config),
            "--num-shards",
            str(num_shards),
            "--shard-index",
            str(i),
        ]
        if args.input:
            cmd += ["--input", args.input]
        if args.output:
            cmd += ["--output", args.output]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.overwrite:
            cmd += ["--overwrite"]
        print(f"[render] starting shard {i}/{num_shards}: {' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd))

    failed = 0
    for i, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            failed += 1
            print(f"[render] shard {i} failed with exit code {rc}", flush=True)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render TableVista visual benchmark images."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    args = parser.parse_args()
    cfg = merged_config(args)
    return run_parallel(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
