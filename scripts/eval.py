import argparse
import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from openai import APIStatusError, AsyncOpenAI

CODEBASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CODEBASE_ROOT / "configs" / "eval.yaml"

SYSTEM_PROMPT = (
    "You are a table understanding expert. "
    "Answer the question based on the given table image. "
    "Provide only the final answer, no explanation."
)
SYSTEM_PROMPT_VISION = (
    "You are a table understanding expert. The image contains both a question and a table. "
    "Read the question from the image and answer it based on the table. "
    "Provide only the final answer, no explanation."
)

IMAGE_NAMES = {
    "web",
    "latex",
    "excel",
    "custom",
    "noise",
    "structural",
    "partial",
    "missing",
    "screenshot",
    "photo",
}
VISION_IMAGE_NAMES = {"screenshot", "photo"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(value: str | Path, *, base: Path = CODEBASE_ROOT) -> Path:
    text = os.path.expandvars(str(value))
    path = Path(text).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")


def normalize_answer(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    text = re.sub(r"^.*</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*◁/think▷", "", text, flags=re.DOTALL)
    return text.strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records(
    data_path: Path, image_names: list[str], limit: int | None
) -> list[dict[str, Any]]:
    data = load_jsonl(data_path)
    if limit:
        data = data[:limit]

    records: list[dict[str, Any]] = []
    for item in data:
        meta = item["visual"]
        for image_name in image_names:
            if image_name not in IMAGE_NAMES:
                raise ValueError(
                    f"Unknown image name: {image_name}. Allowed: {sorted(IMAGE_NAMES)}"
                )
            rel_path = meta[f"{image_name}_path"]
            records.append(
                {
                    "id": item["id"],
                    "image_name": image_name,
                    "image_path": rel_path,
                    "question": item["question"],
                    "answer": item["answer"],
                }
            )
    return records


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    with path.open("rb") as f:
        payload = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{payload}"


def user_prompt(record: dict[str, Any]) -> str:
    if record["image_name"] in VISION_IMAGE_NAMES:
        return (
            "The question is shown in the image. "
            "Answer it based on the table in the image.\n\n"
            "Answer directly with the answer only."
        )
    return (
        "Answer the question based on the table shown in the image.\n\n"
        f"Question: {record['question']}\n\n"
        "Answer directly with the answer only."
    )


async def call_api(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    record: dict[str, Any],
    visual_root: Path,
    max_tokens: int,
    model_cfg: dict[str, Any],
) -> str:
    image_url = image_to_data_url(visual_root / record["image_path"])
    system = (
        SYSTEM_PROMPT_VISION
        if record["image_name"] in VISION_IMAGE_NAMES
        else SYSTEM_PROMPT
    )
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_prompt(record)},
                ],
            },
        ],
        "max_completion_tokens": max_tokens,
        "temperature": 0,
    }
    if model_cfg.get("reasoning_effort"):
        kwargs["reasoning_effort"] = model_cfg["reasoning_effort"]
    if model_cfg.get("extra_body"):
        kwargs["extra_body"] = model_cfg["extra_body"]
    async with sem:
        try:
            response = await client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            body = str(exc.response.text)[:500] if exc.response is not None else ""
            raise RuntimeError(f"API error {exc.status_code}: {body}") from exc
    return response.choices[0].message.content or ""


async def run_api(
    records: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    api_cfg: dict[str, Any],
    visual_root: Path,
) -> list[dict[str, Any]]:
    from tqdm import tqdm

    api_key_env = api_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {api_key_env}")

    sem = asyncio.Semaphore(int(api_cfg.get("max_concurrent", 8)))
    model = model_cfg.get("model") or model_cfg.get("model_id")
    max_tokens = int(model_cfg.get("max_tokens") or api_cfg.get("max_tokens", 4096))
    timeout_s = float(api_cfg.get("timeout", 180))

    client = AsyncOpenAI(
        base_url=api_cfg["base_url"],
        api_key=api_key,
        max_retries=0,
        timeout=timeout_s,
    )
    try:
        async def _indexed_call(idx: int, rec: dict[str, Any]) -> tuple[int, str]:
            return idx, await call_api(
                client, sem, model, rec, visual_root, max_tokens, model_cfg
            )

        tasks = [_indexed_call(i, rec) for i, rec in enumerate(records)]
        ordered: list[str | None] = [None] * len(records)
        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="api inference"
        ):
            idx, output = await coro
            ordered[idx] = output
    finally:
        await client.close()
    return attach_predictions(records, [x or "" for x in ordered])


def apply_chat_template(
    processor: Any, messages: list[dict[str, Any]], model_id: str
) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if model_id.startswith("Qwen/"):
        kwargs["enable_thinking"] = False
    return processor.apply_chat_template(messages, **kwargs)


def run_vllm(
    records: list[dict[str, Any]],
    model_id: str,
    model_cfg: dict[str, Any],
    vllm_cfg: dict[str, Any],
    visual_root: Path,
) -> list[dict[str, Any]]:
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    model_path = resolve_path(model_cfg["model_path"])
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=int(model_cfg.get("tensor_parallel_size", 1)),
        trust_remote_code=trust_remote_code,
        max_model_len=int(
            model_cfg.get("max_model_len", vllm_cfg.get("max_model_len", 12288))
        ),
        gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.9)),
        limit_mm_per_prompt={"image": 1},
    )
    sampling = dict(model_cfg.get("sampling", {}))
    sampling.setdefault("max_tokens", int(vllm_cfg.get("max_tokens", 8192)))
    if model_cfg.get("stop"):
        sampling["stop"] = model_cfg["stop"]
    params = SamplingParams(**sampling)

    prompts = []
    for rec in records:
        image = Image.open(visual_root / rec["image_path"]).convert("RGB")
        system = (
            SYSTEM_PROMPT_VISION
            if rec["image_name"] in VISION_IMAGE_NAMES
            else SYSTEM_PROMPT
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt(rec)},
                ],
            },
        ]
        prompts.append(
            {
                "prompt": apply_chat_template(processor, messages, model_id),
                "multi_modal_data": {"image": image},
            }
        )

    outputs = []
    batch_size = int(vllm_cfg.get("batch_size", 16))
    for start in tqdm(range(0, len(prompts), batch_size), desc="vLLM inference"):
        batch = prompts[start : start + batch_size]
        for out in llm.generate(batch, params):
            outputs.append(out.outputs[0].text)
    return attach_predictions(records, outputs)


def attach_predictions(
    records: list[dict[str, Any]], outputs: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for rec, raw in zip(records, outputs):
        pred = extract_answer(raw)
        exact = normalize_answer(pred) == normalize_answer(rec["answer"])
        rows.append(
            {**rec, "prediction": pred, "raw_prediction": raw, "exact_match": exact}
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    em = sum(1 for r in rows if r["exact_match"])
    combined = sum(1 for r in rows if r.get("combined_correct", r["exact_match"]))
    by_name: dict[str, dict[str, int]] = {}
    for r in rows:
        slot = by_name.setdefault(
            r["image_name"], {"total": 0, "exact_match": 0, "combined": 0}
        )
        slot["total"] += 1
        slot["exact_match"] += int(r["exact_match"])
        slot["combined"] += int(r.get("combined_correct", r["exact_match"]))
    return {
        "total": total,
        "exact_match": em / total if total else 0.0,
        "combined": combined / total if total else 0.0,
        "by_image_name": {
            k: {
                "total": v["total"],
                "exact_match": v["exact_match"] / v["total"] if v["total"] else 0.0,
                "combined": v["combined"] / v["total"] if v["total"] else 0.0,
            }
            for k, v in sorted(by_name.items())
        },
    }


async def call_judge(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    judge_cfg: dict[str, Any],
    row: dict[str, Any],
) -> bool:
    prompt = (
        "Judge whether the predicted answer is semantically equivalent to the gold answer.\n"
        "Return only CORRECT or INCORRECT.\n\n"
        f"Question: {row['question']}\n"
        f"Gold answer: {row['answer']}\n"
        f"Predicted answer: {row['prediction']}\n"
    )
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=judge_cfg["model"],
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict answer equivalence judge.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=int(judge_cfg.get("max_tokens", 256)),
            )
        except APIStatusError as exc:
            body = str(exc.response.text)[:500] if exc.response is not None else ""
            raise RuntimeError(f"Judge API error {exc.status_code}: {body}") from exc
    verdict = (response.choices[0].message.content or "").strip().upper()
    return verdict.startswith("CORRECT")


async def run_judge(
    rows: list[dict[str, Any]], api_cfg: dict[str, Any], judge_cfg: dict[str, Any]
) -> None:
    from tqdm import tqdm

    api_key_env = api_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {api_key_env}")

    targets = [(i, row) for i, row in enumerate(rows) if not row["exact_match"]]
    sem = asyncio.Semaphore(int(judge_cfg.get("max_concurrent", 8)))
    timeout_s = float(api_cfg.get("timeout", 180))

    client = AsyncOpenAI(
        base_url=api_cfg["base_url"],
        api_key=api_key,
        max_retries=0,
        timeout=timeout_s,
    )
    try:
        async def _indexed_call(idx: int, row: dict[str, Any]) -> tuple[int, bool]:
            return idx, await call_judge(client, sem, judge_cfg, row)

        tasks = [_indexed_call(i, row) for i, row in targets]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="judge"):
            idx, correct = await coro
            rows[idx]["judge_correct"] = correct
            rows[idx]["combined_correct"] = correct
    finally:
        await client.close()
    for row in rows:
        if row["exact_match"]:
            row["judge_correct"] = None
            row["combined_correct"] = True
        elif "combined_correct" not in row:
            row["combined_correct"] = False


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(summarize(rows), f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a model on TableVista.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", required=True, help="Model key in configs/eval.yaml")
    parser.add_argument("--image-names", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = cfg["models"][args.model]
    data_path = resolve_path(cfg["data"]["data_path"])
    visual_set = resolve_path(cfg["data"]["visual_set"])
    output_base = resolve_path(args.output or cfg["data"].get("output_dir", "outputs"))
    output_dir = output_base / model_slug(args.model)
    image_names = args.image_names or cfg["image_names"]
    records = load_records(data_path, image_names, args.limit)

    backend = model_cfg["backend"]
    if backend == "api":
        rows = asyncio.run(run_api(records, model_cfg, cfg["api"], visual_set))
    elif backend == "vllm":
        rows = run_vllm(records, args.model, model_cfg, cfg.get("vllm", {}), visual_set)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    if cfg.get("judge", {}).get("enabled"):
        asyncio.run(run_judge(rows, cfg["api"], cfg["judge"]))
    write_outputs(rows, output_dir)
    print(json.dumps(summarize(rows), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
