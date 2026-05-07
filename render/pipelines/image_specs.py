from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class StyledImageSpec:
    name: str
    style: Optional[str]
    seed_key: Optional[str] = None


CUSTOM_THEME_STYLES: List[str] = [
    "ayu",
    "darcula",
    "solarized_light",
    "solarized_dark",
    "github_light",
    "monokai",
    "nord",
    "one_dark",
    "tokyo_night",
    "gruvbox_dark",
]


STYLE_IMAGE_SPECS: List[StyledImageSpec] = [
    StyledImageSpec(name="web", style="web"),
    StyledImageSpec(name="latex", style="latex"),
    StyledImageSpec(name="excel", style="excel"),
    StyledImageSpec(name="custom", style=None, seed_key="custom"),
]


def stable_int_seed(parts: Iterable[str], base_seed: int) -> int:
    raw = f"{base_seed}||" + "||".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def resolve_style_name(spec: StyledImageSpec, *, record_id: str, base_seed: int) -> str:
    if spec.style:
        return spec.style
    if not spec.seed_key:
        raise ValueError(f"Image name {spec.name} missing style resolver")
    theme_seed = stable_int_seed([record_id, spec.seed_key], base_seed=int(base_seed))
    return random.Random(theme_seed).choice(CUSTOM_THEME_STYLES)
