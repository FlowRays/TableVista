from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

from playwright.sync_api import Page

from .table_schema import CanonicalTable
from .renderer import Renderer


class PlaywrightScreenshotSession:
    def __init__(
        self, page: Page, *, template_path: Path, extra_css: str = "", wait_ms: int = 80
    ) -> None:
        self._page = page
        self._template = template_path.read_text(encoding="utf-8")
        self._styles_dir = Path(__file__).resolve().parents[1] / "styles"
        self._extra_css = extra_css
        self._css_cache: Dict[str, str] = {}
        self._wait_ms = int(wait_ms)

    def _get_css(self, style: str) -> str:
        if style not in self._css_cache:
            css_path = self._styles_dir / f"{style}.css"
            self._css_cache[style] = (
                css_path.read_text(encoding="utf-8") + "\n" + self._extra_css
            )
        return self._css_cache[style]

    def _prepare_html_content(
        self,
        tables: List[CanonicalTable],
        *,
        style: str,
        show_panel_label: bool,
    ) -> Tuple[str, str]:
        renderer = Renderer(style=style)
        tables_html = [renderer.json_to_html_table(t) for t in tables]

        if len(tables_html) == 1:
            return tables_html[0], "table"

        panels: List[str] = []
        for idx, html_table in enumerate(tables_html, start=1):
            label_html = (
                f'<p class="panel-label">Panel {idx}</p>' if show_panel_label else ""
            )
            panels.append(f'<div class="panel">{label_html}{html_table}</div>')
        return (
            f'<div id="screenshot" class="multi-panel">{"".join(panels)}</div>',
            "#screenshot",
        )

    def render(
        self,
        tables: List[CanonicalTable],
        *,
        style: str,
        output_path: Path,
        show_panel_label: bool,
        viewport: Tuple[int, int],
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        body_html, selector = self._prepare_html_content(
            tables,
            style=style,
            show_panel_label=show_panel_label,
        )

        css = self._get_css(style)
        html_content = self._template.replace("{{CSS}}", css).replace(
            "{{TABLE}}", body_html
        )

        vw, vh = viewport
        self._page.set_viewport_size({"width": int(vw), "height": int(vh)})
        self._page.set_content(html_content)
        self._page.wait_for_timeout(self._wait_ms)
        bbox = self._page.locator(selector).bounding_box()
        if bbox:
            need_w = int(math.ceil(float(bbox.get("width") or 0.0))) + 40
            need_h = int(math.ceil(float(bbox.get("height") or 0.0))) + 40
            if need_w > int(vw) or need_h > int(vh):
                self._page.set_viewport_size(
                    {"width": max(int(vw), need_w), "height": max(int(vh), need_h)}
                )
                self._page.set_content(html_content)
                self._page.wait_for_timeout(self._wait_ms)
        self._page.locator(selector).screenshot(path=str(output_path))

    def screenshot_bytes(
        self,
        tables: List[CanonicalTable],
        *,
        style: str,
        show_panel_label: bool,
        viewport: Tuple[int, int],
    ) -> bytes:
        body_html, selector = self._prepare_html_content(
            tables,
            style=style,
            show_panel_label=show_panel_label,
        )

        css = self._get_css(style)
        html_content = self._template.replace("{{CSS}}", css).replace(
            "{{TABLE}}", body_html
        )

        vw, vh = viewport
        self._page.set_viewport_size({"width": int(vw), "height": int(vh)})
        self._page.set_content(html_content)
        self._page.wait_for_timeout(self._wait_ms)
        bbox = self._page.locator(selector).bounding_box()
        if bbox:
            need_w = int(math.ceil(float(bbox.get("width") or 0.0))) + 40
            need_h = int(math.ceil(float(bbox.get("height") or 0.0))) + 40
            if need_w > int(vw) or need_h > int(vh):
                self._page.set_viewport_size(
                    {"width": max(int(vw), need_w), "height": max(int(vh), need_h)}
                )
                self._page.set_content(html_content)
                self._page.wait_for_timeout(self._wait_ms)
        return self._page.locator(selector).screenshot()
