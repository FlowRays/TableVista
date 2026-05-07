from pathlib import Path
from typing import Optional
import html as html_lib
import random

from .table_schema import CanonicalTable


class Renderer:
    def __init__(self, style: str = "web", seed: Optional[int] = None):
        self.style = style
        self._rng = random.Random(seed) if seed is not None else random
        self.render_root = Path(__file__).resolve().parents[1]
        self.css_path = self.render_root / "styles" / f"{style}.css"

        if not self.css_path.exists():
            raise ValueError(f"Style not found: {style}")

    def _generate_random_cell_style(self) -> str:
        bg_colors = [
            "#FFE5E5",
            "#E5F5FF",
            "#E5FFE5",
            "#FFF5E5",
            "#F5E5FF",
            "#FFE5F5",
            "#E5FFF5",
            "#F5FFE5",
            "#FFE5D5",
            "#E5E5FF",
            "#FFCCCC",
            "#CCFFCC",
            "#CCCCFF",
            "#FFCCFF",
            "#CCFFFF",
            "#FFFFCC",
            "#FFE0CC",
            "#E0CCFF",
            "#CCE0FF",
            "#FFCCE0",
            "#F0F0F0",
            "#E8E8E8",
            "#D0D0D0",
            "#C8C8C8",
            "#FFFFFF",
            "#FFF0F0",
            "#F0FFF0",
            "#F0F0FF",
            "#FFFFF0",
            "#FFF0FF",
        ]

        text_colors = [
            "#000000",
            "#1A1A1A",
            "#333333",
            "#404040",
            "#4D4D4D",
            "#003366",
            "#006633",
            "#660033",
            "#663300",
            "#330066",
            "#004080",
            "#008040",
            "#800040",
            "#804000",
            "#400080",
            "#1F4788",
            "#2E7D32",
            "#6A1B9A",
            "#C62828",
            "#D84315",
        ]

        fonts = [
            "Arial, sans-serif",
            '"Helvetica Neue", Helvetica, sans-serif',
            '"Times New Roman", Times, serif',
            "Georgia, serif",
            '"Courier New", monospace',
            "Verdana, sans-serif",
            "Tahoma, sans-serif",
            '"Trebuchet MS", sans-serif',
            '"Comic Sans MS", cursive',
            "Impact, fantasy",
        ]

        font_weights = ["300", "400", "500", "600", "700", "bold", "normal"]
        font_sizes = ["10px", "11px", "12px", "13px", "14px", "15px", "16px"]

        text_aligns = ["left", "center", "right"]

        paddings = ["4px", "6px", "8px", "10px", "12px", "14px"]

        border_styles = ["solid", "dotted", "dashed", "double"]
        border_widths = ["1px", "2px", "3px"]
        border_colors = text_colors[:15]

        styles = []

        if self._rng.random() < self._rng.random():
            styles.append(f"background-color: {self._rng.choice(bg_colors)}")

        if self._rng.random() < self._rng.random():
            styles.append(f"color: {self._rng.choice(text_colors)}")

        if self._rng.random() < self._rng.random():
            styles.append(f"font-family: {self._rng.choice(fonts)}")

        if self._rng.random() < self._rng.random():
            styles.append(f"font-size: {self._rng.choice(font_sizes)}")

        if self._rng.random() < self._rng.random():
            styles.append(f"font-weight: {self._rng.choice(font_weights)}")

        if self._rng.random() < self._rng.random():
            styles.append(f"text-align: {self._rng.choice(text_aligns)}")

        if self._rng.random() < self._rng.random():
            styles.append(f"padding: {self._rng.choice(paddings)}")

        if self._rng.random() < self._rng.random():
            if self._rng.random() < 0.5:
                border_width = self._rng.choice(border_widths)
                border_style = self._rng.choice(border_styles)
                border_color = self._rng.choice(border_colors)
                styles.append(f"border: {border_width} {border_style} {border_color}")
            else:
                sides = ["top", "right", "bottom", "left"]
                for side in sides:
                    if self._rng.random() < self._rng.random():
                        border_width = self._rng.choice(border_widths)
                        border_style = self._rng.choice(border_styles)
                        border_color = self._rng.choice(border_colors)
                        styles.append(
                            f"border-{side}: {border_width} {border_style} {border_color}"
                        )

        if self._rng.random() < self._rng.random():
            styles.append("font-style: italic")

        if self._rng.random() < self._rng.random():
            styles.append("text-decoration: underline")

        if self._rng.random() < self._rng.random():
            shadow_x = self._rng.choice(["1px", "2px", "-1px"])
            shadow_y = self._rng.choice(["1px", "2px", "-1px"])
            shadow_blur = self._rng.choice(["0px", "1px", "2px"])
            shadow_color = self._rng.choice(
                ["rgba(0,0,0,0.3)", "rgba(0,0,0,0.2)", "rgba(0,0,0,0.1)"]
            )
            styles.append(
                f"text-shadow: {shadow_x} {shadow_y} {shadow_blur} {shadow_color}"
            )

        return "; ".join(styles)

    def _get_cell_style(self) -> Optional[str]:
        if self.style != "custom":
            return None
        return self._generate_random_cell_style()

    def _render_cell(
        self,
        *,
        cell,
        row_idx: int,
        col_idx: int,
        max_header_row: int,
    ) -> str:
        tag = "th" if row_idx <= max_header_row else "td"
        attrs = [
            f'data-row="{row_idx}"',
            f'data-col="{col_idx}"',
        ]
        if cell.rowspan > 1:
            attrs.append(f'rowspan="{cell.rowspan}"')
        if cell.colspan > 1:
            attrs.append(f'colspan="{cell.colspan}"')
        cell_style = self._get_cell_style()
        if cell_style:
            attrs.append(f'style="{cell_style}"')
        attrs_str = " " + " ".join(attrs) if attrs else ""
        safe_content = html_lib.escape(str(cell.content), quote=False)
        return f"<{tag}{attrs_str}>{safe_content}</{tag}>"

    def json_to_html_table(self, table: CanonicalTable) -> str:
        structure = table.structure
        cells = structure.cells

        grid = [[None] * structure.num_cols for _ in range(structure.num_rows)]

        for cell in cells:
            row = cell.start_row
            col = cell.start_col

            if grid[row][col] is None:
                grid[row][col] = cell

        html_parts = ["<table>"]

        header_cells = [cell for cell in cells if cell.is_header]
        if header_cells:
            max_header_row = max(cell.end_row for cell in header_cells)
            has_header = True
        else:
            max_header_row = -1
            has_header = False

        if has_header:
            html_parts.append("<thead>")

        for row_idx, row in enumerate(grid):
            html_parts.append("<tr>")

            for col_idx, cell in enumerate(row):
                if not cell:
                    continue
                if cell.start_row != row_idx or cell.start_col != col_idx:
                    continue
                html_parts.append(
                    self._render_cell(
                        cell=cell,
                        row_idx=row_idx,
                        col_idx=col_idx,
                        max_header_row=max_header_row,
                    )
                )

            html_parts.append("</tr>")

            if has_header and row_idx == max_header_row:
                html_parts.append("</thead>")
                html_parts.append("<tbody>")

        if has_header:
            html_parts.append("</tbody>")

        html_parts.append("</table>")

        return "\n".join(html_parts)
