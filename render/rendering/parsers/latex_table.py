from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

Matrix = List[List[str]]
FormatType = Literal["latex"]


@dataclass(frozen=True)
class ParsedTable:
    matrix: Matrix
    header_rows: int
    left_header_cols: int
    format: FormatType
    table_name: str = ""


def normalize_rectangular(matrix: Matrix) -> Matrix:
    if not matrix:
        return matrix
    max_cols = max(len(r) for r in matrix)
    return [list(r) + [""] * (max_cols - len(r)) for r in matrix]


def _brace_extract(s: str, start: int) -> Tuple[str, int]:
    if start >= len(s) or s[start] != "{":
        return "", start
    depth = 0
    i = start
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i], i + 1
        i += 1
    return s[start + 1 :], len(s)


def _process_math_content(content: str) -> str:
    s = content.strip()

    s = re.sub(r"\^\{?\\circ\}?", "°", s)
    s = re.sub(r"\^\{?\\dagger\}?", "†", s)
    s = re.sub(r"\^\{?\\ddagger\}?", "‡", s)
    sup_map = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
    }
    sub_map = {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
    }
    s = re.sub(
        r"\^\{(\d+)\}", lambda m: "".join(sup_map.get(d, d) for d in m.group(1)), s
    )
    s = re.sub(r"\^(\d)", lambda m: sup_map.get(m.group(1), m.group(1)), s)
    s = re.sub(
        r"_\{(\d+)\}", lambda m: "".join(sub_map.get(d, d) for d in m.group(1)), s
    )
    s = re.sub(r"_(\d)", lambda m: sub_map.get(m.group(1), m.group(1)), s)

    _MATH_SYMS = {
        r"\times": "×",
        r"\pm": "±",
        r"\mp": "∓",
        r"\geq": "≥",
        r"\leq": "≤",
        r"\ge": "≥",
        r"\le": "≤",
        r"\neq": "≠",
        r"\ne": "≠",
        r"\approx": "≈",
        r"\sim": "~",
        r"\circ": "°",
        r"\bullet": "•",
        r"\cdot": "·",
        r"\dagger": "†",
        r"\ddagger": "‡",
        r"\heartsuit": "♥",
        r"\infty": "∞",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\mu": "μ",
        r"\sigma": "σ",
        r"\pi": "π",
        r"\ldots": "...",
        r"\cdots": "...",
        r"\%": "%",
    }
    for cmd, sym in _MATH_SYMS.items():
        s = s.replace(cmd, sym)

    s = s.replace("''", "″").replace("'", "′")

    s = re.sub(r"\\[A-Za-z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


_ACCENT_MAP: dict = {
    ("'", "a"): "á",
    ("'", "e"): "é",
    ("'", "i"): "í",
    ("'", "o"): "ó",
    ("'", "u"): "ú",
    ("'", "A"): "Á",
    ("'", "E"): "É",
    ("'", "I"): "Í",
    ("'", "O"): "Ó",
    ("'", "U"): "Ú",
    ("'", "c"): "ć",
    ("'", "C"): "Ć",
    ("'", "n"): "ń",
    ("'", "s"): "ś",
    ("'", "z"): "ź",
    ("'", "y"): "ý",
    ("'", "Y"): "Ý",
    ("`", "a"): "à",
    ("`", "e"): "è",
    ("`", "i"): "ì",
    ("`", "o"): "ò",
    ("`", "u"): "ù",
    ("`", "A"): "À",
    ("`", "E"): "È",
    ("`", "I"): "Ì",
    ("`", "O"): "Ò",
    ("`", "U"): "Ù",
    ('"', "a"): "ä",
    ('"', "e"): "ë",
    ('"', "i"): "ï",
    ('"', "o"): "ö",
    ('"', "u"): "ü",
    ('"', "A"): "Ä",
    ('"', "E"): "Ë",
    ('"', "I"): "Ï",
    ('"', "O"): "Ö",
    ('"', "U"): "Ü",
    ("^", "a"): "â",
    ("^", "e"): "ê",
    ("^", "i"): "î",
    ("^", "o"): "ô",
    ("^", "u"): "û",
    ("^", "A"): "Â",
    ("^", "E"): "Ê",
    ("^", "I"): "Î",
    ("^", "O"): "Ô",
    ("^", "U"): "Û",
    ("~", "a"): "ã",
    ("~", "n"): "ñ",
    ("~", "o"): "õ",
    ("~", "A"): "Ã",
    ("~", "N"): "Ñ",
    ("~", "O"): "Õ",
    ("=", "a"): "ā",
    ("=", "e"): "ē",
    ("=", "i"): "ī",
    ("=", "o"): "ō",
    ("=", "u"): "ū",
    ("=", "A"): "Ā",
    ("=", "E"): "Ē",
    ("=", "I"): "Ī",
    ("=", "O"): "Ō",
    ("=", "U"): "Ū",
    (".", "e"): "ė",
    (".", "E"): "Ė",
    (".", "z"): "ż",
    (".", "Z"): "Ż",
}


def _strip_latex_cell(text: str) -> str:
    s = re.sub(r"(?<!\\)%[^\n]*", "", text)

    result: List[str] = []
    i = 0
    while i < len(s):
        c = s[i]

        if c == "$":
            j = i + 1
            while j < len(s) and s[j] != "$":
                if s[j] == "\\":
                    j += 2
                else:
                    j += 1
            if j < len(s):
                result.append(_process_math_content(s[i + 1 : j]))
                i = j + 1
            else:
                result.append("$")
                i += 1
            continue

        if c != "\\":
            result.append(c)
            i += 1
            continue

        j = i + 1
        if j >= len(s):
            i += 1
            continue

        if s[j] == "\\":
            result.append(" ")
            i += 2
            continue

        if not s[j].isalpha():
            ch = s[j]
            if ch in "{}":
                result.append(ch)
                i += 2
            elif ch in ("%", "&", "_", "#", "-", "/", "$"):
                result.append(ch)
                i += 2
            elif ch in ("'", "`", '"', "~", "^", "=", "."):
                i += 2
                while i < len(s) and s[i] == " ":
                    i += 1
                base = ""
                if i < len(s) and s[i] == "{":
                    inner, i = _brace_extract(s, i)
                    base = inner.strip()[:1]
                elif i < len(s) and s[i].isalpha():
                    base = s[i]
                    i += 1
                result.append(_ACCENT_MAP.get((ch, base), base) if base else ch)
            else:
                i += 2
            continue

        k = j
        while k < len(s) and s[k].isalpha():
            k += 1
        cmd = s[j:k]
        i = k
        while i < len(s) and s[i] == " ":
            i += 1

        def _next_brace() -> Tuple[str, int]:
            nonlocal i
            if i < len(s) and s[i] == "{":
                content, i = _brace_extract(s, i)
                return content, i
            return "", i

        def _skip_bracket() -> None:
            nonlocal i
            if i < len(s) and s[i] == "[":
                rb = s.find("]", i)
                i = rb + 1 if rb >= 0 else len(s)

        if cmd in ("rowcolor", "cellcolor", "color"):
            _skip_bracket()
            _next_brace()

        elif cmd == "diagbox":
            _skip_bracket()
            a, _ = _next_brace()
            b, _ = _next_brace()
            result.append(_strip_latex_cell(a) + " / " + _strip_latex_cell(b))

        elif cmd == "multirow":
            if i < len(s) and s[i] == "*":
                i += 1
            _next_brace()
            _skip_bracket()
            _next_brace()
            content, _ = _next_brace()
            result.append(_strip_latex_cell(content))

        elif cmd == "multicolumn":
            _next_brace()
            _next_brace()
            content, _ = _next_brace()
            result.append(_strip_latex_cell(content))

        elif cmd == "textcolor":
            _skip_bracket()
            _next_brace()
            content, _ = _next_brace()
            result.append(_strip_latex_cell(content))

        elif cmd == "makecell":
            _skip_bracket()
            content, _ = _next_brace()
            content = content.replace("\\\\", " ").replace("\n", " ")
            result.append(_strip_latex_cell(content))

        elif cmd == "parbox":
            _next_brace()
            content, _ = _next_brace()
            result.append(_strip_latex_cell(content))

        elif cmd == "rotatebox":
            _next_brace()
            content, _ = _next_brace()
            result.append(_strip_latex_cell(content))

        elif cmd in ("hspace", "vspace", "phantom", "hphantom", "vphantom"):
            if i < len(s) and s[i] == "*":
                i += 1
            _next_brace()

        elif cmd in (
            "textbf",
            "textit",
            "text",
            "emph",
            "texttt",
            "textrm",
            "textsc",
            "textup",
            "textsl",
            "textsf",
            "textmd",
            "textnormal",
            "underline",
            "overline",
            "bm",
            "boldsymbol",
            "thead",
            "mathrm",
            "mathbf",
            "mathit",
            "mathsf",
            "mathtt",
            "mathcal",
            "mbox",
            "hbox",
        ):
            _skip_bracket()
            content, _ = _next_brace()
            result.append(_strip_latex_cell(content))

        elif cmd in (
            "raggedright",
            "raggedleft",
            "centering",
            "bfseries",
            "itshape",
            "mdseries",
            "rmfamily",
            "sffamily",
            "ttfamily",
            "upshape",
            "scshape",
            "slshape",
            "normalfont",
            "footnotesize",
            "scriptsize",
            "tiny",
            "small",
            "normalsize",
            "large",
            "Large",
            "LARGE",
            "huge",
            "Huge",
            "arraybackslash",
            "noindent",
            "hfill",
            "vfill",
            "quad",
            "qquad",
            "tabularnewline",
            "relax",
            "leavevmode",
            "strut",
            "hline",
            "toprule",
            "midrule",
            "bottomrule",
            "addlinespace",
        ):
            pass

        elif cmd == "begin":
            env_name, _ = _next_brace()
            end_marker = "\\end{" + env_name + "}"
            end_idx = s.find(end_marker, i)
            if end_idx >= 0:
                inner = s[i:end_idx]
                i = end_idx + len(end_marker)
                inner = inner.strip()
                while inner.startswith("["):
                    rb = inner.find("]")
                    inner = inner[rb + 1 :].lstrip() if rb >= 0 else inner[1:]
                while inner.startswith("{"):
                    _, p = _brace_extract(inner, 0)
                    inner = inner[p:]
                    if not inner.startswith("{"):
                        break
                inner = inner.replace("\\\\", " ")
                result.append(_strip_latex_cell(inner))

        elif cmd == "end":
            _next_brace()

        else:
            if i < len(s) and s[i] == "{":
                content, _ = _next_brace()
                result.append(_strip_latex_cell(content))

    s = "".join(result)
    while True:
        ss = s.strip()
        if len(ss) >= 2 and ss[0] == "{" and ss[-1] == "}":
            inner, pos = _brace_extract(ss, 0)
            if pos == len(ss):
                s = inner
                continue
        break
    return re.sub(r"\s+", " ", s).strip()


_ROW_SEP = "\\\\"


def _split_top_level(s: str, sep: str) -> List[str]:
    parts: List[str] = []
    cur: List[str] = []
    depth = 0
    env_depth = 0
    i = 0
    n = len(sep)
    while i < len(s):
        if depth == 0 and env_depth == 0 and s[i : i + n] == sep:
            parts.append("".join(cur))
            cur = []
            i += n
            continue
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] != "\\":
            rest = s[i:]
            if rest.startswith("\\begin{"):
                env_depth += 1
            elif rest.startswith("\\end{"):
                env_depth = max(0, env_depth - 1)
            cur.append(c)
            i += 1
            cur.append(s[i])
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        cur.append(c)
        i += 1
    parts.append("".join(cur))
    return parts


_SEP_ROW_RE = re.compile(
    r"^\\(?:hline|toprule|midrule|bottomrule|addlinespace|"
    r"endhead|endfirsthead|endfoot|endlastfoot|cmidrule|noalign)"
    r"|^\{?\\(?:renewcommand|setlength|setcounter)\b"
)


def _is_sep_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    s2 = re.sub(
        r"\\(?:rowcolor|cellcolor)\s*(?:\[[^\]]*\])?\s*\{[^}]*\}", "", s
    ).strip()
    s2 = re.sub(r"^\[[\d.]+(?:ex|em|pt|cm|mm|in)\]\s*", "", s2)
    if not s2:
        return True
    return bool(_SEP_ROW_RE.match(s2))


def _strip_longtable_meta(block: str) -> str:
    if "\\endhead" not in block and "\\endlastfoot" not in block:
        return block

    def _find_end(marker: str) -> int:
        m = re.search(re.escape(marker) + r"\b", block)
        return m.end() if m else -1

    endfirsthead = _find_end("\\endfirsthead")
    endhead = _find_end("\\endhead")
    endfoot = _find_end("\\endfoot")
    endlastfoot = _find_end("\\endlastfoot")

    if endlastfoot >= 0:
        data_start = endlastfoot
    elif endfoot >= 0:
        data_start = endfoot
    elif endhead >= 0:
        data_start = endhead
    else:
        return block

    if endfirsthead >= 0:
        head_end = endfirsthead
    elif endhead >= 0:
        head_end = endhead
    else:
        return block

    if head_end >= data_start:
        return block

    return block[:head_end] + "\n" + block[data_start:]


def _parse_single_tabular(block: str, table_name: str = "") -> Optional[ParsedTable]:
    block = _strip_longtable_meta(block)
    raw_rows = _split_top_level(block, _ROW_SEP)
    matrix: Matrix = []
    header_rows_at_midrule: Optional[int] = None

    for raw in raw_rows:
        if header_rows_at_midrule is None:
            for ln in raw.splitlines():
                if re.match(r"^\s*\\midrule\b", ln):
                    header_rows_at_midrule = len(matrix)
                    break

        clean_lines = [ln for ln in raw.splitlines() if not _is_sep_line(ln)]
        row_s = " ".join(clean_lines).strip()
        if not row_s:
            continue

        raw_cells = _split_top_level(row_s, "&")
        cells: List[str] = []
        for rc in raw_cells:
            rc_s = rc.strip()
            rc_clean = re.sub(
                r"^\\(?:rowcolor|cellcolor)\s*(?:\[[^\]]*\])?\s*\{[^}]*\}\s*", "", rc_s
            ).strip()
            mc = re.match(r"\\multicolumn\s*\{(\d+)\}\s*", rc_clean)
            if mc:
                n_span = max(1, int(mc.group(1)))
                rest = rc_clean[mc.end() :]
                _, p = _brace_extract(rest, 0)
                rest = rest[p:]
                content, _ = (
                    _brace_extract(rest, 0)
                    if rest.startswith("{")
                    else (rest, len(rest))
                )
                cells.append(_strip_latex_cell(content))
                cells.extend([""] * (n_span - 1))
            else:
                cells.append(_strip_latex_cell(rc_s))
        if any(c.strip() for c in cells):
            matrix.append(cells)

    if not matrix:
        return None

    if header_rows_at_midrule is not None and 1 <= header_rows_at_midrule < len(matrix):
        header_rows = header_rows_at_midrule
    else:
        header_rows = 1

    return ParsedTable(
        matrix=normalize_rectangular(matrix),
        header_rows=header_rows,
        left_header_cols=0,
        format="latex",
        table_name=table_name,
    )


def _is_latex(raw: str) -> bool:
    s = raw.lstrip()
    return (
        s.startswith("%")
        or "\\begin{tabular}" in s
        or "\\begin{longtable}" in s
        or "{\\renewcommand" in s
    )


def _extract_pre_tabular_text(pre: str) -> str:
    m = re.search(r"\\parbox\s*", pre)
    if m:
        rest = pre[m.end() :]
        first_open = rest.find("{")
        if first_open >= 0:
            _, end1 = _brace_extract(rest, first_open)
            rest2 = rest[end1:].lstrip()
            second_open = rest2.find("{")
            if second_open >= 0:
                content, _ = _brace_extract(rest2, second_open)
                result = _strip_latex_cell(content).strip()
                if len(result) >= 30:
                    return result

    m2 = re.search(r"\\begin\s*\{minipage\}", pre)
    if m2:
        after = pre[m2.end() :]
        width_open = after.find("{")
        if width_open >= 0:
            _, end_w = _brace_extract(after, width_open)
            content_raw = after[end_w:]
        else:
            content_raw = after
        end_mini = content_raw.find(r"\end{minipage}")
        if end_mini >= 0:
            content_raw = content_raw[:end_mini]
        result = _strip_latex_cell(content_raw).strip()
        if len(result) >= 30:
            return result

    noindent_positions = [m.end() for m in re.finditer(r"\\noindent\b", pre)]
    if noindent_positions:
        text_after = pre[noindent_positions[0] :]
        text_after = re.sub(
            r"^\\(?:centering|small|large|footnotesize|normalsize|medskip|bigskip)\b\s*",
            "",
            text_after.lstrip(),
        )
        result = _strip_latex_cell(text_after).strip()
        result = re.sub(r"(?<!\w)\d+\.?\d*(?:pt|em|cm|mm|ex|bp)?\b", "", result)
        result = re.sub(r"\s{2,}", " ", result).strip()
        if len(result) >= 30:
            return result

    return ""


def _parse_latex_tables(raw: str) -> List[ParsedTable]:
    s = raw
    s = re.sub(r"\\\\([A-Za-z@{}])", lambda m: "\\" + m.group(1), s)

    env_re = re.compile(r"\\begin\{(tabular\*?|longtable)\}")
    results: List[ParsedTable] = []

    first_tabular = env_re.search(s)
    if first_tabular:
        pre_text = _extract_pre_tabular_text(s[: first_tabular.start()])
        if pre_text:
            results.append(
                ParsedTable(
                    matrix=[["Context"], [pre_text]],
                    header_rows=1,
                    left_header_cols=0,
                    format="latex",
                    table_name="context",
                )
            )

    pos = 0
    while True:
        m = env_re.search(s, pos)
        if not m:
            break
        env = m.group(1)
        begin_tag = f"\\begin{{{env}}}"
        end_tag = f"\\end{{{env}}}"
        after_begin = s[m.end() :]
        _, col_end = _brace_extract(after_begin, 0)
        block_start = m.end() + col_end
        depth = 1
        scan = block_start
        end_pos = -1
        while depth > 0 and scan < len(s):
            nb = s.find(begin_tag, scan)
            ne = s.find(end_tag, scan)
            if ne < 0:
                break
            if nb >= 0 and nb < ne:
                depth += 1
                scan = nb + len(begin_tag)
            else:
                depth -= 1
                if depth == 0:
                    end_pos = ne
                scan = ne + len(end_tag)
        block = s[block_start:end_pos] if end_pos >= 0 else s[block_start:]
        pos = (end_pos + len(end_tag)) if end_pos >= 0 else len(s)

        name = ""
        for row_guess in block.split("\\\\")[:3]:
            mc = re.search(
                r"\\multicolumn\s*\{\d+\}\s*\{[^}]*\}\s*\{([^}]+)\}", row_guess
            )
            if mc:
                candidate = _strip_latex_cell(mc.group(1))
                if candidate:
                    name = candidate
                    break

        pt = _parse_single_tabular(block, table_name=name)
        if pt is not None:
            results.append(pt)

    if not results:
        raise ValueError("No LaTeX tabular or longtable block was found.")
    return results


def parse_latex_tables(table_field: str) -> List[ParsedTable]:
    if not isinstance(table_field, str):
        raise ValueError("Expected a LaTeX table field string.")
    raw = table_field.strip()
    if not raw:
        raise ValueError("Expected a non-empty LaTeX table field.")

    if not _is_latex(raw):
        raise ValueError("Expected a LaTeX table field.")

    return _parse_latex_tables(raw)
