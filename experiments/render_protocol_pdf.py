#!/usr/bin/env python3
"""Deterministically render the protocol Markdown to a readable A4 PDF.

This deliberately uses only matplotlib and the Python standard library so the
document can be rebuilt on the project system without Pandoc or a TeX engine.
It supports the Markdown constructs used by the protocol: headings, paragraphs,
lists, block quotes, fenced code, display equations, and multi-page tables.
"""

from __future__ import annotations

import argparse
import os
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("SOURCE_DATE_EPOCH", "1786665600")  # 2026-08-14 00:00 UTC

import matplotlib as mpl

mpl.use("pdf")
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.serif": ["DejaVu Serif"],
        "font.sans-serif": ["DejaVu Sans"],
        "font.monospace": ["DejaVu Sans Mono"],
        "pdf.fonttype": 42,
        "pdf.compression": 6,
        "axes.unicode_minus": False,
    }
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from matplotlib.patches import Rectangle


PAGE_W = 8.2677165354
PAGE_H = 11.692913386
LEFT = 0.67
RIGHT = 0.57
TOP = 0.66
BOTTOM = 0.58
CONTENT_W = PAGE_W - LEFT - RIGHT
INK = "#172033"
MUTED = "#536174"
BLUE = "#164e78"
LIGHT_BLUE = "#eef5f9"
LIGHT_GREY = "#f2f4f7"
RULE = "#aab3bf"
MATH_PARSER = MathTextParser("path")


def _math_source(value: str) -> str:
    """Normalize a few LaTeX commands not implemented by mathtext."""

    value = re.sub(r"\\mathcal\s*([A-Za-z])", r"\\mathit{\1}", value)
    value = re.sub(r"\\mathbb\s*([A-Za-z])", r"\\mathrm{\1}", value)
    value = re.sub(r"\\texttt\{([^{}]*)\}", r"\\mathrm{\1}", value)
    value = value.replace(r"\mathbf1", r"\mathbf{1}")
    # Removing the command leaves a grouped expression with the explanatory
    # subscript intact, which mathtext supports and remains mathematically clear.
    value = value.replace(r"\underbrace", "")
    return re.sub(r"\s+", " ", value).strip()


def _latex_plain(value: str) -> str:
    """Convert the protocol's compact LaTeX fragments to readable plain text."""

    value = value.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\bigl", "").replace("\\bigr", "")
    value = value.replace("\\Bigl", "").replace("\\Bigr", "")
    replacements = {
        r"\operatorname": "",
        r"\mathrm": "",
        r"\mathsf": "",
        r"\mathbf": "",
        r"\text": "",
        r"\underbrace": "",
        r"\overbrace": "",
    }
    for command, replacement in replacements.items():
        value = value.replace(command, replacement)

    # Apply simple one-level constructs repeatedly; nested leftovers remain
    # legible rather than risking an incorrect mathematical rewrite.
    for _ in range(5):
        prior = value
        value = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", value)
        value = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", value)
        value = re.sub(r"\\widehat\{([^{}]+)\}", r"\1-hat", value)
        value = re.sub(r"\\hat\{([^{}]+)\}", r"\1-hat", value)
        if value == prior:
            break

    symbols = {
        r"\gamma": "γ",
        r"\beta": "β",
        r"\pi": "π",
        r"\sigma": "σ",
        r"\rho": "ρ",
        r"\phi": "φ",
        r"\lambda": "λ",
        r"\alpha": "α",
        r"\mu": "μ",
        r"\varepsilon": "ε",
        r"\epsilon": "ε",
        r"\Delta": "Δ",
        r"\Theta": "Θ",
        r"\sum": "Σ",
        r"\prod": "Π",
        r"\times": "×",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\ne": "≠",
        r"\in": "∈",
        r"\notin": "∉",
        r"\subseteq": "⊆",
        r"\propto": "∝",
        r"\to": "→",
        r"\mid": " | ",
        r"\pm": "±",
        r"\ldots": "…",
        r"\cdots": "…",
        r"\operatorname": "",
        r"\mathbb": "",
        r"\mathcal": "",
        r"\texttt": "",
        r"\log": " log ",
        r"\exp": "exp",
        r"\min": "min",
        r"\max": "max",
        r"\bmod": " mod ",
        r"\lceil": "⌈",
        r"\rceil": "⌉",
        r"\lfloor": "⌊",
        r"\rfloor": "⌋",
    }
    for source, target in symbols.items():
        value = value.replace(source, target)
    value = value.replace("\\mathsf T", "T")
    value = value.replace("\\mathsf", "")
    value = value.replace("\\substack", "")
    value = value.replace("\\begin{aligned}", "").replace("\\end{aligned}", "")
    value = value.replace("\\begin{array}", "").replace("\\end{array}", "")
    value = value.replace("\\qquad", "   ").replace("\\quad", "  ")
    value = value.replace("\\text{", "{")
    value = value.replace("\\\\", " ; ")
    value = re.sub(r"\\([A-Za-z]+)", r"\1", value)
    value = value.replace("\\", "")
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def inline_plain(value: str) -> str:
    """Remove inline Markdown while retaining link destinations and maths."""

    value = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"\1 (\2)", value)
    value = re.sub(r"\\\((.*?)\\\)", lambda m: _latex_plain(m.group(1)), value)
    value = value.replace("**", "").replace("__", "")
    value = value.replace("`", "")
    value = re.sub(r"(?<!\w)\*([^*]+)\*", r"\1", value)
    return value.strip()


def wrap_lines(text: str, width: int) -> list[str]:
    if not text:
        return [""]
    return textwrap.wrap(
        text,
        width=max(8, width),
        break_long_words=True,
        break_on_hyphens=True,
        replace_whitespace=True,
        drop_whitespace=True,
    ) or [""]


def split_table_row(line: str) -> list[str]:
    return [inline_plain(cell.strip()) for cell in line.strip().strip("|").split("|")]


def is_table_rule(line: str) -> bool:
    cells = line.strip().strip("|").split("|")
    return bool(cells) and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells)


class ProtocolRenderer:
    def __init__(self, pdf: PdfPages, document_header: str) -> None:
        self.pdf = pdf
        self.document_header = document_header
        self.page_number = 0
        self.fig = None
        self.y = 0.0
        self.new_page()

    def new_page(self) -> None:
        if self.fig is not None:
            self._finish_page()
        self.page_number += 1
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor="white")
        self.y = PAGE_H - TOP
        if self.page_number > 1:
            self.fig.text(
                LEFT / PAGE_W,
                (PAGE_H - 0.34) / PAGE_H,
                self.document_header,
                ha="left",
                va="top",
                fontsize=7.4,
                color=MUTED,
                weight="medium",
            )
            self.fig.text(
                (PAGE_W - RIGHT) / PAGE_W,
                (PAGE_H - 0.34) / PAGE_H,
                "14 August 2026",
                ha="right",
                va="top",
                fontsize=7.4,
                color=MUTED,
            )
            self.fig.lines.append(
                mpl.lines.Line2D(
                    [LEFT / PAGE_W, (PAGE_W - RIGHT) / PAGE_W],
                    [(PAGE_H - 0.48) / PAGE_H] * 2,
                    transform=self.fig.transFigure,
                    color=RULE,
                    linewidth=0.45,
                )
            )

    def _finish_page(self) -> None:
        assert self.fig is not None
        self.fig.lines.append(
            mpl.lines.Line2D(
                [LEFT / PAGE_W, (PAGE_W - RIGHT) / PAGE_W],
                [0.40 / PAGE_H] * 2,
                transform=self.fig.transFigure,
                color=RULE,
                linewidth=0.4,
            )
        )
        self.fig.text(
            0.5,
            0.20 / PAGE_H,
            f"Page {self.page_number}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=MUTED,
        )
        self.pdf.savefig(self.fig, facecolor="white")
        plt.close(self.fig)
        self.fig = None

    def finish(self) -> None:
        if self.fig is not None:
            self._finish_page()

    def ensure(self, height: float) -> None:
        if self.y - height < BOTTOM:
            self.new_page()

    def spacer(self, inches: float) -> None:
        if self.y - inches < BOTTOM:
            self.new_page()
        else:
            self.y -= inches

    def text_block(
        self,
        text: str,
        *,
        size: float = 9.2,
        family: str = "DejaVu Serif",
        weight: str = "normal",
        style: str = "normal",
        color: str = INK,
        indent: float = 0.0,
        hanging: float = 0.0,
        prefix: str = "",
        before: float = 0.02,
        after: float = 0.095,
        line_factor: float = 1.34,
        background: str | None = None,
    ) -> None:
        text = inline_plain(text)
        available = CONTENT_W - indent
        prefix_width = hanging if prefix else 0.0
        chars = int(available * 72.0 / max(1.0, 0.54 * size))
        first_chars = max(8, chars - int(prefix_width * 72.0 / max(1.0, 0.54 * size)))
        first_wrap = wrap_lines(text, first_chars if prefix else chars)
        lines: list[tuple[str, float]] = []
        if prefix:
            lines.append((prefix + first_wrap[0], indent))
            continuation_text = " ".join(first_wrap[1:])
            if continuation_text:
                for line in wrap_lines(continuation_text, max(8, chars - int(prefix_width * 72.0 / (0.54 * size)))):
                    lines.append((line, indent + prefix_width))
        else:
            lines = [(line, indent) for line in first_wrap]

        line_h = size * line_factor / 72.0
        total = before + len(lines) * line_h + after
        self.ensure(total)
        self.y -= before
        if background:
            self.fig.patches.append(
                Rectangle(
                    ((LEFT + indent - 0.07) / PAGE_W, (self.y - len(lines) * line_h - 0.04) / PAGE_H),
                    (available + 0.12) / PAGE_W,
                    (len(lines) * line_h + 0.08) / PAGE_H,
                    transform=self.fig.transFigure,
                    facecolor=background,
                    edgecolor="none",
                    zorder=0,
                )
            )
        for line, line_indent in lines:
            self.fig.text(
                (LEFT + line_indent) / PAGE_W,
                self.y / PAGE_H,
                line,
                ha="left",
                va="top",
                fontsize=size,
                family=family,
                weight=weight,
                style=style,
                color=color,
                linespacing=1.0,
            )
            self.y -= line_h
        self.y -= after

    def heading(self, level: int, text: str) -> None:
        settings = {
            1: (20.0, 0.12, 0.24, BLUE),
            2: (14.2, 0.16, 0.12, BLUE),
            3: (11.3, 0.12, 0.075, "#213d58"),
            4: (10.0, 0.10, 0.06, "#213d58"),
        }
        size, before, after, color = settings.get(level, settings[4])
        text = inline_plain(text)
        chars = int(CONTENT_W * 72.0 / (0.54 * size))
        lines = wrap_lines(text, chars)
        line_h = size * 1.20 / 72.0
        # Reserve enough space for the first lines of the following block so a
        # section heading is not stranded at the foot of a page.
        self.ensure(before + len(lines) * line_h + after + 0.50)
        self.y -= before
        for line in lines:
            self.fig.text(
                LEFT / PAGE_W,
                self.y / PAGE_H,
                line,
                ha="left",
                va="top",
                fontsize=size,
                family="DejaVu Sans",
                weight="bold",
                color=color,
            )
            self.y -= line_h
        if level == 2:
            self.fig.lines.append(
                mpl.lines.Line2D(
                    [LEFT / PAGE_W, (PAGE_W - RIGHT) / PAGE_W],
                    [(self.y + 0.015) / PAGE_H] * 2,
                    transform=self.fig.transFigure,
                    color="#b8c8d5",
                    linewidth=0.55,
                )
            )
        self.y -= after

    def equation(self, source: str) -> None:
        expression = _math_source(source)
        size = 8.5
        try:
            parsed = MATH_PARSER.parse(
                "$" + expression + "$",
                dpi=72,
                prop=FontProperties(family="DejaVu Sans", size=size),
            )
            formula_h = max(0.17, parsed.height / 72.0)
            total = formula_h + 0.18
            self.ensure(total)
            self.fig.patches.append(
                Rectangle(
                    ((LEFT + 0.09) / PAGE_W, (self.y - formula_h - 0.09) / PAGE_H),
                    (CONTENT_W - 0.18) / PAGE_W,
                    (formula_h + 0.13) / PAGE_H,
                    transform=self.fig.transFigure,
                    facecolor=LIGHT_BLUE,
                    edgecolor="none",
                    zorder=0,
                )
            )
            self.fig.text(
                0.5,
                (self.y - 0.035) / PAGE_H,
                "$" + expression + "$",
                ha="center",
                va="top",
                fontsize=size,
                color="#173b55",
            )
            self.y -= total
        except Exception:
            # A future protocol edit may introduce unsupported TeX. Retain a
            # readable source-level formula rather than failing the document.
            self.text_block(
                _latex_plain(source),
                size=8.5,
                family="DejaVu Sans Mono",
                color="#173b55",
                indent=0.16,
                before=0.06,
                after=0.12,
                line_factor=1.32,
                background=LIGHT_BLUE,
            )

    def code(self, source: str) -> None:
        for paragraph in source.splitlines() or [""]:
            self.text_block(
                paragraph,
                size=8.0,
                family="DejaVu Sans Mono",
                indent=0.16,
                before=0.0,
                after=0.0,
                line_factor=1.30,
                background=LIGHT_GREY,
            )
        self.spacer(0.10)

    @staticmethod
    def _column_fracs(headers: Sequence[str]) -> list[float]:
        joined = " ".join(headers).lower()
        count = len(headers)
        if count == 2:
            return [0.23, 0.77]
        if count == 3:
            return [0.20, 0.32, 0.48]
        if count == 4:
            if "particles" in joined or "temperatures" in joined:
                return [0.13, 0.29, 0.29, 0.29]
            if "capability" in joined:
                return [0.22, 0.15, 0.43, 0.20]
            return [0.18, 0.22, 0.31, 0.29]
        if count == 5:
            if headers and headers[0].strip().lower() == "id":
                return [0.07, 0.17, 0.23, 0.29, 0.24]
            return [0.14, 0.16, 0.15, 0.26, 0.29]
        return [1.0 / count] * count

    def table(self, rows: Sequence[Sequence[str]]) -> None:
        if not rows:
            return
        ncol = max(len(row) for row in rows)
        normal = [list(row) + [""] * (ncol - len(row)) for row in rows]
        fracs = self._column_fracs(normal[0])
        font = 7.1 if ncol <= 3 else (6.7 if ncol == 4 else 6.25)
        padding = 0.047
        line_h = font * 1.30 / 72.0

        def prepared(row: Sequence[str]) -> tuple[list[list[str]], float]:
            output: list[list[str]] = []
            for cell, frac in zip(row, fracs):
                width_in = CONTENT_W * frac - 2 * padding
                chars = int(width_in * 72.0 / (0.51 * font))
                output.append(wrap_lines(inline_plain(cell), max(5, chars)))
            height = max(len(lines) for lines in output) * line_h + 2 * padding
            return output, height

        def draw_row(row: Sequence[str], *, header: bool) -> None:
            cells, height = prepared(row)
            self.ensure(height)
            x = LEFT
            top = self.y
            for cell_lines, frac in zip(cells, fracs):
                width = CONTENT_W * frac
                self.fig.patches.append(
                    Rectangle(
                        (x / PAGE_W, (top - height) / PAGE_H),
                        width / PAGE_W,
                        height / PAGE_H,
                        transform=self.fig.transFigure,
                        facecolor="#dfe8ef" if header else "white",
                        edgecolor=RULE,
                        linewidth=0.42,
                    )
                )
                ty = top - padding
                for line in cell_lines:
                    self.fig.text(
                        (x + padding) / PAGE_W,
                        ty / PAGE_H,
                        line,
                        ha="left",
                        va="top",
                        fontsize=font,
                        family="DejaVu Sans",
                        weight="bold" if header else "normal",
                        color="#17364e" if header else INK,
                    )
                    ty -= line_h
                x += width
            self.y -= height

        header = normal[0]
        _, header_h = prepared(header)
        _, first_h = prepared(normal[1]) if len(normal) > 1 else ([], 0.0)
        self.ensure(header_h + first_h + 0.05)
        draw_row(header, header=True)
        for row in normal[1:]:
            _, row_h = prepared(row)
            if self.y - row_h < BOTTOM:
                self.new_page()
                draw_row(header, header=True)
            draw_row(row, header=False)
        self.spacer(0.13)


def render_markdown(markdown: str, renderer: ProtocolRenderer) -> None:
    lines = markdown.splitlines()
    i = 0
    in_references = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            renderer.spacer(0.025)
            i += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            heading_text = inline_plain(heading.group(2))
            if heading_text == "Abstract":
                # Treat page one as a cover: begin the substantive protocol on
                # a fresh page instead of leaving an orphaned Abstract heading.
                if renderer.page_number == 1:
                    renderer.new_page()
                else:
                    renderer.ensure(1.20)
            if heading_text == "References":
                in_references = True
            renderer.heading(len(heading.group(1)), heading_text)
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1 if i < len(lines) else 0
            renderer.code("\n".join(code))
            continue

        if stripped == r"\[":
            i += 1
            equation: list[str] = []
            while i < len(lines) and lines[i].strip() != r"\]":
                equation.append(lines[i].strip())
                i += 1
            i += 1 if i < len(lines) else 0
            renderer.equation(" ".join(equation))
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and is_table_rule(lines[i + 1]):
            table = [split_table_row(line)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table.append(split_table_row(lines[i]))
                i += 1
            renderer.table(table)
            continue

        quote = re.match(r"^\s*>\s?(.*)$", line)
        if quote:
            parts: list[str] = []
            while i < len(lines):
                match = re.match(r"^\s*>\s?(.*)$", lines[i])
                if not match:
                    break
                parts.append(match.group(1))
                i += 1
            renderer.text_block(
                " ".join(parts),
                size=9.0,
                style="italic",
                color="#29465c",
                indent=0.25,
                before=0.07,
                after=0.12,
                background=LIGHT_BLUE,
            )
            continue

        bullet = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        numbered = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if bullet or numbered:
            if bullet:
                indent = 0.18 + len(bullet.group(1)) * 0.03
                prefix = "•  "
                body = bullet.group(2)
                hanging = 0.22
            else:
                indent = 0.14 + len(numbered.group(1)) * 0.03
                prefix = numbered.group(2) + ".  "
                body = numbered.group(3)
                hanging = 0.28
            renderer.text_block(
                body,
                prefix=prefix,
                hanging=hanging,
                indent=indent,
                size=8.35 if in_references else 9.0,
                before=0.0,
                after=0.032 if in_references else 0.055,
            )
            i += 1
            continue

        # Ordinary paragraph: merge source lines until a block boundary.
        paragraph = [stripped]
        i += 1
        while i < len(lines):
            candidate = lines[i]
            candidate_s = candidate.strip()
            if not candidate_s:
                break
            if re.match(r"^(#{1,4})\s+", candidate):
                break
            if candidate_s in {r"\[", "```"} or candidate_s.startswith("```"):
                break
            if candidate_s.startswith("|") and i + 1 < len(lines) and is_table_rule(lines[i + 1]):
                break
            if re.match(r"^\s*(?:[-*]|\d+\.)\s+", candidate) or candidate_s.startswith(">"):
                break
            paragraph.append(candidate_s)
            i += 1
        raw = " ".join(paragraph)
        is_emphasis = raw.startswith("**") and raw.endswith("**")
        renderer.text_block(
            raw,
            size=9.2,
            weight="bold" if is_emphasis else "normal",
            before=0.015,
            after=0.09,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    markdown = args.source.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixed_date = datetime(2026, 8, 14, 0, 0, 0, tzinfo=timezone.utc)
    with PdfPages(args.output) as pdf:
        info = pdf.infodict()
        first_heading = next(
            (
                line.removeprefix("# ").strip()
                for line in markdown.splitlines()
                if line.startswith("# ")
            ),
            "OrderSMC-DAG: Experimental Protocol",
        )
        info["Title"] = inline_plain(first_heading)
        info["Author"] = "OrderSMC-DAG project"
        info["Subject"] = "Publication-oriented experimental design for Bayesian DAG structure learning"
        info["Keywords"] = "OrderSMC Bayesian DAG SMC mixed data experimental protocol reproducibility"
        info["Creator"] = "OrderSMC deterministic matplotlib protocol renderer"
        info["CreationDate"] = fixed_date
        info["ModDate"] = fixed_date
        version_match = re.search(r"Version\s+([0-9.]+)", first_heading)
        document_header = "OrderSMC-DAG Experimental Protocol"
        if version_match:
            document_header += f" v{version_match.group(1)}"
        renderer = ProtocolRenderer(pdf, document_header)
        render_markdown(markdown, renderer)
        renderer.finish()


if __name__ == "__main__":
    main()
