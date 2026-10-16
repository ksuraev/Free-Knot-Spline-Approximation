import sys
from pathlib import Path

import pandas as pd
import pymupdf

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(ROOT))

PLOTS = ROOT / "plots" / "2026-10-12"
RESULTS = SCRIPT_DIR / "results.csv"
OUTPUT = SCRIPT_DIR / "experiment_report_xmin_xmax.pdf"

PAGE = pymupdf.paper_rect("a4")
PAGE_WIDTH = PAGE.height
PAGE_HEIGHT = PAGE.width


def format_value(value):
    if pd.isna(value):
        return "-"
    if isinstance(value, str):
        return value
    try:
        value = float(value)
        return str(int(value)) if value.is_integer() else f"{value:.6g}"
    except (TypeError, ValueError):
        return str(value)


def add_pdf_plot(page, rect, path):
    """Insert the first page of an existing PDF."""
    if not path.exists():
        page.insert_textbox(
            rect,
            f"Missing:\n{path.name}",
            fontsize=9,
            align=pymupdf.TEXT_ALIGN_CENTER,
        )
        print(f"Missing: {path}")
        return

    with pymupdf.open(path) as source:
        page.show_pdf_page(rect, source, 0, keep_proportion=True)


def grid_rects(rows, cols, top=50, bottom=0):
    """Split the page into a rows x cols grid."""
    width = PAGE_WIDTH / cols
    height = (PAGE_HEIGHT - top - bottom) / rows

    return [
        pymupdf.Rect(
            col * width,
            top + row * height,
            (col + 1) * width,
            top + (row + 1) * height,
        )
        for row in range(rows)
        for col in range(cols)
    ]


def add_plots(page, plots, rects):
    for path, rect in zip(plots, rects):
        plot_rect = pymupdf.Rect(rect.x0, rect.y0 + 15, rect.x1, rect.y1)
        add_pdf_plot(page, plot_rect, path)


def add_results_text(page, record, rect):
    left_lines = []
    right_lines = []

    groups = [
        ("x_min", left_lines),
        ("x_max", right_lines),
        ("x_min_rev", left_lines),
        ("x_max_rev", right_lines),
    ]

    for knot_type, lines in groups:
        lines.append(knot_type)
        lines.append(f"  z: {format_value(record[f'{knot_type}_final_z'])}")
        lines.append(
            f"  alt per interval: {format_value(record[f'{knot_type}_alt_per_interval'])}"
        )
        lines.append(
            f"  knot is alt pt: {format_value(record[f'{knot_type}_knot_is_alt_pt'])}"
        )
        lines.append(
            f"  differentiable: {format_value(record[f'{knot_type}_differentiable'])}"
        )
        lines.append("")

    mid = (rect.x0 + rect.x1) / 2

    left_rect = pymupdf.Rect(rect.x0, rect.y0, mid - 5, rect.y1)
    right_rect = pymupdf.Rect(mid + 5, rect.y0, rect.x1, rect.y1)

    page.insert_textbox(
        left_rect,
        "\n".join(left_lines),
        fontsize=9,
        lineheight=1.2,
    )

    page.insert_textbox(
        right_rect,
        "\n".join(right_lines),
        fontsize=9,
        lineheight=1.2,
    )


def new_page(document, function, m, k):
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((20, 25), f"{function}   k={k}   m={m}", fontsize=15)

    return page


def add_page(document, record):
    function = str(record["function"]).strip()
    m = int(record["m"])
    k = int(record["k"])

    prefix = f"{function}_k{k}_m{m}"

    plots = [
        PLOTS / f"{prefix}_final_x_min.pdf",
        PLOTS / f"{prefix}_final_x_max.pdf",
        PLOTS / f"{prefix}_final_x_min_rev.pdf",
        PLOTS / f"{prefix}_final_x_max_rev.pdf",
    ]

    page = new_page(document, function, m, k)

    rects = grid_rects(rows=2, cols=2, top=40, bottom=145)
    add_plots(page, plots, rects)

    add_results_text(
        page,
        record,
        pymupdf.Rect(
            20,
            PAGE_HEIGHT - 140,
            PAGE_WIDTH - 20,
            PAGE_HEIGHT - 5,
        ),
    )


if __name__ == "__main__":
    document = pymupdf.open()

    results = pd.read_csv(RESULTS)
    results.columns = results.columns.str.strip()

    results = results[results["k"].isin([1, 2, 3, 4])]

    results = results.sort_values(["function", "k", "m"])

    for _, record in results.iterrows():
        add_page(document, record)

    if OUTPUT.exists():
        OUTPUT.unlink()

    document.save(
        OUTPUT,
        garbage=4,
        deflate=True,
    )

    document.close()

    print(f"Created {OUTPUT}")
