# Put all experiments into a single PDF
import sys
from pathlib import Path

import pandas as pd
import pymupdf

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(ROOT))

PLOTS = ROOT / "plots" / "2026-09-24"
K1_RESULTS = SCRIPT_DIR / "k1results.csv"
OUTPUT = SCRIPT_DIR / "experiment_report_k1.pdf"

PAGE = pymupdf.paper_rect("a4")
PAGE_WIDTH = PAGE.height
PAGE_HEIGHT = PAGE.width


def format_value(value):
    if pd.isna(value):
        return "-"
    try:
        value = float(value)
        return str(int(value)) if value.is_integer() else f"{value:.6g}"
    except (TypeError, ValueError):
        return str(value)


def add_pdf_plot(page, rect, path):
    """Insert the first page of an existing PDF directly."""

    if not path.exists():
        page.insert_textbox(
            rect, f"Missing:\n{path.name}", fontsize=9, align=pymupdf.TEXT_ALIGN_CENTER
        )
        print(f"Missing: {path}")
        return

    with pymupdf.open(path) as source:
        page.show_pdf_page(rect, source, 0, keep_proportion=True)


def grid_rects(rows, cols, top=50, bottom=0):
    """Split the page into a simple rows × cols grid."""

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


def add_plots(page, plots, titles, rects):
    for path, title, rect in zip(plots, titles, rects):
        page.insert_text(
            (rect.x0 + 10, rect.y0 + 12),
            title,
            fontsize=11,
        )

        plot_rect = pymupdf.Rect(
            rect.x0,
            rect.y0 + 15,
            rect.x1,
            rect.y1,
        )

        add_pdf_plot(page, plot_rect, path)


def add_results_text(page, record, rect):
    fields = [
        ("Initial status", "initialstatus"),
        ("Initial case", "initialcase"),
        ("Initial theta (chosen equidistant)", "initialtheta"),
        ("Initial d_max", "initialdmax"),
        ("theta_min (nurnberger)", "thetastart"),
        ("Final status", "finalstatus"),
        ("Final case", "finalcase"),
        ("final d_max", "finaldmax"),
        ("Iterations", "iterations"),
    ]

    midpoint = len(fields) // 2

    left = "\n".join(
        f"{label}: {format_value(record[key])}" for label, key in fields[:midpoint]
    )

    right = "\n".join(
        f"{label}: {format_value(record[key])}" for label, key in fields[midpoint:]
    )

    half = rect.width / 2

    page.insert_textbox(
        pymupdf.Rect(rect.x0, rect.y0, rect.x0 + half, rect.y1),
        left,
        fontsize=10,
        lineheight=1.25,
    )

    page.insert_textbox(
        pymupdf.Rect(rect.x0 + half, rect.y0, rect.x1, rect.y1),
        right,
        fontsize=10,
        lineheight=1.25,
    )


def new_page(document, function, k, m):
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_text((20, 25), f"{function}   k={k}   m={m}", fontsize=15)
    return page


def add_k1_page(document, record):
    function = str(record["function"]).strip()
    m = int(record["m"])
    k = 1

    prefix = f"{function}_k{k}_m{m}"

    plots = [
        PLOTS / f"{prefix}_equidistant.pdf",
        PLOTS / f"{prefix}_initial.pdf",
        PLOTS / f"psi_opt_path_{prefix}.pdf",
        PLOTS / f"{prefix}_final.pdf",
    ]

    titles = [
        "Equidistant approximation",
        "Initial approximation",
        "Descent path",
        "Final approximation",
    ]

    page = new_page(document, function, k, m)
    rects = grid_rects(rows=2, cols=2, top=40, bottom=120)
    add_plots(page, plots, titles, rects)

    add_results_text(
        page,
        record,
        pymupdf.Rect(20, PAGE_HEIGHT - 115, PAGE_WIDTH - 20, PAGE_HEIGHT - 5),
    )


def main():
    document = pymupdf.open()

    # k = 1
    results = pd.read_csv(K1_RESULTS)
    results = results.sort_values(["function", "m"])

    for _, record in results.iterrows():
        add_k1_page(document, record)

    if OUTPUT.exists():
        OUTPUT.unlink()

    document.save(OUTPUT, garbage=4, deflate=True)
    document.close()

    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
