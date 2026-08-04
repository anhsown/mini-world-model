"""Convert the Phase-1 markdown reports to PDF.

Pipeline: markdown -> styled HTML -> Chrome headless --print-to-pdf.
Chrome is used because it renders Vietnamese diacritics, emoji and wide
tables correctly without needing a LaTeX toolchain.

Usage:
    python md_to_pdf.py            # all .md in this folder
    python md_to_pdf.py 01_*.md    # specific files
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
PDF_DIR = HERE / "pdf"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }

body {
  font-family: "Segoe UI", "Inter", system-ui, sans-serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: #1a1a1a;
  margin: 0;
}

h1 {
  font-size: 19pt; font-weight: 700; color: #0f172a;
  border-bottom: 2.5px solid #0f172a;
  padding-bottom: 6px; margin: 0 0 16px;
}
h2 {
  font-size: 13.5pt; font-weight: 700; color: #0f172a;
  margin: 22px 0 8px; padding-top: 4px;
  border-top: 1px solid #e2e8f0;
  break-after: avoid; page-break-after: avoid;
}
h2:first-of-type { border-top: none; }
h3 {
  font-size: 11.5pt; font-weight: 650; color: #1e293b;
  margin: 15px 0 6px;
  break-after: avoid; page-break-after: avoid;
}
h4 { font-size: 10.5pt; font-weight: 650; margin: 12px 0 4px; }

p { margin: 7px 0; }
ul, ol { margin: 7px 0; padding-left: 22px; }
li { margin: 3px 0; }

/* Tables: the reports are table-heavy, keep them compact and unbroken. */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0 14px;
  font-size: 8.8pt;
  break-inside: avoid; page-break-inside: avoid;
}
thead { display: table-header-group; }
th {
  background: #0f172a; color: #fff;
  text-align: left; font-weight: 600;
  padding: 5px 7px; border: 1px solid #0f172a;
}
td { padding: 4px 7px; border: 1px solid #cbd5e1; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f8fafc; }

code {
  font-family: "Cascadia Mono", Consolas, monospace;
  font-size: 8.6pt; background: #f1f5f9;
  padding: 1px 4px; border-radius: 3px;
  color: #0f172a; word-break: break-word;
}
pre {
  background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #64748b;
  padding: 8px 10px; border-radius: 4px; overflow-x: auto;
  break-inside: avoid; page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8.4pt; }

blockquote {
  margin: 10px 0; padding: 8px 12px;
  background: #fffbeb; border-left: 4px solid #f59e0b;
  break-inside: avoid; page-break-inside: avoid;
}
blockquote p { margin: 3px 0; }

hr { border: none; border-top: 1px solid #e2e8f0; margin: 18px 0; }
a { color: #1d4ed8; text-decoration: none; }
strong { font-weight: 650; color: #0f172a; }

/* Keep a heading with the block that follows it. */
h2 + table, h3 + table, h2 + p, h3 + p { break-before: avoid; page-break-before: avoid; }
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head><body>{body}</body></html>"""


def find_chrome() -> str:
    for path in CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    found = shutil.which("chrome") or shutil.which("msedge")
    if found:
        return found
    raise SystemExit("Khong tim thay Chrome hoac Edge de in PDF.")


def render_html(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    # Point cross-links at the generated PDFs instead of the .md sources.
    text = re.sub(r"\]\((\d\d_[\w]+)\.md\)", r"](\1.pdf)", text)
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "nl2br"],
    )
    title = md_path.stem
    match = re.search(r"^#\s+(.+)$", text, re.M)
    if match:
        title = match.group(1).strip()
    return TEMPLATE.format(title=title, css=CSS, body=body)


def to_pdf(md_path: Path, chrome: str, workdir: Path) -> Path:
    html_path = workdir / (md_path.stem + ".html")
    html_path.write_text(render_html(md_path), encoding="utf-8")
    pdf_path = PDF_DIR / (md_path.stem + ".pdf")

    profile = workdir / f"profile_{md_path.stem}"
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-first-run",
        "--no-pdf-header-footer",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not pdf_path.exists():
        raise RuntimeError(
            f"Chrome khong tao duoc PDF cho {md_path.name}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return pdf_path


def main() -> None:
    targets = (
        [Path(p).resolve() for p in sys.argv[1:]]
        if len(sys.argv) > 1
        else sorted(HERE.glob("*.md"))
    )
    if not targets:
        raise SystemExit("Khong co file .md nao.")

    PDF_DIR.mkdir(exist_ok=True)
    chrome = find_chrome()
    print(f"Dung: {chrome}\nXuat ra: {PDF_DIR}\n")

    made = []
    with tempfile.TemporaryDirectory(prefix="md2pdf_") as tmp:
        workdir = Path(tmp)
        for md_path in targets:
            pdf_path = to_pdf(md_path, chrome, workdir)
            size_kb = pdf_path.stat().st_size / 1024
            print(f"  OK  {md_path.name:32s} -> {pdf_path.name:32s} {size_kb:7.1f} KB")
            made.append(pdf_path)

    # Merge everything into one deliverable, README first.
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("\n(pypdf khong co - bo qua ban gop)")
        return

    order = sorted(made, key=lambda p: (not p.stem.startswith("README"), p.stem))
    writer = PdfWriter()
    for pdf_path in order:
        for page in PdfReader(str(pdf_path)).pages:
            writer.add_page(page)
    merged = PDF_DIR / "PHASE1_ALL_REPORTS.pdf"
    with open(merged, "wb") as handle:
        writer.write(handle)
    print(f"\n  OK  ban gop -> {merged.name}  "
          f"({len(writer.pages)} trang, {merged.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
