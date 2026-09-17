"""Convert markdown text to a styled PDF file."""

import os
import re
from fpdf import FPDF


def _sanitize(text: str) -> str:
    """Replace Unicode chars that latin-1 can't handle."""
    replacements = {
        "\u2192": "->", "\u2190": "<-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2022": "*", "\u2026": "...", "\u00b0": " deg",
        "\u20f0": "+", "\u20e3": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Strip any remaining non-latin1 chars
    return text.encode("latin-1", errors="replace").decode("latin-1")


class PlanPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 12, "Trip Plan", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, _sanitize(f"  {title}"), new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(41, 128, 185)
        self.cell(0, 8, _sanitize(title), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, _sanitize(text))
        self.ln(2)

    def table_row(self, cols, bold=False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        w = (self.w - 20) / len(cols)
        for col in cols:
            self.cell(w, 7, _sanitize(str(col)), border=1)
        self.ln()


def markdown_to_pdf(md_text: str, output_path: str) -> str:
    """Parse markdown and write a styled PDF. Returns the absolute file path."""
    pdf = PlanPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Table separator line (|---|---|)
        if re.match(r"^\|[\s\-|]+\|$", line):
            i += 1
            continue

        # Table row
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            # Check if next line is also a table row
            is_header = False
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("|") and not re.match(r"^\|[\s\-|]+\|$", next_line):
                    # This might be header row
                    is_header = True
            pdf.table_row(cells, bold=(i == 0 or is_header))
            i += 1
            continue

        # Heading 1
        if line.startswith("# "):
            pdf.chapter_title(line[2:].strip())
            i += 1
            continue

        # Heading 2
        if line.startswith("## "):
            pdf.section_title(line[3:].strip())
            i += 1
            continue

        # Heading 3
        if line.startswith("### "):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, _sanitize(line[4:].strip()), new_x="LMARGIN", new_y="NEXT")
            i += 1
            continue

        # Bold line
        if line.startswith("**") and line.endswith("**"):
            pdf.set_font("Helvetica", "B", 10)
            clean = line.strip("*").strip()
            pdf.cell(0, 7, _sanitize(clean), new_x="LMARGIN", new_y="NEXT")
            i += 1
            continue

        # Horizontal rule
        if line.startswith("---") or line.startswith("***"):
            pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue

        # Bullet point
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            text = re.sub(r"^[\s]*[-*]\s+", "", line)
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # strip bold markers
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(8)
            pdf.cell(0, 6, _sanitize(f"  {text}"), new_x="LMARGIN", new_y="NEXT")
            i += 1
            continue

        # Numbered list
        m = re.match(r"^(\d+)\.\s+(.+)", line)
        if m:
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(2))
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(8)
            pdf.cell(0, 6, _sanitize(f"  {m.group(1)}. {text}"), new_x="LMARGIN", new_y="NEXT")
            i += 1
            continue

        # Regular text (strip markdown bold/italic)
        text = line
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)

        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _sanitize(text))
        pdf.ln(1)
        i += 1

    pdf.output(output_path)
    return os.path.abspath(output_path)
