"""Assemble report_sections/NN_*.md into one root-level report.

Keeps the section files untouched. Adds a generated table of contents after the H1, then
concatenates sections 00..23 verbatim. Finally wraps prose lines to a readable width for the
combined file only, protecting math ($...$), links/images, inline code, tables, headings, lists,
and fenced code blocks so nothing renders differently. Image paths stay figures/NN.png so the
root-level report previews correctly.
"""
import os
import re
import glob

ROOT = r"d:/AstroML-Project"
SECT = os.path.join(ROOT, "report_sections")
OUT = os.path.join(ROOT, "Technical-Report.md")
WIDTH = 90

files = sorted(glob.glob(os.path.join(SECT, "[0-9][0-9]_*.md")))
assert len(files) == 24, f"expected 24 section files, found {len(files)}"

# --- collect heading text for the TOC (the '## N. Title' lines) ---
toc = []
for f in files:
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"^##\s+(\d+)\.\s+(.*\S)\s*$", line)
            if m:
                full = f"{m.group(1)}. {m.group(2)}"
                toc.append(full)
                break


def slug(text):
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)      # drop punctuation (periods, colons, ?)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s


# --- read the front matter, split off its H1 so we can inject the TOC right after it ---
with open(files[0], encoding="utf-8") as fh:
    fm = fh.read().split("\n")
assert fm[0].startswith("# "), "front matter must start with the H1 title"
h1 = fm[0]
fm_rest = "\n".join(fm[1:]).lstrip("\n")

toc_lines = ["## Table of contents", ""]
for t in toc:
    toc_lines.append(f"- [{t}](#{slug(t)})")
toc_block = "\n".join(toc_lines)

meta = ("*Technical report. N = 48,398 galaxies; AION-1-Large, frozen. 18 figures. "
        "Every number is measured from the committed analysis; interpretations are tagged "
        "\"interpreted\". Figure paths are relative to this file's folder, so preview from the "
        "repository root.*")

parts = [h1, "", meta, "", toc_block, "", fm_rest, ""]

# remaining sections verbatim
for f in files[1:]:
    with open(f, encoding="utf-8") as fh:
        parts.append(fh.read().rstrip("\n"))
        parts.append("")

raw = "\n".join(parts)

# --- prose wrapper: protect math / links / images / code as atomic tokens,
# swallowing any leading bracket and trailing punctuation so "$x$," stays one piece ---
PROT = re.compile(r"[(\[]*(?:\$\$.*?\$\$|\$[^$]*\$|!?\[[^\]]*\]\([^)]*\)|`[^`]*`)[)\],.;:!?]*")


def is_protected_line(line):
    s = line.strip()
    if s == "":
        return True
    if s.startswith("#"):
        return True                       # heading
    if s.startswith("|") or " | " in s:
        return True                       # table row
    if re.match(r"^[-*+]\s", s) or re.match(r"^\d+[.)]\s", s):
        return True                       # list item (keep whole; usually short)
    if s.startswith(">"):
        return True                       # blockquote
    if s.startswith("![") or s.startswith("!["):
        return True                       # image
    if re.match(r"^-{3,}\s*$", s) or re.match(r"^={3,}\s*$", s):
        return True                       # horizontal rule
    if s.startswith("<") and s.endswith(">"):
        return True                       # raw html
    return False


def tokenize(line):
    out, last = [], 0
    for m in PROT.finditer(line):
        pre = line[last:m.start()]
        out.extend(pre.split())
        out.append(m.group(0))            # atomic protected token
        last = m.end()
    out.extend(line[last:].split())
    return out


def wrap_prose(line, width=WIDTH):
    toks = tokenize(line)
    if not toks:
        return line
    lines, cur = [], ""
    for tk in toks:
        if cur == "":
            cur = tk
        elif len(cur) + 1 + len(tk) <= width:
            cur += " " + tk
        else:
            lines.append(cur)
            cur = tk
    if cur:
        lines.append(cur)
    return "\n".join(lines)


out_lines = []
in_code = False
for line in raw.split("\n"):
    if line.strip().startswith("```"):
        in_code = not in_code
        out_lines.append(line)
        continue
    if in_code or is_protected_line(line):
        out_lines.append(line)
    else:
        out_lines.append(wrap_prose(line))

text = "\n".join(out_lines).rstrip("\n") + "\n"
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(text)

nlines = text.count("\n")
nwords = len(text.split())
print(f"wrote {OUT}")
print(f"lines={nlines}  words={nwords}  sections={len(toc)}")
