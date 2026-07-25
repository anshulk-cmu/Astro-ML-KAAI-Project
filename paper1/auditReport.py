"""Check every number written in the report against the artifact it came from.

Reading rule 4 says a number enters the report only if a diagnostic script produced it and an
automated comparison confirms the written value. This is that comparison. It reads the report,
pulls every numeric token out of the requested sections, and looks for a value in the results
JSONs that matches at the precision the report writes it to.

Anything that does not match a stored value must be declared, with a reason, in CONSTANTS.
The point is that every exception is visible rather than silently tolerated.

    python -B paper1/auditReport.py            every written section
    python -B paper1/auditReport.py 5          one section
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C

REPORT = C.PAPER1 / "techReportPaperScopingV1.md"
# A minus sign only counts as one when it does not follow a word character, so "mod-180" and
# "0.00-0.05" are read as the numbers they contain rather than as negatives.
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?", re.I)

# Values that are definitions, targets or arithmetic rather than measurements. Each carries the
# reason it is not expected to appear in a results file.
CONSTANTS = {
    "0.5": "half, in the theta_hat = (1/2) atan2 definition and in tolerance arithmetic",
    "-1.0": "the exact-equivariance target for a slope, signed",
    "-2.0": "the axial doubling factor carrying a sign, as in exp(-2 i phi)",
    "85.0": "90 minus the 5 degree node tolerance: arithmetic on two stored values",
    "95.0": "the coverage level of every interval, as in '95% CI'",
    "135.0": "an antinode location on the circle, a place rather than a measurement",
    "127000.0": "8 x 15,893 encodes, arithmetic on values that are themselves stored",
    "1e-09": "the tolerance the catalog identities are verified to",
    "1e-06": "the tolerance the inclination identity is verified to",
    "-99.0": "the sentinel value itself; d0 records the rule as sentinel_min = -90 and the counts",
    "360.0": "the full circle, in the definition of a period 360/k",
    "576.0": "image tokens implied by the cutout geometry, arithmetic not measurement",
    "256.0": "the pretraining token budget, a property of the model rather than of this run",
    "97.5": "the upper percentile of a 95 per cent interval",
    "179.0": "a worked example of the wrap, not a measurement",
    "178.0": "a worked example of the wrap, not a measurement",
    "5.1": "the ratio of two stored values, 10.329 over 2.027",
}

SECTION = re.compile(r"^#{2,3}\s+(\d+)(?:\.(\d+))?\.?\s+")
BANNED = ["rerun", "re-run", "defect", "bug", "workaround", "prior work", "old code",
          "pre-harness", "corrections ledger", "TODO", "FIXME"]


def load_values():
    """Every value in every results file: numbers to compare against, and strings so that
    recorded package versions and paths can satisfy a token like 3.11 in 3.11.14."""
    vals, texts = [], []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, bool):
            pass
        elif isinstance(node, (int, float)):
            vals.append((path, float(node)))
        elif isinstance(node, str):
            texts.append(node)

    for p in sorted(C.RESULTS.glob("*.json")):
        walk(json.loads(p.read_text()), p.name)
    return vals, texts


def matches(token, value):
    """True if the stored value rounds to the written token at the precision it is written to.

    Compared as |stored - written| <= half a unit in the last written place, which is the
    definition of 'rounds to' and avoids the double rounding that a round-then-compare does.
    """
    txt = token.replace(",", "").lower()
    try:
        written = float(txt)
    except ValueError:
        return False
    if "e" in txt:
        mant, exp = txt.split("e")
        dp = len(mant.split(".")[1]) if "." in mant else 0
        half = 0.5 * 10 ** (-dp) * 10 ** int(exp)
    else:
        dp = len(txt.split(".")[1]) if "." in txt else 0
        half = 0.5 * 10 ** (-dp)
    return abs(value - written) <= half * (1 + 1e-9) + 1e-300


def scrub(line, section):
    """Remove the parts of a line that carry digits which are not measurements."""
    s = re.sub(r"`[^`]*`", " ", line)
    s = re.sub(r"&#\d+;", " ", s)
    s = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", s)
    s = re.sub(r"\b(float|int|uint)\d+\b", " ", s)
    s = re.sub(r"SHA-\d+", " ", s)
    s = re.sub(r"\bRTX\s*\d+\s*\w*", " ", s)               # a hardware product name
    s = re.sub(r"\(\d{4}\)", " ", s)                       # citation years
    s = re.sub(r"\b\d+(st|nd|rd|th)\b", " ", s)            # ordinals: 99th percentile
    s = re.sub(r"Sections?\s+\d+(\.\d+)?(\s*(to|and|,)\s*\d+(\.\d+)?)*", " ", s)
    s = re.sub(r"Diagnostics?\s+\d+([ ,and]+\d+)*", " ", s)
    s = re.sub(r"Appendix [A-F]", " ", s)
    s = re.sub(r"\b(19|20)\d\d-\d\d-\d\d\b", " ", s)
    if section == "13" and s.lstrip().startswith("|"):
        cells = s.split("|")
        if len(cells) > 4:
            cells[1] = cells[3] = " "   # the row number and the section-reference column
            s = "|".join(cells)
    return s


def sections_of(text):
    """Split the report into (section key, heading, body lines)."""
    out, cur, head = [], [], None
    for line in text.splitlines():
        m = SECTION.match(line)
        if m:
            if head is not None:
                out.append((head, cur))
            head, cur = (m.group(1), line), []
        elif head is not None:
            cur.append(line)
    if head is not None:
        out.append((head, cur))
    return out


def audit(want=None):
    text = REPORT.read_text(encoding="utf-8")
    values, texts = load_values()
    checked = unmatched = declared = in_string = 0
    problems = []

    for (num, heading), body in sections_of(text):
        if want and num != want:
            continue
        for line in body:
            if line.strip().startswith("![") or line.strip().startswith("```"):
                continue
            clean = scrub(line, num)
            for m in NUMBER.finditer(clean):
                tok = m.group(0)
                checked += 1
                if str(float(tok.replace(",", ""))) in CONSTANTS or tok in CONSTANTS:
                    declared += 1
                    continue
                if any(matches(tok, v) for _, v in values):
                    continue
                # a value written as a percentage of a stored fraction
                tail = clean[m.end():m.end() + 9].lstrip()
                if (tail.startswith("%") or tail.startswith("percent")) and \
                        any(matches(tok, v * 100.0) for _, v in values):
                    continue
                if any(tok in t for t in texts):
                    in_string += 1
                    continue
                unmatched += 1
                problems.append((num, tok, line.strip()[:150]))

    print(f"numeric tokens checked: {checked}")
    print(f"  matched to an artifact value: {checked - unmatched - declared - in_string}")
    print(f"  matched inside a recorded string (versions, paths): {in_string}")
    print(f"  declared constants:           {declared}")
    print(f"  UNMATCHED:                    {unmatched}")
    for num, tok, line in problems:
        print(f"    section {num}: {tok!r} in: {line}")

    bad_words = [(w, ln.strip()[:120]) for ln in text.splitlines()
                 for w in BANNED if w.lower() in ln.lower()]
    print(f"logbook vocabulary hits: {len(bad_words)}")
    for w, ln in bad_words:
        print(f"    {w!r}: {ln}")

    pend = [ln.strip()[:90] for ln in text.splitlines() if "*pending*" in ln]
    print(f"pending markers: {len(pend)}")
    return unmatched + len(bad_words)


if __name__ == "__main__":
    sys.exit(1 if audit(sys.argv[1] if len(sys.argv) > 1 else None) else 0)
