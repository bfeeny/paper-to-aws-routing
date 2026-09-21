#!/usr/bin/env python3
"""Answer grading, shared by the labeller and the offline re-grader.

Third revision. The first treated everything non-letter as a number; the second
fixed that but still marked correct answers wrong whenever the model dressed
the final value up — `Answer: **6**`, `Answer: $\\boxed{6}$`, `45^\\circ`. That
defect is *tier-biased*, because the cheap model formats its answers more
eagerly than the expensive one, so it inflated every escalation rate and every
router AUC derived from them.

The lesson, twice over: a grader is a measuring instrument, and an instrument
whose error correlates with the thing being measured does not add noise, it
manufactures a result. Grading now lives in one module with a test suite.
"""

import re

LETTERS = "ABCDEFGHIJ"


def _strip_wrappers(s: str) -> str:
    """Remove presentation the model chose freely, never anything semantic."""
    s = s.strip()
    s = re.sub(r"\*+", "", s)                          # markdown bold/italic
    s = re.sub(r"\\boxed\s*\{(.+)\}", r"\1", s)        # \boxed{...}
    s = re.sub(r"\\(?:left|right|!|,|;|:|quad|qquad)", "", s)
    s = s.replace("$", "").replace("\\$", "")
    s = re.sub(r"\^\{?\\circ\}?|°", "", s)             # degrees
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\s*\{([^{}]*)\}", r"\1", s)
    return s.strip().rstrip(".").strip()


def norm_math(s: str) -> str:
    s = _strip_wrappers(s)
    s = re.sub(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\[dt]?frac(\d)(\d)", r"(\1)/(\2)", s)   # \frac12
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\pi", "pi").replace("\\sqrt", "sqrt")
    s = s.replace("dfrac", "frac")
    return s.lower()


def _as_float(s: str):
    # Convert LaTeX fractions first, so \frac{1}{2} and 0.5 compare equal.
    s = _strip_wrappers(s)
    s = re.sub(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\[dt]?frac(\d)(\d)", r"(\1)/(\2)", s)
    s = s.replace(",", "").strip()
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?\s*/\s*\(?(-?\d+(?:\.\d+)?)\)?", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _final_segment(reply: str) -> str:
    """Text after the last 'Answer:' marker, or the last non-empty line."""
    m = list(re.finditer(r"[Aa]nswer\s*:", reply))
    if m:
        return reply[m[-1].end():].strip()
    lines = [l for l in reply.strip().splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


def graded(reply: str | None, item: dict) -> bool:
    if not reply:
        return False
    kind = item.get("grade")
    want = str(item["answer"]).strip()
    seg = _final_segment(reply)

    if kind == "letter":
        # No re.I: the capture must be an uppercase letter, so "Answer: because"
        # cannot match "B". The bare-letter fallback is confined to the final
        # segment, because scanning the whole reply matches the pronoun "I".
        m = (re.findall(r"[Aa]nswer\s*:\s*\(?\*{0,2}([A-J])\*{0,2}\)?(?![a-zA-Z])", reply)
             or re.findall(r"\b([A-J])\b", seg))
        return bool(m) and m[-1].upper() == want.upper()

    if kind == "number":
        got, exp = _as_float(seg), _as_float(want)
        if got is not None and exp is not None:
            return abs(got - exp) < 1e-6
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", _strip_wrappers(seg))
        if not nums or exp is None:
            return False
        try:
            return abs(float(nums[-1].replace(",", "")) - exp) < 1e-6
        except ValueError:
            return False

    if kind == "math":
        if norm_math(seg) == norm_math(want):
            return True
        got, exp = _as_float(seg), _as_float(want)
        return got is not None and exp is not None and abs(got - exp) < 1e-6

    # "exact" (BBH): case- and punctuation-insensitive
    def n(s: str) -> str:
        return re.sub(r"[\s().,]", "", _strip_wrappers(s)).lower()

    return n(seg) == n(want)


SELFTEST = [
    ("Answer: **6**", "6", "math", True),
    ("Answer: $\\boxed{6}$", "6", "math", True),
    ("Answer: \\boxed{\\frac{1}{2}}", "\\frac{1}{2}", "math", True),
    ("Answer: 0.5", "\\frac{1}{2}", "math", True),
    ("Answer: 45^\\circ", "45", "math", True),
    ("Answer: \\dfrac{3}{4}", "3/4", "math", True),
    ("Answer: 7", "8", "math", False),
    ("Answer: **1,024**", "1024", "number", True),
    ("Answer: G", "G", "letter", True),
    ("Answer: **(E)**", "E", "letter", True),
    ("Answer: because it is so", "B", "letter", False),
    ("I think so.\nAnswer: C", "C", "letter", True),
    ("Answer: valid", "valid", "exact", True),
    ("Answer: **(A)**", "(A)", "exact", True),
    ("Answer: no", "yes", "exact", False),
]

if __name__ == "__main__":
    bad = 0
    for reply, ans, kind, exp in SELFTEST:
        got = graded(reply, {"answer": ans, "grade": kind})
        if got != exp:
            bad += 1
            print(f"FAIL [{kind}] {reply!r} want={ans!r} -> {got}, expected {exp}")
    print(f"{len(SELFTEST) - bad}/{len(SELFTEST)} passed")
    raise SystemExit(1 if bad else 0)
