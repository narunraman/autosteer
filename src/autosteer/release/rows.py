"""Turn one sampled question (ElementSampler.sample_question) into dataset rows, one per part.

Used by the exporter and by /api/sample, so both return the same fields with the same values. The helpers
(content_key, split_tags, row_problems, the risk-pair parsers) are ported from STEER-generation
src/release/export_steer_me.py and export_risk_pairs.py (recover-generators@9d851d4).
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
import subprocess
from fractions import Fraction
from importlib import metadata as importlib_metadata
from pathlib import Path

from autosteer.taxonomy import taxonomy_for_element

from .consistency import (COMMON_RATIO, ELLSBERG, MIRROR_TRADE, PAIRWISE_COMPLETE, REFLECTION, SAME_BRAND,
                          ellsberg_rule, mirror_trade_rule, pairwise_complete_rule, same_brand_rule, same_side_rule)

PACKAGE_DIR = Path(__file__).resolve().parent.parent
ELEMENTS_JSON = PACKAGE_DIR / "web" / "content" / "elements.json"

# ai_assistant: the reader is an AI assistant in charge of the decision (sunk_cost)
PERSPECTIVES = {"first_person", "second_person", "third_person", "anon", "man", "woman", "ai_assistant"}
PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
NUMPY_REPR_RE = re.compile(r"\bnp\.(float|int|bool|str)_?\d*\(")
BAD_OPTION_TEXT = {"", "No options found", "nan", "None", "NaN", "{}", "[]"}


# --------------------------------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------------------------------
def git_state() -> dict:
    """Commit and cleanliness of the checkout the package runs from ({} when not run from a git checkout)."""
    root = PACKAGE_DIR.parent.parent

    def run(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)

    head = run("rev-parse", "HEAD")
    if head.returncode != 0:
        return {}
    status = run("status", "--porcelain", "--untracked-files=no").stdout.splitlines()
    return {"sha": head.stdout.strip(), "short_sha": head.stdout.strip()[:7], "dirty": bool(status),
            "dirty_files": [line[3:] for line in status]}


def package_version() -> str:
    try:
        return importlib_metadata.version("autosteer")
    except importlib_metadata.PackageNotFoundError:
        return "0+unknown"


@functools.lru_cache(maxsize=None)
def generator_id() -> str:
    """'autosteer@<version>+<short sha>'; '.dirty' is appended when tracked files differ from the commit."""
    git = git_state()
    if not git:
        return f"autosteer@{package_version()}"
    return f"autosteer@{package_version()}+{git['short_sha']}" + (".dirty" if git["dirty"] else "")


# --------------------------------------------------------------------------------------------------------------
# Element catalogue (web/content/elements.json is the single source for benchmark, setting, module and status)
# --------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def catalog() -> dict[str, dict]:
    return {e["element"]: e for e in json.loads(ELEMENTS_JSON.read_text(encoding="utf-8"))}


def element_info(element: str) -> dict:
    """{benchmark, setting, module, status} of an element; taxonomy.py fills in elements the site does not list."""
    e = catalog().get(element)
    if e is not None:
        return {"benchmark": e["version"], "setting": e["setting"], "module": e["module"], "status": e["status"]}
    t = taxonomy_for_element(element)
    return {"benchmark": None, "setting": t["setting"], "module": t["module"], "status": "unlisted"}


# --------------------------------------------------------------------------------------------------------------
# Helpers ported from the staged exporters
# --------------------------------------------------------------------------------------------------------------
def split_tags(raw) -> tuple[str | None, list[str]]:
    """Template `tags` packs the perspective with other descriptors (e.g. 'third_person;man')."""
    toks = [t.strip() for t in str(raw or "").split(";") if t.strip()]
    persp = [t for t in toks if t in PERSPECTIVES]
    rest = [t for t in toks if t not in PERSPECTIVES]
    if len(persp) == 2 and "third_person" in persp:  # 'third_person;man' -> 'man'
        persp = [p for p in persp if p != "third_person"]
    if len(persp) > 1:
        return None, toks
    return (persp[0] if persp else None), rest


def content_key(element: str, parts: list[dict]) -> str:
    """Order-independent content hash of a (possibly multi-part) question: element, texts, sorted option texts."""
    h = hashlib.sha1(element.encode())
    for p in parts:
        h.update(b"\x00Q" + p["question"].encode())
        for t in sorted(p["options"]):
            h.update(b"\x00O" + t.encode())
    return h.hexdigest()


PCT_RE = re.compile(r"(\d+(?:\.\d+)?)%")
AMT_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d+")
QUOTED_RE = re.compile(r'"([^"]+)"')


def _numbers(text: str) -> tuple[list[Fraction], list[int]]:
    pcts = [Fraction(m) / 100 for m in PCT_RE.findall(text)]
    amts = [int(a.replace(",", "")) for a in AMT_RE.findall(PCT_RE.sub(" ", text))]
    return pcts, amts


def parse_pair(a: str, b: str, c: str, d: str) -> dict:
    """certainty_effect: displayed numbers of options A, B (part 0) and C, D (part 1)."""
    (pa_, xa), (pb, yb), (pc, xc), (pd, yd) = (_numbers(t) for t in (a, b, c, d))
    if pa_ or len(xa) != 1 or any(len(p) != 1 or len(x) != 1 for p, x in ((pb, yb), (pc, xc), (pd, yd))):
        raise ValueError(f"unexpected option shape: {[a, b, c, d]}")
    return {"X": xa[0], "Y": yb[0], "p": pb[0], "k": pc[0], "kp": pd[0], "X_C": xc[0], "Y_D": yd[0]}


def check_pair(v: dict) -> list[str]:
    """Violations of the common-ratio design (certainty_effect) on displayed numbers."""
    errs = []
    if not (v["Y"] > v["X"] > 0):
        errs.append("need Y > X > 0")
    if v["X_C"] != v["X"] or v["Y_D"] != v["Y"]:
        errs.append("prizes differ between parts")
    if not (Fraction(60, 100) <= v["p"] <= Fraction(95, 100)):
        errs.append("p outside [0.6, 0.95]")
    if not (Fraction(10, 100) <= v["k"] <= Fraction(50, 100)):
        errs.append("k outside [0.1, 0.5]")
    if v["kp"] != v["k"] * v["p"]:
        errs.append("k*p not exact")
    if abs(v["p"] * v["Y"] - v["X"]) < Fraction(1, 20) * v["X"]:
        errs.append("near EV tie")
    return errs


def parse_mirror(a: str, b: str, c: str, d: str) -> dict:
    """reflection_effect: displayed numbers of A (sure gain), B (risky gain), C (sure loss), D (risky loss)."""
    (pa_, xa), (pb, yb), (pc, xc), (pd, yd) = (_numbers(t) for t in (a, b, c, d))
    if pa_ or pc or len(xa) != 1 or len(xc) != 1 or any(len(p) != 1 or len(x) != 1 for p, x in ((pb, yb), (pd, yd))):
        raise ValueError(f"unexpected option shape: {[a, b, c, d]}")
    return {"X": xa[0], "Y": yb[0], "p": pb[0], "kp": pd[0], "X_C": xc[0], "Y_D": yd[0]}


def check_reflection(v: dict) -> list[str]:
    """Violations of the gain / loss mirror design (reflection_effect) on displayed numbers."""
    errs = []
    if not (v["Y"] > v["X"] > 0):
        errs.append("need Y > X > 0")
    if v["X_C"] != v["X"] or v["Y_D"] != v["Y"] or v["kp"] != v["p"]:
        errs.append("loss frame does not mirror the gain frame")
    if not (Fraction(20, 100) <= v["p"] <= Fraction(80, 100)):
        errs.append("p outside [0.2, 0.8]")
    if v["p"] * v["Y"] != v["X"]:
        errs.append("expected values not equal")
    return errs


def _favors(v: dict) -> str:
    return "ev_favors_lottery" if v["p"] * v["Y"] > v["X"] else "ev_favors_sure"


# element -> (parse, check, consistency rule id for the two_part type, extra tag from the displayed numbers)
RISK_PAIRS = {
    "certainty_effect": (parse_pair, check_pair, COMMON_RATIO, _favors),
    "reflection_effect": (parse_mirror, check_reflection, REFLECTION, lambda v: "equal_ev"),
}
CONSISTENCY_TYPES = {(e, "two_part"): spec[2] for e, spec in RISK_PAIRS.items()}
# Elements graded by consistency in every type. Their fillers put the options in a fixed template order:
# ambiguity_aversion: the known-probability bet first in both parts; independence: part 0 [target, other brand],
# part 1 [target, other brand, decoy]; endowment_effect: [Yes, No] in both parts; each part then ends with the
# "indifferent" option (October 2026). completeness / completeness_risk: one part per pair (a, b) of the n
# alternatives, in itertools.combinations order, options [prefer a, prefer b, indifferent, cannot decide].
CONSISTENCY_ELEMENTS = {"ambiguity_aversion": ELLSBERG, "independence": SAME_BRAND, "endowment_effect": MIRROR_TRADE,
                        "completeness": PAIRWISE_COMPLETE, "completeness_risk": PAIRWISE_COMPLETE}
PAIRWISE_ALTERNATIVES = {3: 3, 6: 4, 10: 5}  # parts -> alternatives


def consistency_rule(q: dict) -> str | None:
    """The consistency rule of a sampled question (released option indices), or None for answer-graded ones."""
    element, parts = q["element"], q["parts"]
    rule_id = CONSISTENCY_ELEMENTS.get(element) or CONSISTENCY_TYPES.get((element, str(q["metadata"]["type"]).strip()))
    if rule_id is None:
        return None
    if rule_id == PAIRWISE_COMPLETE:
        n = PAIRWISE_ALTERNATIVES[len(parts)]
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
        return pairwise_complete_rule([(a, b, *(p["order"].index(k) for k in range(4)))
                                       for (a, b), p in zip(pairs, parts)])
    first = [p["order"].index(0) for p in parts]  # where template option 0 is shown, per part
    second = [p["order"].index(1) for p in parts]
    # the "indifferent" option, when offered, is the last template option of each part (October 2026)
    strict = {ELLSBERG: (2, 2), SAME_BRAND: (2, 3), MIRROR_TRADE: (2, 2)}.get(rule_id)
    ind = None
    if strict and [len(p["order"]) for p in parts] == [strict[0] + 1, strict[1] + 1]:
        ind = tuple(p["order"].index(len(p["order"]) - 1) for p in parts)
    if rule_id == ELLSBERG:
        return ellsberg_rule(first[0], first[1], second[0], second[1], ind)
    if rule_id == SAME_BRAND:
        return same_brand_rule(first[0], first[1], second[0], second[1], ind)
    if rule_id == MIRROR_TRADE:
        return mirror_trade_rule(first[0], first[1], second[0], second[1], ind)
    # template option 0 is the sure side in both parts
    return same_side_rule(rule_id, first[0], first[1])


def _template_order(part: dict) -> list[str]:
    """Option texts in the template's order (option 0 = sure side for the risk pairs)."""
    out = [None] * len(part["options"])
    for shown, orig in enumerate(part["order"]):
        out[orig] = part["options"][shown]
    return out


def risk_pair_check(q: dict) -> tuple[str | None, list[str]]:
    """(extra tag, design violations) for certainty_effect / reflection_effect questions; (None, []) otherwise."""
    spec = RISK_PAIRS.get(q["element"])
    if spec is None:
        return None, []
    parse, check, _, tag_fn = spec
    qtype = q["metadata"]["type"]
    try:
        if qtype == "two_part":
            if len(q["parts"]) != 2 or any(len(p["options"]) != 2 for p in q["parts"]):
                return None, ["two_part: expected 2 parts x 2 options"]
            (a, b), (c, d) = (_template_order(p) for p in q["parts"])
            v = parse(a, b, c, d)
            return tag_fn(v), check(v)
        if qtype == "stated_prior":
            part = q["parts"][0]
            quoted = QUOTED_RE.findall(part["question"])
            c, d = _template_order(part)
            if len(q["parts"]) != 1 or len(quoted) != 3 or quoted[2] not in quoted[:2]:
                return None, ["stated_prior: unexpected shape"]
            a, b, chosen = quoted
            v = parse(a, b, c, d)
            errs = check(v)
            stated_sure = chosen == a
            template_mask = [part["correct"][part["order"].index(i)] for i in range(2)]
            if template_mask != [stated_sure, not stated_sure]:
                errs.append("template answer does not match the stated side")
            return tag_fn(v), errs
    except ValueError as e:
        return None, [str(e)]
    return None, []


# --------------------------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------------------------
def build_rows(q: dict, generator: str | None = None, manifest: dict | None = None) -> list[dict]:
    """Dataset rows (autosteer.schemas.DatasetRow fields, in order) for one sample_question() result.

    `difficulty` is the relative level within the element (autosteer.sampler.difficulty; 1 = its easiest), not the
    sampler's raw level; `manifest` (default: the loaded one) gives the element's levels."""
    from autosteer.sampler.difficulty import to_relative

    element, md, parts = q["element"], q["metadata"], q["parts"]
    difficulty = to_relative(element, md["difficulty_level"], manifest)
    info = element_info(element)
    base_id = content_key(element, parts)[:16]
    perspective, tags = split_tags(md["tags"])
    qtype = str(md["type"]).strip() or "default"
    extra, _ = risk_pair_check(q)
    if extra:
        tags = tags + [extra]
    rule = consistency_rule(q)
    rows = []
    for i, p in enumerate(parts):
        consistency = rule is not None
        n_true = sum(p["correct"])
        rows.append({
            "element": element,
            "question_id": f"{element}/{base_id}/{i}",
            "base_id": base_id,
            "part": i,
            "template_id": q["template_id"],
            "question_text": p["question"],
            "options": list(p["options"]),
            "grading": "consistency" if consistency else "answer",
            "correct_index": None if consistency or n_true != 1 else p["correct"].index(True),
            "consistency_group": f"{element}/{base_id}" if consistency else None,
            "consistency_rule": rule,
            "type": qtype,
            "domain": md["domain"],
            "perspective": perspective,
            "tags": tags,
            "difficulty": difficulty,
            "setting": info["setting"],
            "module": info["module"],
            "generator": generator or generator_id(),
            "seed": q["seed"],
        })
    return rows


def row_problems(q: dict, rows: list[dict]) -> list[str]:
    """Why a sampled question must not be released (empty list = fine)."""
    probs = []
    _, design = risk_pair_check(q)
    probs += [f"design:{e}" for e in design]
    for p, r in zip(q["parts"], rows):
        texts = r["options"]
        if r["grading"] == "answer":
            n_true = sum(p["correct"])
            if n_true == 0:
                probs.append("no_correct_option")
            elif n_true > 1:
                probs.append("multiple_correct_options")
        if len(texts) < 2:
            probs.append("fewer_than_2_options")
        if len(set(texts)) != len(texts):
            probs.append("duplicate_option_text")
        if any(t.strip() in BAD_OPTION_TEXT for t in texts):
            probs.append("placeholder_option_text")
        if any(NUMPY_REPR_RE.search(t) for t in texts) or NUMPY_REPR_RE.search(r["question_text"]):
            probs.append("numpy_repr_in_text")
        if PLACEHOLDER_RE.search(r["question_text"]) or any(PLACEHOLDER_RE.search(t) for t in texts):
            probs.append("unfilled_placeholder")
        if not isinstance(r["difficulty"], int) or r["difficulty"] < 1:
            probs.append("difficulty_not_a_relative_level")
    return sorted(set(probs))
