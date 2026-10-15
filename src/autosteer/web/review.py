"""Template review (developer mode only): read a fixed sample of an element's templates, filled, and mark each
fine or a problem.

The release rule says an element ships only after someone has read its sample; the review record
lives in the data directory (``<data dir>/review/<element>.json``), next to the templates it describes.
"""
from __future__ import annotations

import json
import math
import random
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from autosteer import paths
from autosteer.sampler import ElementSampler, load_manifest


REVIEW_DIR: Path | None = None   # set to keep review records somewhere else (tests); default: review/ in the data directory


def _review_dir() -> Path:
    return REVIEW_DIR or paths.review_dir()
_LOCK = threading.Lock()
_SAMPLER: dict = {}
_ROWS: dict = {}  # (element, manifest) -> its template rows: building them is slow for the big elements


def _sampler() -> ElementSampler:
    manifest = load_manifest()
    key = (manifest.get("generated_at"), tuple(sorted(manifest.get("elements", {}))))
    with _LOCK:
        if _SAMPLER.get("key") != key:
            _SAMPLER.update(key=key, sampler=ElementSampler(manifest=manifest))
        return _SAMPLER["sampler"]


def _record(element: str) -> dict:
    path = _review_dir() / f"{element}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _templates(element: str) -> list[dict]:
    """The element's templates as the sampler uses them (after load-time fixes, unfillable ones dropped)."""
    from autosteer.sampler.difficulty import relative_scale
    from autosteer.sampler.difficulty import public_difficulty

    key = (element, load_manifest().get("generated_at"))
    with _LOCK:
        cached = _ROWS.get(key)
    if cached is None:
        dataset = _sampler()._load_template(element)
        q = dataset.questions.sort_values(["question_id", "sub_id"], kind="stable")
        # the level shown is the relative one (1 = the element's easiest), as on the site; the raw one is in the tooltip
        scale = relative_scale(public_difficulty(element, x) for x in q["difficulty_level"].unique())
        grouped = q.groupby("question_id", sort=True)
        parts = grouped["question_text"].agg(lambda t: [str(x) for x in t])
        first = grouped.first()
        cached = [{
            "id": int(qid),
            "template_id": f"{element}/questions/{int(qid)}",
            "parts": parts[qid],
            "type": str(f["type"]),
            "domain": str(f["domain"]),
            "tags": str(f["tags"] or "") if f["tags"] == f["tags"] else "",
            "difficulty": scale[public_difficulty(element, f["difficulty_level"])],
            "difficulty_raw": int(f["difficulty_level"]),
        } for qid, f in zip(first.index, first.to_dict("records"))]
        with _LOCK:
            if len(_ROWS) > 8:
                _ROWS.clear()
            _ROWS[key] = cached
    return [dict(r) for r in cached]


# One rule for every element, so a release can say what was read: a share of the element's templates, at least
# SAMPLE_MIN and at most SAMPLE_MAX (all of them when it has fewer), drawn and filled with one fixed seed, so the
# sample is the same on every opening of the page and on every machine.
REVIEW_SEED = "steer-review-1"
SAMPLE_SHARE, SAMPLE_MIN, SAMPLE_MAX = 0.02, 10, 20


def sample_size(n_templates: int) -> int:
    return min(n_templates, max(SAMPLE_MIN, min(SAMPLE_MAX, math.ceil(SAMPLE_SHARE * n_templates))))


def distinct_templates(rows: list[dict]) -> list[dict]:
    """One row per template text and level. Rows that repeat a text at a level differ only in what the filler is told
    (mult_div's positive / negative / both, the number of candidates in a vote): those are fill settings, not
    templates, and are kept on the row as ``variants`` (with ``fills``: the fields that vary and their values)."""
    groups: dict = {}
    for r in rows:
        key = (tuple(" ".join(p.split()) for p in r["parts"]), r["difficulty"])
        groups.setdefault(key, []).append(r)
    out = []
    for members in groups.values():
        first = members[0]
        fills = {f: sorted({m[f] for m in members}) for f in ("type", "domain", "tags") if len({m[f] for m in members}) > 1}
        out.append({**first, "variants": [{k: m[k] for k in ("template_id", "type", "domain", "tags")} for m in members],
                    "fills": fills})
    return out


def _allocate(sizes: dict, n: int) -> dict:
    """How many of n picks each cell gets: in proportion to its size (largest remainder), at least one per cell
    when n allows, never more than the cell has."""
    total = sum(sizes.values())
    n = min(n, total)
    quota = {k: v * n / total for k, v in sizes.items()}
    alloc = {k: min(sizes[k], int(q)) for k, q in quota.items()}
    if n >= len(sizes):
        alloc = {k: max(1, a) for k, a in alloc.items()}
    while sum(alloc.values()) > n:  # the "at least one" overshot: take from the most over-served cell
        k = max((k for k in alloc if alloc[k] > 1), key=lambda k: alloc[k] - quota[k])
        alloc[k] -= 1
    while sum(alloc.values()) < n:
        k = max((k for k in alloc if alloc[k] < sizes[k]), key=lambda k: quota[k] - alloc[k])
        alloc[k] += 1
    return alloc


def _draw(element: str) -> dict:
    """The picks of the review sample with the fill setting and seed of each, without filling them."""
    rows = distinct_templates(_templates(element))
    total = len(rows)
    if not rows:
        return {"picks": [], "total": 0, "n": 0}
    cells: dict = {}
    for r in rows:
        cells.setdefault((r["type"], r["difficulty"]), []).append(r)
    rng = random.Random(f"{REVIEW_SEED}:{element}")
    n = sample_size(total)
    alloc = _allocate({k: len(v) for k, v in cells.items()}, n)
    picks = [r for k in sorted(cells) for r in rng.sample(cells[k], alloc[k])]
    rng.shuffle(picks)
    out = []
    for r in picks:
        variant = rng.choice(r["variants"])
        out.append({**r, "key": r["template_id"], "fill_variant": variant, "fill_seed": rng.randrange(2**31)})
    return {"picks": out, "total": total, "n": n}


def review_sample(element: str) -> dict:
    """The element's review sample: sample_size(N) of its N distinct templates, spread over its (type, level) cells
    in proportion to their size, in a fixed order, each filled once with one of its fill settings. The seed is the
    same for every element (REVIEW_SEED, with the element's name), so the picks and the filled questions never change
    unless the templates do."""
    drawn = _draw(element)
    out = []
    for r in drawn["picks"]:
        variant, seed = r["fill_variant"], r["fill_seed"]
        try:
            filled, error = _sampler().sample_question(element, seed, template_id=variant["template_id"]), None
        except Exception as exc:  # a template the filler cannot fill is itself a finding
            filled, error = None, f"{type(exc).__name__}: {exc}"
        out.append({**r, "pieces": [_split_slots(p) for p in r["parts"]], "filled": filled, "error": error})
    return {**drawn, "picks": out}


def marks_of(rec: dict) -> dict:
    return rec.get("marks") or {}


def upper_bound(problems: int, n: int, confidence: float = 0.95) -> float | None:
    """With `problems` of n templates read at random found wrong, the one-sided upper confidence bound on the share of
    the element's templates that are wrong (Clopper-Pearson; with none wrong it is 1 - 0.05^(1/n))."""
    if not n or problems >= n:
        return None
    alpha = 1 - confidence

    def at_most(p):  # P(X <= problems) for X ~ Binomial(n, p)
        return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(problems + 1))

    lo, hi = problems / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if at_most(mid) > alpha else (lo, mid)
    return hi


def standing(marks: dict, keys: list[str]) -> dict:
    """How far the reading of the sample has got: marks on templates no longer in the sample do not count."""
    mine = {k: m for k, m in marks.items() if k in set(keys)}
    n, read = len(keys), len(mine)
    problems = sum(m.get("mark") == "problem" for m in mine.values())
    return {"n": n, "marked": read, "ok": read - problems, "problems": problems,
            "bound": upper_bound(problems, n) if n and read == n else None}


def reviewer_name() -> str:
    """Who is reading: the git user of this checkout, as a contributor's review is sent in a pull request."""
    try:
        return subprocess.run(["git", "config", "user.name"], capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return ""


def _write_record(element: str, rec: dict) -> None:
    _review_dir().mkdir(parents=True, exist_ok=True)
    (_review_dir() / f"{element}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _split_slots(text: str) -> list[dict]:
    return [{"slot": p[1:-1]} if p.startswith("{") and p.endswith("}") else {"text": p}
            for p in re.split(r"(\{[^{}]+\})", text) if p]


def template_rows(element: str) -> list[dict]:
    """The templates with their text split into plain text and {slot} pieces (used by steps 3 and 4)."""
    rows = _templates(element)
    for r in rows:
        r["pieces"] = [_split_slots(p) for p in r["parts"]]
    return rows


def install(app: FastAPI, templates: Jinja2Templates, public_mode: Callable[[], bool]) -> None:
    """The fill endpoint and saving a review; the pages are developer/e/{element}/templates and /review (devmode)."""
    def local_only() -> None:
        if public_mode():
            raise HTTPException(status_code=404, detail="Not found")

    @app.get("/developer/review/{element}/fill", include_in_schema=False)
    @app.get("/developer/e/{element}/fill", include_in_schema=False)
    def review_fill(element: str, template_id: str | None = None, seed: int = 0):
        local_only()
        try:
            return JSONResponse(_sampler().sample_question(element, seed, template_id=template_id))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/developer/review/{element}", include_in_schema=False)
    @app.post("/developer/e/{element}/review", include_in_schema=False)
    def review_save(element: str, notes: str = Form("")):
        """Save the notes on the element (the reading itself is the marks on the sample)."""
        local_only()
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", element):
            raise HTTPException(status_code=404, detail="No such element")
        rec = {**_record(element), "element": element, "notes": notes[:20000]}
        _write_record(element, rec)
        from autosteer.web import devmode

        token = devmode.flash({"success": "Notes saved."})
        return RedirectResponse(url=f"/developer/e/{element}/review?f={token}#saved", status_code=303)

    def _checked(element: str) -> None:
        local_only()
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", element):
            raise HTTPException(status_code=404, detail="No such element")

    @app.post("/developer/e/{element}/review/mark", include_in_schema=False)
    def review_mark(element: str, template_id: str = Form(...), mark: str = Form(...), note: str = Form("")):
        """Mark one sampled question: ok, problem, or clear."""
        _checked(element)
        rec = _record(element)
        marks = rec.setdefault("marks", {})
        if mark == "clear":
            marks.pop(template_id, None)
        elif mark in ("ok", "problem"):
            marks[template_id] = {"mark": mark, "note": note[:2000]}
        else:
            raise HTTPException(status_code=400, detail="mark is ok, problem or clear")
        rec["element"] = element
        try:
            keys = [p["key"] for p in _draw(element)["picks"]]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not draw the sample: {exc}") from exc
        rec["marks"] = marks = {k: marks[k] for k in keys if k in marks}  # marks from an earlier sample go
        st = standing(marks, keys)
        # the sample's standing, for the element's step and the dashboard; read in full, it says who read it and when
        prev = rec.get("sample") if isinstance(rec.get("sample"), dict) and rec["sample"].get("seed") == REVIEW_SEED else {}
        rec.pop("sample_n", None)
        rec["sample"] = {"seed": REVIEW_SEED, "n": st["n"], "read": st["marked"], "ok": st["ok"], "problems": st["problems"]}
        if st["marked"] == st["n"]:
            rec["sample"]["read_by"] = prev.get("read_by") or reviewer_name()
            rec["sample"]["read_at"] = prev.get("read_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_record(element, rec)
        return JSONResponse(st)
