"""Pages for agents and scripts: /llms.txt (what the site is and where things are, llmstxt.org) and the Evals
results as plain data: /evals.csv (one row per benchmark, element, model, format and slice, with named metric
columns) and /evals.json (the models, metrics and formats that the CSV's columns refer to).

The Evals pages fetch static/evals_results.json, a compact structure made for their script; these give the same
numbers in a form that needs no reading of that script. Both follow the Evals switch: without it they are 404s, and
llms.txt leaves them out.
"""
from __future__ import annotations

import csv
import io
import json
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response

from autosteer.sampler.manifest import load_manifest

from . import public, site

# the values of a results entry, in order: site.EVAL_METRICS, then n and the share left unanswered
METRIC_COLUMNS = {"norm": "normalized_accuracy", "exact": "accuracy", "ece": "ece", "brier": "brier", "epa": "epa"}
FORMAT_NOTES = {
    "mc": "multiple choice, the letter alone (no reasoning)",
    "free": "free text: the answer itself, no options (elements whose answer can be checked without options)",
    "shown": "multiple choice, reasoning first with the options shown",
    "hidden": "multiple choice, reasoning first with the options hidden, then the options",
    "nota": "multiple choice with a 'none of the above' option",
}
METRIC_NOTES = {
    "normalized_accuracy": "1 for a correct answer, -1/(k-1) for a wrong one with k options, so guessing scores 0",
    "accuracy": "share answered correctly",
    "ece": "expected calibration error of the option probabilities (10 bins); lower is better",
    "brier": "Brier score averaged over the options, (1/k) sum (p - correct)^2; lower is better",
    "epa": "expected probability assigned to the correct option",
}
COLUMNS = (["benchmark", "setting", "module", "element", "element_name", "model", "provider", "open", "reasoning",
            "size_b", "format", "slice", "slice_value"] + list(METRIC_COLUMNS.values()) + ["n", "unanswered"])


def _places(version: str) -> dict:
    """element -> (setting, module, display name) in a benchmark's taxonomy."""
    out = {}
    for s in site.tree(version):
        for m in s["modules"]:
            for e in m["elements"]:
                if e.get("element"):
                    out.setdefault(e["element"], (s["name"], m["name"], e["display_name"]))
    return out


def evals_rows(results: dict, manifest: dict) -> list[dict]:
    """The published results as rows (see COLUMNS). slice is 'all' or a breakdown of the multiple-choice score
    (domain, perspective, difficulty: the level within the element, 1 its easiest, as on the Evals pages)."""
    metrics = [k for k, *_ in site.EVAL_METRICS]
    difficulty = site.eval_difficulty_keys(results, manifest)
    rows = []
    for version, elements in results.get("benchmarks", {}).items():
        places = _places(version)
        for element, per_model in sorted(elements.items()):
            setting, module, name = places.get(element, ("", "", element))
            rel = {v: k for k, v in difficulty.get(version, {}).get(element, {}).items()}
            for k, entry in per_model.items():
                m = results["models"][int(k)]
                base = {"benchmark": version, "setting": setting, "module": module, "element": element,
                        "element_name": name, "model": m["name"], "provider": m.get("provider"), "open": m.get("open"),
                        "reasoning": m.get("reasoning"), "size_b": m.get("size")}

                def row(fmt, kind, value, vec):
                    r = dict(base, format=fmt, slice=kind, slice_value=value)
                    r.update({METRIC_COLUMNS[mk]: vec[i] for i, mk in enumerate(metrics)})
                    r["n"], r["unanswered"] = vec[len(metrics)], (vec[len(metrics) + 1] if len(vec) > len(metrics) + 1 else None)
                    return r

                for fmt, vec in entry.items():
                    if fmt in ("by", "nocot") or not isinstance(vec, list):
                        continue
                    rows.append(row(fmt, "all", "", vec))
                for kind, parts in (entry.get("by") or {}).items():
                    for value, vec in parts.items():
                        if kind == "difficulty":
                            if value not in rel:
                                continue
                            value = rel[value]
                        rows.append(row("mc", kind, value, vec))
    return rows


_CACHE: dict = {}


def _cached(key: str, build: Callable[[], bytes]) -> bytes:
    results = site.eval_results()
    if _CACHE.get(key, (None,))[0] is not results:
        _CACHE[key] = (results, build())
    return _CACHE[key][1]


def evals_csv() -> bytes:
    def build():
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in evals_rows(site.eval_results(), load_manifest()):
            w.writerow({k: "" if r.get(k) is None else r[k] for k in COLUMNS})
        return out.getvalue().encode("utf-8")
    return _cached("csv", build)


def evals_json() -> bytes:
    def build():
        results = site.eval_results()
        doc = {
            "about": ("Scores of language models on STEER and STEER-ME, as shown on " + public.site_url() + "/evals. "
                      "The rows are in /evals.csv; this file describes their columns."),
            "scored_on": site._scored_on(results),
            "csv": public.site_url() + "/evals.csv",
            "columns": {
                "slice": "'all', or a breakdown of the multiple-choice score: domain, perspective, or difficulty (the "
                         "level within the element, 1 its easiest; not comparable across elements)",
                "n": "questions scored", "unanswered": "share of questions without a valid answer",
                **METRIC_NOTES,
            },
            "calibration": ("ece, brier and epa use the probabilities the model put on the option letters, "
                            "renormalized over the valid letters; they are empty for runs without them (API models "
                            "that return no logprobs, and reasoning models, whose answer letter is not the first token)"),
            "formats": FORMAT_NOTES,
            "models": [{k: m.get(k) for k in ("name", "provider", "open", "family", "size", "reasoning", "logprobs",
                                              "run_dates")} | {"benchmarks": m.get("data")} for m in results["models"]],
        }
        return json.dumps(doc, indent=1, ensure_ascii=False).encode("utf-8")
    return _cached("json", build)


def llms_txt(with_evals: bool) -> str:
    base = public.site_url()
    lines = ["# STEER", "", f"> {site.SITE_DESCRIPTION}", "",
             "STEER and STEER-ME are benchmarks of economic reasoning, organized as a taxonomy: settings, modules, "
             "and elements (one skill each, such as finding a dominated strategy or computing consumer surplus). "
             "Questions are generated from reviewed templates by autoSTEER, so any number of fresh questions can be "
             "drawn; fixed samples are on Hugging Face.", ""]
    lines += ["## Guide", ""]
    for g in site.GUIDE_NAV:
        for i in g["items"]:
            desc = site.GUIDE_META.get(i["slug"], (None, ""))[1]
            lines.append(f"- [{i['label']}]({base}{i['href']})" + (f": {desc}" if desc else ""))
    lines += ["", "## API", "",
              f"- [Reference]({base}/reference): every endpoint and its parameters",
              f"- [Sample questions]({base}/api/sample?element_name=consumer_surplus&n=2&seed=1): "
              "`/api/sample?element_name=<element>&n=<count>&seed=<int>`, rows with the dataset's fields",
              f"- [Elements]({base}/api/elements): every element with its types, domains and difficulty levels", ""]
    lines += ["## Data", "",
              "- [STEER on Hugging Face](https://huggingface.co/datasets/narunraman/steer)",
              "- [STEER-ME on Hugging Face](https://huggingface.co/datasets/narunraman/steer_me)"]
    if with_evals:
        lines += [f"- [Evals results as CSV]({base}/evals.csv): one row per benchmark, element, model, format and slice",
                  f"- [Evals columns, models and formats]({base}/evals.json)"]
    lines += ["", "## Code", "",
              "- [steer-bench](https://github.com/narunraman/STEER): Inspect harness to evaluate a model",
              "- [autoSTEER](https://github.com/narunraman/autosteer): the question generator, this site and developer mode",
              "", "## Optional", ""]
    for version, vslug in site.SLUGS.items():
        for s in site.tree(version):
            lines.append(f"- [{version}: {s['name']}]({base}/{vslug}/setting/{s['number']})")
    lines.append(f"- Element pages: {base}/elements/<element>, listed in {base}/sitemap.xml")
    return "\n".join(lines) + "\n"


def install(app: FastAPI, public_mode: Callable[[], bool]) -> None:
    def evals_on() -> bool:
        return public.evals_enabled(public_mode())

    @app.get("/llms.txt", include_in_schema=False)
    def llms() -> PlainTextResponse:
        return PlainTextResponse(llms_txt(evals_on()), media_type="text/markdown; charset=utf-8")

    @app.get("/evals.csv", include_in_schema=False)
    def evals_csv_route() -> Response:
        if not evals_on():
            raise HTTPException(status_code=404, detail="Not Found")
        return Response(evals_csv(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'inline; filename="steer-evals.csv"'})

    @app.get("/evals.json", include_in_schema=False)
    def evals_json_route() -> Response:
        if not evals_on():
            raise HTTPException(status_code=404, detail="Not Found")
        return Response(evals_json(), media_type="application/json")
