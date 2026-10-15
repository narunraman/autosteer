"""The public site: the guide, the STEER and STEER-ME taxonomies, and element pages.

Text content lives in ``content/`` (element descriptions and the two taxonomies, whose
setting and module descriptions are the papers' own text) and in the ``guide_*.html`` templates.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from autosteer.sampler import ElementSampler, load_manifest
from autosteer.sampler.difficulty import public_entry
from autosteer.services.dataset_store import DatasetStore, get_dataset_store
from autosteer.web import public


CONTENT_DIR = Path(__file__).resolve().parent / "content"

ELEMENTS = json.loads((CONTENT_DIR / "elements.json").read_text(encoding="utf-8"))
ELEMENTS_BY_NAME = {e["element"]: e for e in ELEMENTS}
# Old element ids that now live under another element (e.g. two elements merged into one).
ALIASES = {a: e["element"] for e in ELEMENTS for a in e.get("aliases", [])}
STEER_TAXONOMY = json.loads((CONTENT_DIR / "steer_taxonomy.json").read_text(encoding="utf-8"))
# Questions per element in the latest local export (autosteer export); replaced when the datasets are published.
DATASET_COUNTS = json.loads((CONTENT_DIR / "dataset_counts.json").read_text(encoding="utf-8"))["elements"]
STEER_ME_TAXONOMY = json.loads((CONTENT_DIR / "steer_me_taxonomy.json").read_text(encoding="utf-8"))

SLUGS = {"STEER": "steer", "STEER-ME": "steer-me"}
VERSIONS = {slug: version for version, slug in SLUGS.items()}
PAPERS = {"STEER": "https://arxiv.org/abs/2402.09552", "STEER-ME": "https://arxiv.org/abs/2502.13119"}
HF_DATASETS = {"STEER": "narunraman/steer", "STEER-ME": "narunraman/steer_me"}

PERSPECTIVES = {
    "anon": "third person",
    "first_person": "first person",
    "second_person": "second person",
    "man": "third person, a man",
    "woman": "third person, a woman",
}
# The ways questions within an element vary: (label, plural, query parameter of /api/sample, manifest key).
FACETS = {
    "type": ("Type", "types", "type", "types"),
    "domain": ("Domain", "domains", "domain", "domains"),
    "perspective": ("Perspective", "perspectives", "tags", "tags"),
}
SCORING_LABELS = {
    "answer": "One correct answer",
    "consistency": "Consistency across related questions",
    "mixed": "Mixed: some types have one answer, some are scored for consistency",
}

# One hue per setting, after the papers.
STEER_LOOK = {
    1: ("Foundations", "red"),
    2: ("Single-agent decisions", "green"),
    3: ("Multi-agent decisions", "blue"),
    4: ("Decisions for others", "purple"),
}

STEER_ME_LOOK = {
    1: ("Foundations", "red"),
    2: ("Consumption decisions", "green"),
    3: ("Production decisions", "blue"),
    4: ("Multi-agent environments", "yellow"),
    5: ("Equilibria and externalities", "purple"),
}


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _first_sentence(text: str) -> str:
    match = re.search(r"(?<!e\.g)(?<!i\.e)(?<!c\.f)(?<!vs)(?<!etc)\.(?=\s+[A-Z])", text)
    return text[: match.end()] if match else text


def _described(node: dict) -> dict:
    """The paper's paragraphs for a setting or module, plus a one-sentence lead."""
    paragraphs = node.get("description") or []
    return {"paragraphs": paragraphs, "lead": _first_sentence(paragraphs[0]) if paragraphs else ""}


def _steer_elements(module: dict) -> list[dict]:
    elements = []
    for e in module["elements"]:
        described = ELEMENTS_BY_NAME.get(e.get("data_name") or "", {})
        held = not described or described.get("status") == "held" or not e.get("in_release")
        if e.get("covered_by") in ELEMENTS_BY_NAME:
            cover = ELEMENTS_BY_NAME[e["covered_by"]]
            elements.append({
                "element": None, "id": e["id"], "display_name": e["paper_name"],
                "summary": e.get("definition", ""), "status": "covered",
                "covered_by": e["covered_by"], "covered_name": cover["display_name"],
            })
            continue
        elements.append({
            "element": e.get("data_name") if described else None,
            "id": e["id"],
            "display_name": e["paper_name"],
            "summary": e.get("definition") or described.get("summary", ""),
            "status": "held" if held else "release-candidate",
        })
    return elements


@lru_cache(maxsize=None)
def tree(version: str) -> list[dict]:
    """Settings -> modules -> elements, in paper order."""
    taxonomy, look = (STEER_TAXONOMY, STEER_LOOK) if version == "STEER" else (STEER_ME_TAXONOMY, STEER_ME_LOOK)
    settings = []
    for s in taxonomy["settings"]:
        short, color = look[s["number"]]
        mine = [e for e in ELEMENTS if e["version"] == version and _norm(e.get("setting")) == _norm(s["name"])]
        modules, placed = [], set()
        for m in s["modules"]:
            if version == "STEER":
                elements = _steer_elements(m)
            else:
                elements = [dict(e, id=None) for e in mine if _norm(e.get("module")) == _norm(m["name"])]
                placed.update(e["element"] for e in elements)
            modules.append({"number": m["number"], "name": m["name"], "elements": elements, **_described(m)})
        rest = [dict(e, id=None) for e in mine if e["element"] not in placed] if version != "STEER" else []
        shared = [ELEMENTS_BY_NAME[x["element"]] for x in s.get("shared_elements", []) if x["element"] in ELEMENTS_BY_NAME]
        if shared:
            modules.append({
                "number": f"{s['number']}.s", "name": "Shared with STEER",
                "elements": [dict(e, id=None, shared=True) for e in shared],
                "paragraphs": ["These elements belong to STEER and are part of STEER-ME's Foundations as well. They are published once, in STEER."],
                "lead": "Elements that belong to STEER and are part of STEER-ME's Foundations as well.",
            })
        if rest:
            modules.append({
                "number": f"{s['number']}.x", "name": "Unplaced", "elements": rest,
                "paragraphs": ["Elements whose module is still being checked."], "lead": "",
            })
        settings.append({
            "number": s["number"], "name": s["name"], "short": short, "color": color,
            "summary": s.get("summary", ""), "modules": modules,
            "count": sum(len(m["elements"]) for m in modules), **_described(s),
        })
    return settings


def _locate(version: str, element_name: str) -> tuple[dict | None, dict | None]:
    for s in tree(version):
        for m in s["modules"]:
            if any(e["element"] == element_name for e in m["elements"]):
                return s, m
    return None, None


def _item(slug: str, label: str, href: str | None = None) -> dict:
    return {"slug": slug, "label": label, "href": href or f"/{slug}"}


GUIDE_NAV = [
    {"title": "Basics", "items": [_item("welcome", "Welcome", "/"), _item("tutorial", "Tutorial"), _item("questions", "Questions and templates")]},
    {"title": "Benchmarks", "items": [_item("benchmarks", "Overview"), _item("steer", "STEER"), _item("steer-me", "STEER-ME")]},
    {"title": "Scoring", "items": [_item("scoring", "Overview"), _item("formats", "Question formats"), _item("metrics", "Metrics"), _item("diagnostics", "Diagnostics")]},
    {"title": "Artifacts", "items": [_item("datasets", "Datasets"), _item("autosteer", "autoSTEER")]},
]
# Pages reached from the header or the sidebar foot rather than the guide navigation: (group, label).
EXTRA_PAGES = {"api": ("Reference", "Reference")}

SITE_DESCRIPTION = ("Benchmarks for the economic reasoning of language models: the STEER and STEER-ME taxonomies, "
                    "a question generator, and an API for fresh questions.")
# The <title> and meta description of each guide page (the home page uses the site's own).
GUIDE_META = {
    "tutorial": ("Tutorial", "How to explore the STEER benchmarks and evaluate a model on them: steer-bench (Inspect) "
                             "to score it, slurm-llm to serve open models on a cluster, and QuickScope to find where it fails."),
    "questions": ("Questions and templates", "What a STEER question is made of: elements, types, domains and perspectives, "
                                           "reviewed templates, and values filled in with answers computed by code."),
    "benchmarks": ("Benchmarks", "STEER and STEER-ME, two benchmarks for economic reasoning: strategic and "
                                 "behavioral decisions, and non-strategic microeconomics."),
    "steer": ("STEER", "STEER: the elements of economic rationality, from foundations to decisions in games and "
                       "on behalf of a group."),
    "steer-me": ("STEER-ME", "STEER-ME: the microeconomics of price-taking consumers and firms, markets and "
                             "welfare."),
    "scoring": ("Scoring", "How answers to STEER questions are scored: question formats, metrics and diagnostics."),
    "formats": ("Question formats", "The ways a STEER question can be put to a model: multiple choice, free text, "
                                    "and variants of multiple choice."),
    "metrics": ("Metrics", "Exact-match and normalized accuracy, and the calibration measures used to score "
                           "STEER."),
    "diagnostics": ("Diagnostics", "Finding where a model is weak: slicing, worst-case analysis and answering from "
                                   "first principles."),
    "datasets": ("Datasets", "The fixed STEER and STEER-ME datasets, for results that are comparable across "
                             "papers."),
    "autosteer": ("autoSTEER", "autoSTEER, the tool used to build STEER-ME, for writing new elements on your own "
                               "machine."),
    "api": ("Reference", "The STEER API: generated questions over HTTP, in the same form as the dataset."),
}

# The Evals section has its own navigation.


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

# Scored results for the Evals pages: written by `steer-scoring site-export` (desktop
# github.com/narunraman/steer-scoring) from the STEER-ME paper runs; canonical runs only, element ids already mapped
# to the current ones. The browser fetches the same file; this side reads the models and dimensions.
EVAL_RESULTS_PATH = Path(__file__).resolve().parent / "static" / "evals_results.json"
# Question formats offered in the dropdown, and the multiple-choice variants in the breakdown.
EVAL_FORMATS = [("mc", "Multiple choice"), ("free", "Free text")]
EVAL_VARIANTS = [("nocot", "No CoT"), ("shown", "Options shown"), ("hidden", "Options hidden"), ("nota", "NOTA")]
# (key, short label, full name, which way is better)
EVAL_METRICS = [("norm", "Norm. acc.", "Normalized accuracy", "higher"), ("exact", "Exact", "Exact match", "higher"),
                ("ece", "ECE", "Expected calibration error", "lower"), ("brier", "Brier", "Brier score", "lower"),
                ("epa", "EPA", "Expected probability assignment", "higher")]


@lru_cache(maxsize=None)
def eval_results() -> dict:
    return json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))


def eval_difficulty_keys(results: dict, manifest: dict) -> dict:
    """{benchmark: {element: {relative level: results key}}} for the Evals difficulty breakdown.

    The scored runs key their difficulty breakdown by the raw template level of the paper run; the site shows the
    relative level within the element (autosteer.sampler.difficulty), so levels 1, 2, ... of a module or setting
    gather each element's easiest, next, ... level. An element is left out (no difficulty breakdown) when its run's
    levels do not map one to one onto its current levels, except that an element with a single level both then and
    now has level 1."""
    from autosteer.sampler.difficulty import relative_scale
    from autosteer.sampler.difficulty import public_difficulty

    out: dict = {}
    for bench, elements in results.get("benchmarks", {}).items():
        for element, models in elements.items():
            keys = sorted({k for e in models.values() for k in ((e or {}).get("by") or {}).get("difficulty", {})}, key=int)
            entry = manifest.get("elements", {}).get(element)
            if not keys or not entry:
                continue
            scale = relative_scale(entry.get("difficulty_levels", []))
            if len(keys) == 1 and len(scale) == 1:
                out.setdefault(bench, {})[element] = {"1": keys[0]}
                continue
            rel: dict | None = {}
            for k in keys:
                r = scale.get(public_difficulty(element, k))
                if r is None or str(r) in rel:
                    rel = None
                    break
                rel[str(r)] = k
            if rel:
                out.setdefault(bench, {})[element] = rel
    return out


def _scored_on(results: dict) -> str:
    built = (results.get("provenance", {}).get("built_at") or "")[:10]
    try:
        y, m, d = (int(x) for x in built.split("-"))
    except ValueError:
        return built
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
    return f"{d} {months[m - 1]} {y}"


# An open model appears on the Evals pages for a benchmark once its release run has multiple-choice
# scores on at least this share of the benchmark's elements; until then it is held back (left out of the
# published results by scripts/publish_evals.py). Closed models appear however limited their runs.
EVAL_LISTED_SHARE = 0.8


# Short display names for the Evals tables; the full name stays in the breakdown. Models not listed here
# fall back to the name without its instruction-tuning suffix.
EVAL_SHORT_NAMES = {
    "gemma-3-4b-it": "Gemma 3 4B", "Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B", "Llama-3.3-70B-Instruct": "Llama 3.3 70B",
    "aya-expanse-8b": "Aya Expanse 8B", "c4ai-command-r7b": "Command R7B", "tiny-aya-global": "Tiny Aya Global",
    "Qwen3.5-122B-A10B": "Qwen3.5 122B-A10B", "Qwen3.6-35B-A3B": "Qwen3.6 35B-A3B", "Qwen3.8-27B": "Qwen3.8 27B",
    "deepseek-llm-7b-chat": "DeepSeek LLM 7B", "Falcon3-7B-Instruct": "Falcon3 7B", "granite-3.1-8b-instruct": "Granite 3.1 8B",
    "OLMo-2-1124-7B-Instruct": "OLMo 2 7B", "Phi-3.5-mini-instruct": "Phi-3.5 mini", "Phi-4-mini-instruct": "Phi-4 mini",
    "Qwen3-4B-Instruct-2507": "Qwen3 4B 2507", "Yi-1.5-9B-Chat": "Yi-1.5 9B",
    # held back for now
    "gemma-3-12b-it": "Gemma 3 12B", "gemma-3-27b-it": "Gemma 3 27B", "Meta-Llama-3-8B-Instruct": "Llama 3 8B",
    "Meta-Llama-3-70B-Instruct": "Llama 3 70B", "Meta-Llama-3.1-70B-Instruct": "Llama 3.1 70B",
    "DeepSeek-R1-Distill-Qwen-7B": "R1-Distill-Qwen 7B", "DeepSeek-R1-Distill-Qwen-32B": "R1-Distill-Qwen 32B",
    "DeepSeek-R1-Distill-Llama-70B": "R1-Distill-Llama 70B", "Phi-4-mini-reasoning": "Phi-4 mini reasoning",
    "Phi-4-reasoning": "Phi-4 reasoning", "Phi-4-reasoning-plus": "Phi-4 reasoning plus", "QwQ-32B": "QwQ 32B",
}


def eval_short_name(name: str) -> str:
    if name in EVAL_SHORT_NAMES:
        return EVAL_SHORT_NAMES[name]
    short = re.sub(r"-(Instruct|instruct|Chat|chat|it)$", "", name)
    return re.sub(r"^(Qwen[\d.]+)-(\d)", r"\1 \2", short)   # Qwen2.5-7B -> Qwen2.5 7B


def eval_element_ids(version: str) -> list[str]:
    """The elements the Evals pages show for a benchmark, in taxonomy order."""
    return [e["element"] for s in tree(version) for m in s["modules"] for e in m["elements"] if e["element"]]


def eval_listed(results: dict, model_index: int, version: str) -> bool:
    """Whether a model's results for a benchmark are shown (see EVAL_LISTED_SHARE)."""
    m = results["models"][model_index]
    scored = results["benchmarks"].get(version, {})
    ids = eval_element_ids(version)
    k = sum(1 for e in ids if "mc" in scored.get(e, {}).get(str(model_index), {}))
    if not m.get("open"):
        return k > 0
    release = str((m.get("data") or {}).get(version, "")).startswith("release")
    return release and bool(ids) and k >= EVAL_LISTED_SHARE * len(ids)


def publishable_eval_results(results: dict) -> dict:
    """``results`` with every held-back (model, benchmark) pair removed, and models with nothing left dropped."""
    keep = {v: [i for i in range(len(results["models"])) if eval_listed(results, i, v)] for v in SLUGS}
    shown = sorted({i for v in keep.values() for i in v})
    new = {old: str(n) for n, old in enumerate(shown)}
    out = dict(results)
    out["models"] = []
    for i in shown:
        m = dict(results["models"][i])
        m["data"] = {v: d for v, d in (m.get("data") or {}).items() if i in keep.get(v, [])}
        if isinstance(m.get("lp"), dict):
            m["lp"] = {v: x for v, x in m["lp"].items() if i in keep.get(v, [])}
        out["models"].append(m)
    out["benchmarks"] = {v: {el: {new[int(k)]: c for k, c in cells.items() if k.isdigit() and int(k) in keep.get(v, [])}
                             for el, cells in els.items()}
                         for v, els in results["benchmarks"].items()}
    out["held_back"] = {"rule": f"open models with release mc on >= {EVAL_LISTED_SHARE:.0%} of a benchmark's elements; closed models with any",
                        "models_in": len(results["models"]), "models_out": len(shown)}
    return out


def evals_data() -> dict:
    """Everything the Evals pages' script needs: the taxonomies, the models and the slicing dimensions.

    The scores themselves are fetched by the script from ``static/evals_results.json``.
    """
    results = eval_results()
    benchmarks = []
    for version, vslug in SLUGS.items():
        scored = results["benchmarks"].get(version, {})
        settings = []
        for s in tree(version):
            modules = [{"number": m["number"], "name": m["name"],
                        "elements": [{"id": e["element"], "name": e["display_name"]} for e in m["elements"] if e["element"]]}
                       for m in s["modules"]]
            modules = [m for m in modules if m["elements"]]
            if modules:
                settings.append({"number": s["number"], "name": s["name"], "short": s["short"], "color": s["color"], "modules": modules})
        benchmarks.append({"key": version, "slug": vslug, "settings": settings, "scored": bool(scored)})
    dims = results.get("dims", {})
    difficulty_keys = eval_difficulty_keys(results, load_manifest())
    top = max((int(r) for els in difficulty_keys.values() for m in els.values() for r in m), default=0)
    years = sorted({d[:4] for m in results["models"] for d in m["run_dates"] if d})
    return {
        "benchmarks": benchmarks,
        "models": [{**{k: m.get(k) for k in ("name", "model", "provider", "open", "family", "size", "reasoning", "logprobs", "run_dates",
                                             "data", "via", "lp")}, "short": eval_short_name(m["name"]), "listed": {v: eval_listed(results, i, v) for v in SLUGS}}
                   for i, m in enumerate(results["models"])],
        "formats": [{"key": k, "label": l, "scored": k in results["formats"]} for k, l in EVAL_FORMATS],
        "variants": [{"key": k, "label": l} for k, l in EVAL_VARIANTS],
        "metrics": [{"key": k, "short": s, "label": l, "better": b} for k, s, l, b in EVAL_METRICS],
        "domains": [{"key": d, "label": _facet_label("domain", d)} for d in dims.get("domain", [])],
        "perspectives": [{"key": k, "label": PERSPECTIVES.get(k, k)} for k in dims.get("perspective", [])],
        "difficulties": [str(i) for i in range(1, top + 1)],
        "difficulty_keys": difficulty_keys,
        "note": {"scored_on": _scored_on(results), "years": f"{years[0]}\u2013{years[-1]}" if len(years) > 1 else "".join(years),
                 "n_models": len(results["models"])},
    }


def _search_items() -> list[dict]:
    """What the header search can find: guide pages and released elements, with a kind and a sub-label for the
    dropdown, and extra keys (element id, paper name, earlier ids) that match but are not shown."""
    items = [{"label": i["label"], "href": i["href"], "kind": g["title"]} for g in GUIDE_NAV for i in g["items"]]
    items += [{"label": "Reference", "href": "/reference", "kind": "API"}, {"label": "steer-bench", "href": "/tutorial#inspect", "kind": "Tutorial"},
              {"label": "QuickScope", "href": "/tutorial#quickscope", "kind": "Tutorial"}, {"label": "Cite", "href": "/#cite", "kind": "Welcome"}]
    for version in ("STEER", "STEER-ME"):
        for st in tree(version):
            for m in st["modules"]:
                for e in m["elements"]:
                    d = ELEMENTS_BY_NAME.get(e.get("element") or "")
                    if not d or d.get("status") == "held" or e.get("status") == "covered":
                        continue
                    keys = " ".join([d["element"].replace("_", " "), d.get("paper_name") or "", *d.get("aliases", [])])
                    items.append({"label": e["display_name"], "href": f"/elements/{d['element']}", "kind": version,
                                  "sub": m["name"], "keys": keys})
    seen, out = set(), []
    for it in items:
        if it["href"] not in seen:
            seen.add(it["href"]); out.append(it)
    return out


def _facet_label(kind: str, code: str) -> str:
    if kind == "perspective":
        return PERSPECTIVES[code]
    return str(code).replace("_", " ")


def _facet_values(kind: str, entry: dict) -> list[str]:
    values = [v for v in entry.get(FACETS[kind][3], []) if str(v).strip()]
    return [v for v in values if v in PERSPECTIVES] if kind == "perspective" else values


EXAMPLE_SEED = 1  # every element page shows the question this seed generates, so the page is stable
_EXAMPLES: dict = {}


def example_template(element_name: str) -> list[list[dict]]:
    """The template behind the element page's example, one list of text / {slot} pieces per part. The page shows
    it first and fills it when the reader clicks Generate. Empty if the element cannot be sampled. Cached per
    manifest build."""
    manifest = load_manifest()
    key = (manifest.get("generated_at"), element_name)
    if key not in _EXAMPLES:
        try:
            sampler = _EXAMPLES.get("sampler")
            if sampler is None or _EXAMPLES.get("built") != manifest.get("generated_at"):
                sampler = ElementSampler(manifest=manifest)
                _EXAMPLES.update(sampler=sampler, built=manifest.get("generated_at"))
            q = sampler.sample_question(element_name, EXAMPLE_SEED)
            rows = sampler._load_template(element_name).questions
            qid = int(str(q["template_id"]).rsplit("/", 1)[1])
            texts = [str(t) for t in rows[rows["question_id"] == qid].sort_values("sub_id")["question_text"]]
            _EXAMPLES[key] = [[{"slot": p[1:-1]} if p.startswith("{") and p.endswith("}") else {"text": p}
                               for p in re.split(r"(\{[^{}]+\})", text) if p] for text in texts]
        except Exception:
            _EXAMPLES[key] = []
    return _EXAMPLES[key]


def sitemap_paths(with_evals: bool = False) -> list[str]:
    """Every page a search engine should know about: the guide, the taxonomies, and the element pages."""
    paths = ["/"] + [i["href"] for g in GUIDE_NAV for i in g["items"] if i["href"] != "/"] + ["/reference"]
    for version, vslug in SLUGS.items():
        for s in tree(version):
            paths.append(f"/{vslug}/setting/{s['number']}")
            paths += [f"/{vslug}/module/{m['number']}" for m in s["modules"]]
    paths += [f"/elements/{e['element']}" for e in ELEMENTS if e.get("status") != "held"]
    if with_evals:
        paths.append("/evals")
    return list(dict.fromkeys(paths))


def install(app: FastAPI, templates: Jinja2Templates, public_mode: Callable[[], bool]) -> None:
    """Register the site's pages on ``app``."""
    static = Path(__file__).resolve().parent / "static"
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(static.glob("*")) if p.is_file())).hexdigest()[:10]
    templates.env.globals.update(guide_nav=GUIDE_NAV, search_items=_search_items(), public_mode=public_mode,
                                 asset_version=digest, site_description=SITE_DESCRIPTION)
    public.install(app, templates, public_mode, sitemap_paths=lambda: sitemap_paths(public.evals_enabled(public_mode())))

    def evals_shown() -> None:
        if not public.evals_enabled(public_mode()):
            raise HTTPException(status_code=404, detail="Not Found")

    def guide_page(request: Request, slug: str) -> HTMLResponse:
        found = [(g["title"], i["label"]) for g in GUIDE_NAV for i in g["items"] if i["slug"] == slug]
        if not found and slug in EXTRA_PAGES:
            found = [EXTRA_PAGES[slug]]
        if not found:
            raise HTTPException(status_code=404, detail="No such page")
        context = {"page": slug, "group_title": found[0][0], "page_label": found[0][1]}
        if slug in GUIDE_META:
            context["page_title"], context["page_description"] = GUIDE_META[slug]
        if slug in VERSIONS:
            version = VERSIONS[slug]
            context.update({"version": version, "vslug": slug, "taxonomy": tree(version), "paper": PAPERS[version], "hf": HF_DATASETS[version]})
        return templates.TemplateResponse(request, f"guide_{slug}.html", context)

    def browse_context(vslug: str, current: str = "") -> dict:
        if vslug not in VERSIONS:
            raise HTTPException(status_code=404, detail="No such benchmark")
        version = VERSIONS[vslug]
        return {"page": "", "version": version, "vslug": vslug, "tree": tree(version), "current": current}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(request: Request) -> HTMLResponse:
        return guide_page(request, "welcome")

    @app.get("/evals", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/evals/{path:path}", response_class=HTMLResponse, include_in_schema=False)
    def evals_page(request: Request, path: str = "") -> HTMLResponse:
        """One page per level: both benchmarks, a setting, a module, or an element."""
        evals_shown()
        data = evals_data()
        parts = [p for p in path.split("/") if p]
        crumbs = [{"label": "Evals", "href": "/evals"}]
        where = {"level": "root"}
        title = "Results"
        if parts:
            bench = next((b for b in data["benchmarks"] if b["slug"] == parts[0]), None)
            if bench is None or len(parts) == 1:
                raise HTTPException(status_code=404, detail="No such page")
            setting = next((s for s in bench["settings"] if _slugify(s["name"]) == parts[1]), None)
            if setting is None:
                raise HTTPException(status_code=404, detail="No such setting")
            base = f"/evals/{bench['slug']}/{parts[1]}"
            crumbs.append({"label": f"{bench['key']} · {setting['short']}", "href": base})
            where = {"level": "setting", "bench": bench["key"], "setting": setting["number"], "base": base}
            title = setting["name"]
            if len(parts) > 2:
                module = next((m for m in setting["modules"] if _slugify(m["name"]) == parts[2]), None)
                if module is None:
                    raise HTTPException(status_code=404, detail="No such module")
                base += "/" + parts[2]
                crumbs.append({"label": module["name"], "href": base})
                where.update(level="module", module=module["number"], base=base)
                title = module["name"]
                if len(parts) > 3:
                    return RedirectResponse(url=base, status_code=303)
        crumbs[-1]["href"] = None
        return templates.TemplateResponse(request, "evals.html", {
            "page": "", "section": "evals", "crumbs": crumbs, "title": title, "page_title": "Evals" if title == "Results" else f"Evals: {title}",
            "where": where, "evals_data": data,
        })

    # Guide pages live at the top level: /tutorial, /steer, /reference, ...
    def _guide_route(slug: str):
        def handler(request: Request) -> HTMLResponse:
            return guide_page(request, slug)
        return handler

    GUIDE_PATHS = {i["slug"]: i["href"] for g in GUIDE_NAV for i in g["items"]} | {"api": "/reference"}
    for slug, path in GUIDE_PATHS.items():
        if path != "/":
            app.add_api_route(path, _guide_route(slug), methods=["GET"], response_class=HTMLResponse, include_in_schema=False)

    @app.get("/{vslug}/setting/{number}", response_class=HTMLResponse, include_in_schema=False)
    def setting_page(request: Request, vslug: str, number: int) -> HTMLResponse:
        context = browse_context(vslug, f"s{number}")
        setting = next((s for s in context["tree"] if s["number"] == number), None)
        if setting is None:
            raise HTTPException(status_code=404, detail="No such setting")
        context.update(setting=setting, page_title=setting["name"],
                       page_description=setting["lead"] or setting["summary"])
        return templates.TemplateResponse(request, "setting.html", context)

    @app.get("/{vslug}/module/{number}", response_class=HTMLResponse, include_in_schema=False)
    def module_page(request: Request, vslug: str, number: str) -> HTMLResponse:
        context = browse_context(vslug, f"m{number}")
        for s in context["tree"]:
            for m in s["modules"]:
                if m["number"] == number:
                    context.update({"setting": s, "module": m, "page_title": m["name"],
                                    "page_description": m["lead"]})
                    return templates.TemplateResponse(request, "module.html", context)
        raise HTTPException(status_code=404, detail="No such module")

    @app.get("/elements/{element_name}", response_class=HTMLResponse, include_in_schema=False)
    def element_page(request: Request, element_name: str, store: DatasetStore = Depends(get_dataset_store)):
        if element_name in ALIASES:
            return RedirectResponse(url=f"/elements/{ALIASES[element_name]}", status_code=301)
        described = ELEMENTS_BY_NAME.get(element_name)
        if described is None:
            raise HTTPException(status_code=404, detail="No such element")
        if described.get("status") == "held" and public_mode():
            # superseded elements, kept for developer mode; their notes are internal
            raise HTTPException(status_code=404, detail="No such element")
        version = described["version"]
        context = browse_context(SLUGS[version], element_name)
        setting, module = _locate(version, element_name)
        entry = load_manifest().get("elements", {}).get(element_name, {})
        held = described.get("status") == "held"
        can_generate = bool(entry.get("has_filler")) and not held
        axes = [
            {"kind": kind, "label": label, "plural": plural, "param": param,
             "values": [(code, _facet_label(kind, code)) for code in _facet_values(kind, entry)]}
            for kind, (label, plural, param, _) in FACETS.items()
        ]
        # relative levels (1 = the element's easiest), as in the dataset and the API filter
        difficulties = (public_entry(element_name, entry) or {}).get("difficulty_levels", [])
        axes.append({"kind": None, "label": "Difficulty", "plural": "difficulty levels", "param": "difficulty_level",
                     "values": [(code, f"level {code}") for code in difficulties]})
        varying = [axis for axis in axes if len(axis["values"]) > 1]
        cells = max(1, len(axes[0]["values"])) * max(1, len(axes[1]["values"])) * max(1, len(difficulties))
        in_tree = next((e for e in (module["elements"] if module else []) if e["element"] == element_name), None)
        context.update({
            "element": element_name, "d": described, "te": in_tree, "setting": setting, "module": module,
            "page_title": in_tree["display_name"] if in_tree else described["display_name"],
            "page_description": (in_tree.get("summary") if in_tree else "") or described.get("summary", ""),
            "held": held, "can_generate": can_generate,
            "template": example_template(element_name) if can_generate else [],
            "axes": varying,
            "variation_count": " × ".join(f"{len(axis['values'])} {axis['plural']}" for axis in varying),
            "dataset_n": DATASET_COUNTS.get(element_name, {}).get("questions"),
            "hf": HF_DATASETS[version],
            "scoring": SCORING_LABELS.get(described.get("grading"), "To be documented"),
        })
        return templates.TemplateResponse(request, "element.html", context)

    @app.get("/by/{kind}/{value}", response_class=HTMLResponse, include_in_schema=False)
    def facet_page(request: Request, kind: str, value: str) -> HTMLResponse:
        """Every element that has questions of one type, in one domain, or from one perspective."""
        if kind not in FACETS:
            raise HTTPException(status_code=404, detail="No such grouping")
        entries = load_manifest().get("elements", {})
        having = {name for name, entry in entries.items() if value in _facet_values(kind, entry)}
        if not having:
            raise HTTPException(status_code=404, detail=f"No elements with this {kind}")
        groups = []
        for version, vslug in SLUGS.items():
            rows = [
                {"element": e, "color": s["color"], "module": m["name"]}
                for s in tree(version) for m in s["modules"] for e in m["elements"] if e["element"] in having
            ]
            if rows:
                groups.append({"version": version, "vslug": vslug, "rows": rows})
        others = sorted({v for entry in entries.values() for v in _facet_values(kind, entry)} - {value}) if kind != "type" else []
        return templates.TemplateResponse(request, "facet.html", {
            "page": "", "group_title": "Elements by " + FACETS[kind][0].lower(), "page_label": _facet_label(kind, value),
            "page_title": f"{FACETS[kind][0]}: {_facet_label(kind, value)}",
            "kind": kind, "kind_label": FACETS[kind][0], "value_label": _facet_label(kind, value), "groups": groups,
            "count": sum(len(g["rows"]) for g in groups),
            "others": [(v, _facet_label(kind, v)) for v in others],
        })


