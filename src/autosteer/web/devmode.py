"""Developer mode as an element workspace.

Every developer page belongs either to the developer home (Elements, New element, Setup) or to one element:

    /developer                         Elements
    /developer/new                     New element
    /developer/setup                   Setup: providers, default models, data directory
    /developer/e/{element}             Element overview
    /developer/e/{element}/seeds       1. Example seeds
    /developer/e/{element}/candidates  2. Seed candidates
    /developer/e/{element}/templates   3. Templates
    /developer/e/{element}/review      4. Review

The pages read the same data as before (example seeds, generated seeds and their review.json, the template
datasets, the manifest, the filler registry, the template review records). Elements created on the New element page
get a draft record in ``<data dir>/devmode/elements/<id>.json``; they are not listed on the public site.
Setup's default models are kept in ``<data dir>/devmode/settings.json``. API keys are never read into a page,
written to disk or logged: the pages only report whether the environment variables are set.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from autosteer import dataset_io
from autosteer.constants import DOMAINS, TAGS
from autosteer.filler_tools import code_templates, function_registry
from autosteer.sampler import load_manifest
from autosteer.services.dataset_store import DatasetStore, get_dataset_store
from autosteer.services.generated_seed_service import load_review_state
from autosteer.taxonomy import display_name_for_element
from autosteer.web import review, site

STEPS = [
    ("seeds", "1. Example seeds", "Example seeds"),
    ("candidates", "2. Seed candidates", "Seed candidates"),
    ("templates", "3. Templates", "Templates"),
    ("review", "4. Review", "Review"),
]
STEP_KEYS = [key for key, _, _ in STEPS]
STEP_LABELS = {key: label for key, label, _ in STEPS}
STEP_NAMES = {key: name for key, _, name in STEPS}
GROUPS = [("STEER", "STEER"), ("STEER-ME", "STEER-ME"), ("offsite", "Not on the site yet")]
PACKAGE_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parents[3] / "tests"

# ---------------------------------------------------------------------------------------------------------------
# Providers. Only the presence of each environment variable is ever reported.
# requires: groups of alternatives; every group needs one variable set.
# ---------------------------------------------------------------------------------------------------------------
PROVIDERS = [
    {"key": "openai", "label": "OpenAI", "client": "OPENAI", "requires": [["OPENAI_API_KEY"]], "optional": [],
     "models": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"], "note": ""},
    {"key": "azure", "label": "Azure OpenAI", "client": "AZURE",
     "requires": [["AZURE_OPENAI_API_KEY"], ["AZURE_OPENAI_ENDPOINT"]], "optional": [],
     "models": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"], "note": "Model names are your deployment names."},
    {"key": "anthropic", "label": "Anthropic", "client": "ANTHROPIC", "requires": [["ANTHROPIC_API_KEY"]], "optional": [],
     "models": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"], "note": ""},
    {"key": "gemini", "label": "Google Gemini", "client": "GEMINI", "requires": [["GEMINI_API_KEY", "GOOGLE_API_KEY"]],
     "optional": [], "models": ["gemini-2.5-pro", "gemini-2.5-flash"],
     "note": ""},
    {"key": "openai_compatible", "label": "OpenAI-compatible endpoint", "client": "OPENAI_COMPATIBLE",
     "requires": [["OPENAI_COMPATIBLE_BASE_URL"]], "optional": ["OPENAI_COMPATIBLE_API_KEY"],
     "models": [], "note": "vLLM, Together, Fireworks, a local server: any endpoint that speaks the OpenAI chat API."},
]
PROVIDERS_BY_KEY = {p["key"]: p for p in PROVIDERS}
# Variables a key can be entered for on the AI setup page (kept in this process's environment only).
SESSION_VARS = {n for p in PROVIDERS for group in p["requires"] for n in group} | {n for p in PROVIDERS for n in p["optional"]}
_SESSION_KEYS: set[str] = set()
LEGACY_PROVIDER = {"OpenAI": "openai", "Azure OpenAI": "azure"}


def _env_set(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def provider_rows() -> list[dict]:
    rows = []
    for p in PROVIDERS:
        groups = [[{"name": n, "set": _env_set(n), "session": n in _SESSION_KEYS} for n in group] for group in p["requires"]]
        configured = all(any(v["set"] for v in group) for group in groups)
        models = list(p["models"])
        if p["key"] == "openai_compatible":
            models = [m.strip() for m in os.getenv("OPENAI_COMPATIBLE_MODELS", "").split(",") if m.strip()]
        if p["key"] in ("openai", "azure"):
            models = list(dict.fromkeys([os.getenv("GENERATION_MODEL", "gpt-5.4"), *models]))
        rows.append({**p, "groups": groups, "configured": configured, "models": models,
                     "optional_vars": [{"name": n, "set": _env_set(n)} for n in p["optional"]]})
    return rows


def _settings_path(store: DatasetStore) -> Path:
    return Path(store.data_root) / "devmode" / "settings.json"


def load_settings(store: DatasetStore) -> dict:
    try:
        return json.loads(_settings_path(store).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(store: DatasetStore, settings: dict) -> None:
    path = _settings_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=1) + "\n", encoding="utf-8")


def model_choices(status: dict, store: DatasetStore, role: str) -> dict:
    """Model menu for a step form: options 'provider:model' from the configured providers, and the default."""
    providers = status.get("providers") or []
    configured = [p for p in providers if p["configured"]]
    shown = configured or providers
    settings = load_settings(store).get(role) or {}
    options = []
    for p in shown:
        models = list(p["models"])
        if settings.get("provider") == p["key"] and settings.get("model") and settings["model"] not in models:
            models.insert(0, settings["model"])
        options += [{"value": f"{p['key']}:{m}", "label": f"{p['label']} · {m}"} for m in models]
    default = ""
    if settings.get("provider") in {p["key"] for p in configured} and settings.get("model"):
        default = f"{settings['provider']}:{settings['model']}"
    elif configured and configured[0]["models"]:
        default = f"{configured[0]['key']}:{configured[0]['models'][0]}"
    if not options:  # status from an older caller (tests): plain model names
        options = [{"value": m, "label": m} for m in status.get("models", [])]
        default = status.get("default_model", "")
    # The page asks for a provider and types any model name; the known names are only suggestions.
    pickers = [{"key": p["key"], "label": p["label"], "suggest": list(p["models"])} for p in shown]
    return {"options": options, "default": default, "providers": pickers}


def resolve_model(ai_model: str, status: dict) -> tuple[str, str]:
    """(LanguageModelClient provider name, model) for a submitted 'provider:model' or plain model name."""
    ai_model = (ai_model or "").strip()
    key, _, model = ai_model.partition(":")
    if model and key in PROVIDERS_BY_KEY:
        return PROVIDERS_BY_KEY[key]["client"], model
    provider = PROVIDERS_BY_KEY.get(LEGACY_PROVIDER.get(status.get("provider") or "", "openai"), PROVIDERS[0])
    return provider["client"], ai_model or status.get("default_model", "")


# ---------------------------------------------------------------------------------------------------------------
# Drafts: elements created on the New element page.
# ---------------------------------------------------------------------------------------------------------------
def _drafts_dir(store: DatasetStore) -> Path:
    return Path(store.data_root) / "devmode" / "elements"


def load_drafts(store: DatasetStore) -> dict[str, dict]:
    drafts = {}
    root = _drafts_dir(store)
    if root.is_dir():
        for path in sorted(root.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            drafts[record.get("element", path.stem)] = record
    return drafts


def derive_id(display_name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", display_name.lower())).strip("_")


NEW_PLACE = "__new__"  # the menu value for "New setting…" / "New module…" on the New element page


def menu_taxonomy(store: DatasetStore) -> dict[str, list[dict]]:
    """Settings and modules offered on the New element page: the paper taxonomies, plus any new setting or module an
    earlier draft introduced (``custom: True``, no number), so several drafts can share one."""
    menus = {}
    for version in ("STEER", "STEER-ME"):
        menus[version] = [{"name": s["name"], "number": s["number"], "custom": False,
                           "modules": [{"name": m["name"], "number": m["number"], "custom": False} for m in s["modules"]
                                       if not str(m["number"]).endswith((".s", ".x"))]}
                          for s in site.tree(version)]
    for draft in load_drafts(store).values():
        settings = menus.get(draft.get("version"))
        if settings is None or not draft.get("setting") or not draft.get("module"):
            continue
        entry = next((x for x in settings if x["name"].lower() == draft["setting"].lower()), None)
        if entry is None:
            entry = {"name": draft["setting"], "number": "", "custom": True, "modules": []}
            settings.append(entry)
        if not any(m["name"].lower() == draft["module"].lower() for m in entry["modules"]):
            entry["modules"].append({"name": draft["module"], "number": "", "custom": True})
    return menus


def taken_ids(store: DatasetStore) -> set[str]:
    names = set(site.ELEMENTS_BY_NAME) | set(site.ALIASES) | set(load_drafts(store)) | set(store.list_elements())
    names |= set(code_templates.elements()) | set(code_templates.composites())
    names |= set(load_manifest().get("elements", {}))
    return names


# ---------------------------------------------------------------------------------------------------------------
# Counts per element, cached by file modification time.
# ---------------------------------------------------------------------------------------------------------------
_COUNT_CACHE: dict = {}


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _cached(key: tuple, paths: list[Path], compute: Callable):
    stamp = tuple(_mtime(p) for p in paths)
    hit = _COUNT_CACHE.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    value = compute()
    _COUNT_CACHE[key] = (stamp, value)
    return value


def seed_count(store: DatasetStore, element: str) -> int:
    pkl = dataset_io.saved_file(store.dataset_path("example_seeds", element))

    def compute():
        try:
            return int(len(store.load_dataset("example_seeds", element).example_questions))
        except Exception:
            return 0
    return _cached(("seeds", str(store.data_root), element), [pkl], compute) if pkl else 0


def candidate_counts(store: DatasetStore, element: str) -> dict:
    folder = store.dataset_path("generated_seeds", element)
    pkl = dataset_io.saved_file(folder)
    if not pkl:
        return {"total": 0, "accepted": 0, "rejected": 0, "open": 0}

    def compute():
        try:
            questions = store.load_dataset("generated_seeds", element).example_questions
            keys = [f"{int(q)}:{int(s)}" for q, s in zip(questions["question_id"], questions["sub_id"])]
        except Exception:
            keys = []
        state = load_review_state(store, element)
        statuses = [(state.get(k) or {}).get("status", "draft") for k in keys]
        return {"total": len(keys), "accepted": statuses.count("accepted"), "rejected": statuses.count("rejected"),
                "open": sum(1 for s in statuses if s in ("draft", "needs_edit"))}
    return _cached(("cands", str(store.data_root), element), [pkl, folder / "review.json"], compute)


def stored_template_count(store: DatasetStore, element: str) -> int | None:
    pkl = dataset_io.saved_file(store.dataset_path("templates", element))
    if not pkl:
        return None

    def compute():
        try:
            return int(store.load_dataset("templates", element).questions["question_id"].nunique())
        except Exception:
            return None
    return _cached(("tmpl", str(store.data_root), element), [pkl], compute)


_TEST_FILES: dict = {}
GENERIC_TESTS = {"test_all_elements.py", "test_sampler.py", "test_api_sample.py", "test_export.py", "test_export_solvers.py",
                 "test_release_schema.py", "test_reproducibility.py", "test_question_seeds.py"}


def solver_tests(element: str) -> list[str]:
    """Test files (other than the all-element ones) that name the element."""
    if "map" not in _TEST_FILES:
        found: dict[str, list[str]] = {}
        if TESTS_DIR.is_dir():
            for path in sorted(TESTS_DIR.glob("test_*.py")):
                if path.name in GENERIC_TESTS or path.name.startswith("test_web_"):
                    continue
                try:
                    words = set(re.findall(r"['\"]([a-z][a-z0-9_]+)['\"]", path.read_text(encoding="utf-8")))
                except OSError:
                    continue
                for w in words:
                    found.setdefault(w, []).append(f"tests/{path.name}")
        _TEST_FILES["map"] = found
    return _TEST_FILES["map"].get(element, [])


def code_source(element: str) -> str | None:
    """The file that writes an element's templates in code (its filler module), relative to the package."""
    if not (code_templates.has_code_templates(element) or code_templates.is_composite(element)):
        return None
    func = function_registry.functions.get(element)
    try:
        path = Path(inspect.getsourcefile(func)).resolve() if func else None
    except TypeError:
        path = None
    if path is None:
        return "filler_tools/code_templates.py"
    try:
        return path.relative_to(PACKAGE_DIR).as_posix()
    except ValueError:
        return path.name


def _placement() -> dict[str, dict]:
    where = {}
    for version in ("STEER", "STEER-ME"):
        for s in site.tree(version):
            for m in s["modules"]:
                for e in m["elements"]:
                    if e.get("element") and e["element"] not in where and not e.get("shared"):
                        where[e["element"]] = {"setting": s["name"], "setting_number": s["number"],
                                               "module": m["name"], "module_number": m["number"]}
    return where


def known_elements(store: DatasetStore) -> set[str]:
    names = set(site.ELEMENTS_BY_NAME) | set(load_drafts(store))
    names |= set(store.list_elements()) | set(code_templates.elements()) | set(code_templates.composites())
    names |= set(load_manifest().get("elements", {}))
    return {n for n in names if re.fullmatch(r"[A-Za-z0-9_\-]+", n)}


def element_info(store: DatasetStore, element: str, manifest: dict | None = None, drafts: dict | None = None,
                 placement: dict | None = None) -> dict:
    manifest = manifest if manifest is not None else load_manifest().get("elements", {})
    drafts = drafts if drafts is not None else load_drafts(store)
    placement = placement if placement is not None else _placement()
    d = site.ELEMENTS_BY_NAME.get(element)
    draft = drafts.get(element)
    entry = manifest.get(element) or {}
    code = code_source(element)
    if d:
        group, version, status = d["version"], d["version"], d.get("status", "")
        display_name = d["display_name"]
        place = placement.get(element) or {"setting": d.get("setting", ""), "module": d.get("module", ""),
                                           "module_number": "", "setting_number": ""}
    elif draft:
        group, version, status = "offsite", draft.get("version", ""), "draft"
        display_name = draft.get("display_name") or display_name_for_element(element)
        setting_number = next((t["number"] for t in site.tree(version) if t["name"] == draft.get("setting")), "") \
            if version in ("STEER", "STEER-ME") else ""
        place = {"setting": draft.get("setting", ""), "module": draft.get("module", ""), "module_number": draft.get("module_number", ""),
                 "setting_number": setting_number}
    else:
        group, version, status = "offsite", "", "legacy"
        display_name = display_name_for_element(element)
        place = {"setting": "", "module": "", "module_number": "", "setting_number": ""}
    merged_into = site.ALIASES.get(element)
    seeds = seed_count(store, element)
    cands = candidate_counts(store, element)
    templates = entry.get("template_question_count")
    if templates is None:
        templates = stored_template_count(store, element)
    record = review._record(element)
    info = {
        "element": element, "display_name": display_name, "group": group, "version": version, "status": status,
        "status_label": {"release-candidate": "released"}.get(status, status),
        "status_reason": (d or {}).get("status_reason", ""), "summary": (d or draft or {}).get("summary", ""),
        "description": (d or {}).get("description", ""), "on_site": bool(d), "draft": draft, "merged_into": merged_into,
        "merged_into_name": site.ELEMENTS_BY_NAME[merged_into]["display_name"] if merged_into in site.ELEMENTS_BY_NAME else "",
        **place, "code": code, "filler": element in function_registry.functions, "tests": solver_tests(element),
        "seeds": seeds, "candidates": cands, "templates": templates, "in_manifest": bool(entry),
        "review": record, "reviewed": bool(record.get("reviewed_at")) or _sample_read(record),
    }
    info["steps"] = _steps(info)
    return info


def _sample_read(record: dict) -> bool:
    s = record.get("sample")
    return isinstance(s, dict) and bool(s.get("read_at"))


def _steps(i: dict) -> list[dict]:
    c = i["candidates"]
    rows = []
    for key, label, title in STEPS:
        if key == "seeds":
            count, done = ("in code" if i["code"] and not i["seeds"] else str(i["seeds"])), bool(i["seeds"]) or bool(i["code"])
        elif key == "candidates":
            if i["code"] and not c["total"]:
                count, done = "in code", True
            else:
                count, done = (f"{c['accepted']} of {c['total']} accepted" if c["total"] else "0"), c["accepted"] > 0
        elif key == "templates":
            n = i["templates"]
            count, done = (f"{n}" if n is not None else "0") + (" in code" if i["code"] else ""), bool(n)
        else:
            from autosteer.web.review import sample_size

            s = i["review"].get("sample")
            s = s if isinstance(s, dict) and "read" in s else {}
            n = s.get("n") or sample_size(i["templates"] or 0)  # before any mark: the rule on the template count
            count = "reviewed" if i["review"].get("reviewed_at") and not s.get("read") else f"{s.get('read', 0)} of {n} read"
            done = i["reviewed"]
        linked = not (i["code"] and key in ("seeds", "candidates"))  # code elements: steps 1 and 2 are not used
        if not linked:
            count = "written in code"
        short = "in code" if not linked else (f"{int(count):,}" if str(count).isdigit() else
                 str(count).replace(" of ", "/").replace(" accepted", "").replace(" read", ""))
        rows.append({"key": key, "label": label, "title": title, "count": count, "short": short, "done": done, "linked": linked})
    return rows


def all_elements(store: DatasetStore) -> list[dict]:
    manifest = load_manifest().get("elements", {})
    drafts = load_drafts(store)
    placement = _placement()
    rows = [element_info(store, e, manifest, drafts, placement) for e in known_elements(store)]
    order = {"STEER": 0, "STEER-ME": 1, "offsite": 2}

    def module_key(r):
        nums = re.findall(r"\d+", str(r.get("module_number") or ""))
        return [int(n) for n in nums] or [99]
    rows.sort(key=lambda r: (order[r["group"]], r["status"] == "legacy", module_key(r), r["display_name"].lower()))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Flash messages for redirect-after-post.
# ---------------------------------------------------------------------------------------------------------------
_FLASH: dict[str, tuple[float, dict]] = {}


def flash(payload: dict) -> str:
    now = time.time()
    for k in [k for k, (t, _) in _FLASH.items() if now - t > 600]:
        _FLASH.pop(k, None)
    token = secrets.token_urlsafe(6)
    _FLASH[token] = (now, payload)
    return token


def take_flash(token: str | None) -> dict:
    if not token:
        return {}
    hit = _FLASH.pop(token, None)
    return hit[1] if hit else {}


def step_url(element: str, step: str = "", flash_payload: dict | None = None, **params) -> str:
    url = f"/developer/e/{element}" + (f"/{step}" if step else "")
    if flash_payload and (flash_payload.get("success") or flash_payload.get("errors")):
        params["f"] = flash(flash_payload)
    query = "&".join(f"{k}={v}" for k, v in params.items() if v not in (None, ""))
    return url + ("?" + query if query else "")


# ---------------------------------------------------------------------------------------------------------------
# Page context
# ---------------------------------------------------------------------------------------------------------------
def crumbs_for(info: dict | None, step: str | None = None, label: str | None = None) -> list[dict]:
    """Dashboard › element › step, with the same names as the sidebar (step numbers only in the sidebar list)."""
    crumbs = [{"label": "Dashboard", "href": "/developer"}]
    if info:
        crumbs.append({"label": info["display_name"], "href": f"/developer/e/{info['element']}"})
        if step in STEP_NAMES:
            crumbs.append({"label": STEP_NAMES[step], "href": f"/developer/e/{info['element']}/{step}"})
    elif label and label != "Dashboard":
        crumbs.append({"label": label, "href": None})
    crumbs[-1]["href"] = None
    return crumbs


def dev_context(info: dict | None = None, step: str | None = None, page: str = "", label: str | None = None,
                status: dict | None = None) -> dict:
    no_provider = status is not None and not status.get("ai_configured")
    crumbs = crumbs_for(info, step, label)
    parts = [STEP_LABELS.get(step, ""), info["display_name"] if info else (label or ""), "Developer"]
    title = " · ".join(p for p in parts if p)
    return {"dev": {"info": info, "step": step, "page": page, "crumbs": crumbs, "no_provider": no_provider},
            "developer_nav": True, "page_title": title}


def canonical(store: DatasetStore, element: str) -> str | None:
    """The element id to use, or None if the URL names no element."""
    if element in site.ALIASES and element not in known_elements(store):
        return site.ALIASES[element]
    return element if element in known_elements(store) else None


# ---------------------------------------------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------------------------------------------
def install(app: FastAPI, templates: Jinja2Templates, public_mode: Callable[[], bool], status: Callable[[], dict],
            step_context: Callable) -> None:
    """step_context(store, element, step, request) -> extra template context for a step page (app.py builds it)."""

    def guard():
        if public_mode():
            return RedirectResponse(url="/autosteer", status_code=303)
        return None

    @app.get("/developer", response_class=HTMLResponse, include_in_schema=False)
    def dashboard(request: Request, f: str | None = None, store: DatasetStore = Depends(get_dataset_store)):
        """The Dashboard: every element with its steps and status, and the counts that need attention."""
        if (r := guard()):
            return r
        if request.query_params.get("view") == "review":  # old link to the review list
            return RedirectResponse(url="/developer?review=not", status_code=301)
        rows = all_elements(store)
        ctx = {"sections": element_sections(rows), "standing": _standing(store), "message": take_flash(f),
               **dev_context(page="dashboard", label="Dashboard")}
        ctx["dev"]["crumbs"] = []  # the page title says it
        return templates.TemplateResponse(request, "dev_elements.html", ctx)

    def _standing(store: DatasetStore) -> dict:
        rows = all_elements(store)
        on_site = [r for r in rows if r["on_site"] and r["status"] != "held"]  # the released elements
        return {"on_site": len(on_site), "not_reviewed": sum(not r["reviewed"] for r in on_site),
                "drafts": sum(r["status"] == "draft" for r in rows), "held": sum(r["status"] == "held" for r in rows),
                "open_candidates": sum(r["candidates"]["open"] for r in rows if r["status"] != "legacy")}

    def _overview_page(request: Request, page: str, title: str, template: str, extra: dict | None = None):
        ctx = {**(extra or {}), **dev_context(page=page, label=title)}
        ctx["dev"]["crumbs"] = []  # Guide and Settings pages: the title says it
        ctx["page_title"] = f"{title} · Developer"
        return templates.TemplateResponse(request, template, ctx)

    @app.get("/developer/how", response_class=HTMLResponse, include_in_schema=False)
    def how_page(request: Request):
        if (r := guard()):
            return r
        return _overview_page(request, "how", "How elements are made", "dev_how.html")

    @app.get("/developer/submission", response_class=HTMLResponse, include_in_schema=False)
    def submission_page(request: Request):
        if (r := guard()):
            return r
        return _overview_page(request, "submission", "Review and submission", "dev_submission.html")

    @app.get("/developer/status", include_in_schema=False)
    def status_page():
        if (r := guard()):
            return r
        return RedirectResponse(url="/developer", status_code=301)

    @app.get("/developer/elements", include_in_schema=False)
    def elements_page(request: Request):
        """The element table moved to /developer (the Dashboard); keep the filters of old links."""
        if (r := guard()):
            return r
        q = request.url.query
        return RedirectResponse(url="/developer" + ("?" + q if q else ""), status_code=301)

    @app.get("/developer/new", response_class=HTMLResponse, include_in_schema=False)
    def new_page(request: Request, store: DatasetStore = Depends(get_dataset_store)):
        if (r := guard()):
            return r
        return templates.TemplateResponse(request, "dev_new.html", _new_context(store, {}, []))

    def _new_context(store: DatasetStore, form: dict, errors: list[str]) -> dict:
        taxonomy = menu_taxonomy(store)
        known = known_elements(store)
        fillers = sorted(n for n in function_registry.functions if n not in known)
        return {"taxonomy": taxonomy, "taken": sorted(taken_ids(store)), "fillers": fillers, "form": form,
                "errors": errors, **dev_context(page="new", label="New element")}

    @app.post("/developer/new", response_class=HTMLResponse, include_in_schema=False)
    def create_element(request: Request, display_name: str = Form(""), element_id: str = Form(""), version: str = Form(""),
                       setting: str = Form(""), module: str = Form(""), new_setting: str = Form(""),
                       new_module: str = Form(""), summary: str = Form(""),
                       store: DatasetStore = Depends(get_dataset_store)):
        if public_mode():
            raise HTTPException(status_code=403, detail="Authoring is only available in the local autosteer console.")
        form = {"display_name": display_name.strip(), "element_id": element_id.strip(), "version": version,
                "setting": setting, "module": module, "new_setting": new_setting.strip(),
                "new_module": new_module.strip(), "summary": summary.strip()}
        errors = []
        new_id = derive_id(form["element_id"] or form["display_name"])
        if not form["display_name"]:
            errors.append("Give the element a name.")
        if not new_id:
            errors.append("The name needs at least one letter or digit to make an id.")
        elif new_id in taken_ids(store):
            errors.append(f"The id {new_id} is already used. Change the name or the id.")
        if version not in ("STEER", "STEER-ME"):
            errors.append("Choose a benchmark.")
        settings = {x["name"]: x for x in menu_taxonomy(store).get(version, [])}
        module_number, entry, custom_setting, custom_module = "", None, False, False
        if setting == NEW_PLACE:
            setting, custom_setting = form["new_setting"], True
            if not setting:
                errors.append("Name the new setting.")
            elif setting.lower() in {k.lower() for k in settings}:
                errors.append(f"A setting called {setting} already exists in {version}; choose it from the list.")
        elif setting in settings:
            entry = settings[setting]
            custom_setting = entry["custom"]
        else:
            errors.append("Choose a setting.")
        if module == NEW_PLACE:
            module, custom_module = form["new_module"], True
            if not module:
                errors.append("Name the new module.")
            elif entry and module.lower() in {m["name"].lower() for m in entry["modules"]}:
                errors.append(f"A module called {module} already exists in {entry['name']}; choose it from the list.")
        elif entry is not None:
            match = next((m for m in entry["modules"] if m["name"] == module), None)
            if match is None:
                errors.append("Choose a module.")
            else:
                module_number, custom_module = match["number"], match["custom"]
        elif custom_setting:
            errors.append("A new setting needs a new module: name it.")
        if not form["summary"]:
            errors.append("Write a one-sentence summary.")
        if errors:
            return templates.TemplateResponse(request, "dev_new.html", _new_context(store, form, errors), status_code=400)
        record = {"element": new_id, "display_name": form["display_name"], "version": version, "setting": setting,
                  "module": module, "module_number": module_number, "custom_setting": custom_setting,
                  "custom_module": custom_module, "summary": form["summary"], "status": "draft",
                  "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        _drafts_dir(store).mkdir(parents=True, exist_ok=True)
        (_drafts_dir(store) / f"{new_id}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        store.dataset_path("example_seeds", new_id).mkdir(parents=True, exist_ok=True)
        return RedirectResponse(url=step_url(new_id, "seeds", {"success": f"Created {form['display_name']} ({new_id}). Start with an example seed."}),
                                status_code=303)

    @app.post("/developer/e/{element}/discard", include_in_schema=False)
    def discard_draft(element: str, store: DatasetStore = Depends(get_dataset_store)):
        if public_mode():
            raise HTTPException(status_code=403, detail="Authoring is only available in the local autosteer console.")
        drafts = load_drafts(store)
        if element not in drafts:
            raise HTTPException(status_code=404, detail="Only draft elements can be discarded.")
        info = element_info(store, element)
        if info["seeds"] or info["candidates"]["total"] or info["templates"]:
            raise HTTPException(status_code=400, detail="This draft has data; delete its seeds and candidates first.")
        (_drafts_dir(store) / f"{element}.json").unlink(missing_ok=True)
        # The folders hold at most an emptied dataset (every seed deleted); anything else is left alone.
        for kind in ("example_seeds", "generated_seeds"):
            folder = store.dataset_path(kind, element)
            if folder.is_dir() and all(p.name in ("dataset.pkl", "dataset.json", "review.json") or p.suffix == ".parquet" for p in folder.iterdir()):
                for p in folder.iterdir():
                    p.unlink()
                folder.rmdir()
        return RedirectResponse(url="/developer", status_code=303)

    @app.get("/developer/setup", response_class=HTMLResponse, include_in_schema=False)
    def setup_page(request: Request, f: str | None = None, store: DatasetStore = Depends(get_dataset_store)):
        if (r := guard()):
            return r
        return _overview_page(request, "setup", "AI setup", "dev_setup.html", _setup_context(store, take_flash(f)))

    @app.post("/developer/setup/key", include_in_schema=False)
    def set_session_key(name: str = Form(...), value: str = Form("")):
        """Use a key for this server process only: it goes into the process environment, never to disk or a page."""
        if public_mode():
            raise HTTPException(status_code=403, detail="Authoring is only available in the local autosteer console.")
        if name not in SESSION_VARS:
            raise HTTPException(status_code=400, detail="Unknown variable")
        if value.strip():
            os.environ[name] = value.strip()
            _SESSION_KEYS.add(name)
            message = f"{name} is set until the server stops."
        else:
            message = "Nothing entered."
        return RedirectResponse(url="/developer/setup?f=" + flash({"success": message}), status_code=303)

    @app.post("/developer/setup/forget", include_in_schema=False)
    def forget_session_keys():
        if public_mode():
            raise HTTPException(status_code=403, detail="Authoring is only available in the local autosteer console.")
        for name in list(_SESSION_KEYS):
            os.environ.pop(name, None)
            _SESSION_KEYS.discard(name)
        return RedirectResponse(url="/developer/setup?f=" + flash({"success": "Keys entered on this page are forgotten."}),
                                status_code=303)

    def _setup_context(store: DatasetStore, message: dict) -> dict:
        from autosteer import data_files

        try:
            verify = data_files.verify(Path(store.data_root))
        except Exception as exc:  # the pinned list is missing in a development checkout
            verify = {"error": str(exc)}
        manifest = load_manifest()
        st = status()
        return {"providers": st.get("providers") or provider_rows(), "settings": load_settings(store),
                "session_keys": bool(_SESSION_KEYS),
                "data_dir": str(store.data_root), "verify": verify, "manifest_date": manifest.get("generated_at", ""),
                "manifest_count": len(manifest.get("elements", {})), "message": message, "status": st,
                }

    @app.post("/developer/setup", include_in_schema=False)
    def save_setup(gen_provider: str = Form(""), gen_model: str = Form(""), solve_provider: str = Form(""),
                   solve_model: str = Form(""), store: DatasetStore = Depends(get_dataset_store)):
        if public_mode():
            raise HTTPException(status_code=403, detail="Authoring is only available in the local autosteer console.")
        settings = load_settings(store)
        for role, provider, model in (("generation", gen_provider, gen_model), ("solve", solve_provider, solve_model)):
            if provider in PROVIDERS_BY_KEY and model.strip():
                settings[role] = {"provider": provider, "model": model.strip()[:120]}
        save_settings(store, settings)
        return RedirectResponse(url="/developer/setup?f=" + flash({"success": "Default models saved."}), status_code=303)

    def _element_or_redirect(store: DatasetStore, element: str, step: str = ""):
        name = canonical(store, element)
        if name is None:
            raise HTTPException(status_code=404, detail=f"No element {element}")
        if name != element:
            return None, RedirectResponse(url=step_url(name, step), status_code=301)
        return element_info(store, name), None

    @app.get("/developer/e/{element}", response_class=HTMLResponse, include_in_schema=False)
    def overview(request: Request, element: str, store: DatasetStore = Depends(get_dataset_store)):
        if (r := guard()):
            return r
        info, redirect = _element_or_redirect(store, element)
        if redirect:
            return redirect
        return templates.TemplateResponse(request, "dev_overview.html", {"i": info, **dev_context(info, "overview", status=status())})

    @app.get("/developer/e/{element}/template-rows", include_in_schema=False)
    def template_rows(element: str, offset: int = 0, q: str = "", limit: int = 25, domain: str = "", type: str = "",
                      tag: str = "", level: str = ""):
        if public_mode():
            raise HTTPException(status_code=404, detail="Not found")
        rows, total, facets = template_page(element, offset, q, limit, {"domain": domain, "type": type, "tag": tag, "level": level})
        return {"rows": rows, "total": total, "offset": offset, "facets": facets}

    @app.get("/developer/e/{element}/{step}", response_class=HTMLResponse, include_in_schema=False)
    def step_page(request: Request, element: str, step: str, f: str | None = None, store: DatasetStore = Depends(get_dataset_store)):
        if (r := guard()):
            return r
        if step not in STEP_KEYS:
            raise HTTPException(status_code=404, detail="No such step")
        info, redirect = _element_or_redirect(store, element, step)
        if redirect:
            return redirect
        if not next(s for s in info["steps"] if s["key"] == step)["linked"]:
            return RedirectResponse(url=step_url(element), status_code=302)
        return render_step(request, store, element, step, message=take_flash(f))

    def render_step(request: Request, store: DatasetStore, element: str, step: str, message: dict | None = None,
                    status_code: int = 200, **extra) -> HTMLResponse:
        info = element_info(store, element)
        st = status()
        ctx = {"i": info, "status": st, "message": message or {}, **dev_context(info, step, status=st)}
        ctx.update(step_context(store, element, step, request, st, **extra))
        return templates.TemplateResponse(request, f"dev_{step}.html", ctx, status_code=status_code)

    app.state.render_step = render_step

    # Addresses from the stage-by-stage layout.
    def _old(store: DatasetStore, element_name: str, step: str, **params) -> RedirectResponse:
        name = canonical(store, element_name.strip().replace(" ", "_").lower()) if element_name else None
        if not name:
            return RedirectResponse(url="/developer", status_code=301)
        return RedirectResponse(url=step_url(name, step, **params), status_code=301)

    @app.get("/seeds/new", include_in_schema=False)
    def old_new_seed(element_name: str = "", question_id: int | None = None, sub_id: int = 0, store: DatasetStore = Depends(get_dataset_store)):
        if (r := guard()):
            return r
        return _old(store, element_name, "seeds", edit=f"{question_id}.{sub_id}" if question_id is not None else "new")

    @app.get("/generated-seeds", include_in_schema=False)
    def old_generated(element_name: str = "", store: DatasetStore = Depends(get_dataset_store)):
        if (r := guard()):
            return r
        return _old(store, element_name, "candidates")

    @app.get("/templates/generate", include_in_schema=False)
    def old_templates(element_name: str = "", store: DatasetStore = Depends(get_dataset_store)):
        if (r := guard()):
            return r
        return _old(store, element_name, "templates")

    @app.get("/elements/{element_name}/{kind}", include_in_schema=False)
    def old_dataset(element_name: str, kind: str, store: DatasetStore = Depends(get_dataset_store)):
        steps = {"example_seeds": "seeds", "generated_seeds": "candidates", "templates": "templates"}
        if kind not in steps:
            raise HTTPException(status_code=404, detail="Not found")
        if public_mode():  # developer views are not on the public site
            raise HTTPException(status_code=404, detail="Not found")
        return _old(store, element_name, steps[kind])

    @app.get("/developer/review", include_in_schema=False)
    def old_review_index():
        if public_mode():
            raise HTTPException(status_code=404, detail="Not found")
        if (r := guard()):
            return r
        return RedirectResponse(url="/developer?review=not", status_code=301)

    @app.get("/developer/review/{element}", include_in_schema=False)
    def old_review(element: str, store: DatasetStore = Depends(get_dataset_store)):
        if public_mode():
            raise HTTPException(status_code=404, detail="Not found")
        if (r := guard()):
            return r
        return _old(store, element, "review")


def domain_options() -> list[dict]:
    return [{"name": k, "label": v} for k, v in DOMAINS.items()]


def tag_options() -> list[str]:
    return list(TAGS)


_ROWS_CACHE: dict = {}
ROW_KEYS = ("id", "template_id", "pieces", "type", "domain", "tags", "difficulty", "difficulty_raw")


FACETS = ("domain", "type", "tag", "level")


def _facet_values(r: dict, facet: str) -> list[str]:
    if facet == "tag":
        return [t.strip() for t in str(r.get("tags") or "").replace(",", ";").split(";") if t.strip()]
    if facet == "level":
        return [str(r.get("difficulty"))]
    return [str(r.get(facet) or "")]


def template_page(element: str, offset: int = 0, q: str = "", limit: int = 25,
                  filters: dict | None = None) -> tuple[list[dict], int, dict]:
    """One page of an element's templates (as the sampler uses them), optionally only those containing all the words
    of q and matching the metadata filters ({facet: value}). Also the counts of each facet's values among the
    templates the other filters (and q) leave. Pages keep elements with thousands of templates fast."""
    key = (element, load_manifest().get("generated_at"))
    if _ROWS_CACHE.get("key") != key:
        rows = review.template_rows(element)
        _ROWS_CACHE.update(key=key, rows=[{**{k: r[k] for k in ROW_KEYS}, "text": " ".join(r["parts"]).lower()} for r in rows])
    rows = _ROWS_CACHE["rows"]
    words = q.lower().split()
    if words:
        rows = [r for r in rows if all(w in r["text"] for w in words)]
    filters = {f: v for f, v in (filters or {}).items() if f in FACETS and v}

    def keep(r, skip=None):
        return all(v in _facet_values(r, f) for f, v in filters.items() if f != skip)

    facets = {}
    for f in FACETS:
        counts: dict = {}
        for r in rows:
            if keep(r, f):
                for v in _facet_values(r, f):
                    counts[v] = counts.get(v, 0) + 1
        order = sorted(counts, key=lambda v: (int(v) if v.isdigit() else 0, v) if f == "level" else (-counts[v], v))
        facets[f] = [{"value": v, "count": counts[v], "on": filters.get(f) == v} for v in order]
    rows = [r for r in rows if keep(r)]
    page = [{k: r[k] for k in ROW_KEYS} for r in rows[offset:offset + max(1, min(limit, 200))]]
    return page, len(rows), facets


BENCHES = [("STEER", "STEER"), ("STEER-ME", "STEER-ME"), ("draft", "Drafts"), ("legacy", "Legacy folders")]


def element_sections(rows: list[dict]) -> list[dict]:
    """Rows grouped by benchmark, then setting, then module, in the taxonomy's order (numbers as in the papers)."""
    sections = []
    for bench, label in BENCHES:
        mine = [r for r in rows if (r["group"] if r["group"] != "offsite" else ("draft" if r["status"] == "draft" else "legacy")) == bench]
        settings: dict = {}
        for r in mine:
            r["bench"] = bench
            skey = (str(r.get("setting_number") or ""), r.get("setting") or "")
            mkey = (str(r.get("module_number") or ""), r.get("module") or "")
            settings.setdefault(skey, {}).setdefault(mkey, []).append(r)

        def num(key):
            parts = [int(x) for x in re.findall(r"\d+", key[0])]
            return (parts or [999], key[1])
        sections.append({"bench": bench, "label": label, "count": len(mine), "settings": [
            {"number": sk[0], "name": sk[1] or ("No setting" if bench != "legacy" else ""), "modules": [
                {"number": mk[0], "name": mk[1], "rows": mods[mk]} for mk in sorted(mods, key=num)]}
            for sk, mods in sorted(settings.items(), key=lambda kv: num(kv[0]))]})
    return sections

