from __future__ import annotations

import os
import re
import random
import secrets
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from autosteer.sampler import ElementSampler, SamplerConfig, get_sampleable_elements, load_manifest
from autosteer.sampler.difficulty import level_count, public_entry, raw_levels, relative_scale, to_raw, to_relative
from autosteer.sampler.generator import SEED_MAX, derive_seed
from autosteer.filler_tools import function_registry
from autosteer.constants import DOMAINS, TAGS
from autosteer import instruction_parts
from autosteer.schemas import Metadata, OptionRecord, SamplerRequest
from autosteer.services.dataset_store import DATASET_KINDS, DatasetStore, get_dataset_store
from autosteer.services.generated_seed_service import (
    build_target_grid,
    delete_generated_seed,
    example_seed_summary,
    generate_generated_seed,
    generated_seed_records,
    uses_template_options,
    set_review_status,
    update_generated_seed,
)
from autosteer.services.generated_seed_validation import (
    run_generated_seed_validation,
    run_single_generated_seed_validation,
)
from autosteer.services.template_generation_service import (
    has_accepted_generated_seeds,
    generate_template_batch,
    template_generation_summary,
)
from autosteer.taxonomy import (
    display_name_for_element,
    element_category_summary,
    is_non_strategic_element,
)
from autosteer.web import devmode, review, site


WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
# Where the hosted, read-only site sends requests for the authoring pages.
LOCAL_ONLY_REDIRECT = "/autosteer"


def _highlight_placeholders(text: str) -> Markup:
    escaped = escape(text)
    highlighted = re.sub(
        r"(\{[^{}]+\})",
        r'<span class="slot">\1</span>',
        str(escaped),
    )
    return Markup(highlighted)


templates.env.filters["highlight_placeholders"] = _highlight_placeholders


def _seed_parts(store, element: str, records=None) -> dict:
    """The element's instruction parts; an element with none saved gets its task from its hand-written instructions."""
    parts = instruction_parts.load(store.data_root, element)
    if not parts["task"]:
        if records is None:
            try:
                records = store.records("example_seeds", element, dataframe_type="example_questions", limit=500)
            except FileNotFoundError:
                records = []
        parts["task"] = instruction_parts.default_task(getattr(r, "instructions", "") or "" for r in records)
    return parts


def _split_tags(tags: str) -> list[str]:
    return [t.strip() for t in re.split(r"[;,]", tags or "") if t.strip()]


def _parts_with_form(parts: dict, *, type_key: str, raw_level, domain: str, task: str, type_text: str,
                     level_text: str, domain_text: str, tag_keys: list[str], tag_texts: list[str]) -> dict:
    parts = {**parts, "types": dict(parts["types"]), "levels": dict(parts["levels"]),
             "domains": dict(parts["domains"]), "tags": dict(parts["tags"])}
    parts["task"] = task
    if type_key:
        parts["types"][type_key] = type_text
    if str(raw_level) != "":
        parts["levels"][str(raw_level)] = level_text
    if domain and domain != "base":
        parts["domains"][domain] = domain_text
    for k, v in zip(tag_keys, tag_texts):
        parts["tags"][k.strip()] = v
    return parts


def _compose_seed(parts: dict, metadata: dict, question_text: str, options_text: str, correct_answer: str) -> list[dict]:
    options = [line.strip() for line in options_text.splitlines() if line.strip()]
    variables = [v for v in _template_variables(question_text, options_text, correct_answer).split(";") if v]
    return instruction_parts.compose(parts, metadata, variables=variables, num_options=len(options),
                                     include_answer=bool(options and correct_answer.strip()))


def _seed_raw_difficulty(store, element: str, level) -> int:
    """The raw level stored for a level typed in the seed editor (1 = the element's easiest, as on the site);
    one above the hardest adds a new, harder level."""
    try:
        records = store.records("example_seeds", element, dataframe_type="example_questions", limit=500)
    except FileNotFoundError:
        records = []
    raws = []
    for r in records:
        md = r.get("metadata") if isinstance(r, dict) else getattr(r, "metadata", None)
        raws.append(md.get("difficulty_level") if isinstance(md, dict) else getattr(md, "difficulty_level", None))
    scale = _level_scale(element, raws)
    by_level = {rel: raw for raw, rel in scale.items()}
    level = int(str(level).strip())
    if level < 1:
        raise ValueError("Difficulty must be 1 or more (1 is this element's easiest level).")
    if level in by_level:
        return by_level[level]
    return max(scale) + (level - len(scale))  # above the hardest: a new level, as far above as typed


def _level_scale(element: str, raws=()) -> dict[int, int]:
    """{raw level: level shown} for an element, as on the site (1 = its easiest): the manifest's raw levels plus any
    given ones (seeds of a draft element can use levels no template has yet)."""
    levels = set()
    try:
        levels |= set(raw_levels(element))
    except KeyError:
        pass
    for r in raws:
        try:
            levels.add(int(r))
        except (TypeError, ValueError):
            pass
    return relative_scale(levels or {0})


templates.env.globals["level_scale"] = _level_scale
templates.env.filters["split_semicolons"] = lambda s: [t.strip() for t in str(s or "").split(";") if t.strip()]


def _sampleable_names() -> list[str]:
    return sorted(
        element_name
        for element_name in get_sampleable_elements(load_manifest())
        if is_non_strategic_element(element_name)
    )


def _sampleable_set() -> set[str]:
    return set(_sampleable_names())


def _seed_authorable_names() -> list[str]:
    return sorted(set(_sampleable_names()) | set(function_registry.names))


def _seed_authorable_set(store: DatasetStore | None = None) -> set[str]:
    names = set(_seed_authorable_names())
    if store is not None:
        names |= set(devmode.load_drafts(store))
    return names


def _normalize_element_name(element_name: str) -> str:
    return element_name.strip().replace(" ", "_").lower()


def _ensure_sampleable(element_name: str) -> None:
    if element_name not in _sampleable_set():
        raise HTTPException(status_code=404, detail=f"Element is not available in the sampler: {element_name}")


def _ensure_seed_authorable(element_name: str, store: DatasetStore | None = None) -> None:
    if element_name not in _seed_authorable_set(store):
        raise HTTPException(
            status_code=404,
            detail=f"Element does not have a registered filler function: {element_name}",
        )


def _sampleable_summaries(store: DatasetStore) -> list[dict]:
    manifest = load_manifest()
    sampleable = set(_sampleable_names())
    return [
        {
            **store.summarize_element(element_name),
            "display_name": display_name_for_element(element_name),
            "sampleable": True,
            "manifest": public_entry(element_name, manifest.get("elements", {}).get(element_name, {})),
        }
        for element_name in sorted(sampleable)
        if element_name in store.list_elements()
    ]


def _parse_options(options_text: str, correct_answer: str) -> list[OptionRecord]:
    option_texts = [line.strip() for line in options_text.splitlines() if line.strip()]
    if not option_texts:
        return [OptionRecord(option_id=0, option_text="", correct_answer=True)]

    correct_answer = correct_answer.strip()
    if correct_answer.isdigit():
        correct_index = int(correct_answer)
        if correct_index >= len(option_texts):
            raise HTTPException(status_code=400, detail="Correct answer index is outside the options list.")
        return [
            OptionRecord(option_id=i, option_text=text, correct_answer=i == correct_index)
            for i, text in enumerate(option_texts)
        ]

    if correct_answer not in option_texts:
        raise HTTPException(status_code=400, detail="Correct answer must match an option or be a zero-based index.")

    return [
        OptionRecord(option_id=i, option_text=text, correct_answer=text == correct_answer)
        for i, text in enumerate(option_texts)
    ]


def _seed_draft(
    element_name: str = "",
    question_id: str | int = "",
    sub_id: str | int = 0,
    saved_element_name: str = "",
    question_text: str = "",
    domain: str = "",
    type: str = "",
    difficulty_level: str | int = 0,
    tags: str = "",
    options_text: str = "",
    correct_answer: str = "",
    instructions: str = "",
    ai_model: str = "",
) -> dict:
    return {
        "element_name": element_name,
        "question_id": question_id,
        "sub_id": sub_id,
        "saved_element_name": saved_element_name or element_name,
        "question_text": question_text,
        "domain": domain,
        "type": type,
        "difficulty_level": difficulty_level,
        "tags": tags,
        "options_text": options_text,
        "correct_answer": correct_answer,
        "instructions": instructions,
        "ai_model": ai_model,
    }


def _decorate_generated_records(records: list[dict]) -> list[dict]:
    decorated = []
    for record in records:
        item = dict(record)
        review = item.get("review", {})
        checks = review.get("checks", {}) if isinstance(review, dict) else {}
        solve_check = checks.get("solve_check") or checks.get("ai_validation") if isinstance(checks, dict) else None
        validation_status = solve_check.get("status") if solve_check else "not_run"
        item["validation_status"] = validation_status
        if validation_status == "not_run":
            item["validation_label"] = "unchecked"
        elif validation_status == "error":
            item["validation_label"] = "filler error"
        elif validation_status == "passed":
            item["validation_label"] = "solved"
        elif validation_status == "failed":
            item["validation_label"] = "missed"
        else:
            item["validation_label"] = validation_status
        decorated.append(item)
    return decorated


def _generated_validation_summary(records: list[dict]) -> dict:
    summary = {
        "checkable": 0,
        "not_run": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
    }
    for record in records:
        review = record.get("review", {})
        if review.get("status", "draft") == "rejected":
            continue
        summary["checkable"] += 1
        checks = review.get("checks", {})
        solve_check = (checks.get("solve_check") or checks.get("ai_validation")) if isinstance(checks, dict) else None
        if not solve_check:
            summary["not_run"] += 1
            continue
        status = solve_check.get("status")
        if status == "passed":
            summary["passed"] += 1
        elif status == "failed":
            summary["failed"] += 1
        else:
            summary["errors"] += 1
    return summary


def _validate_seed_draft(
    element_name: str,
    question_text: str,
    domain: str,
    type: str,
    difficulty_level: str,
    options_text: str,
    correct_answer: str,
    instructions: str,
    store: DatasetStore | None = None,
) -> tuple[str, int, list[str], list[str]]:
    errors = []
    warnings = []
    normalized_element_name = _normalize_element_name(element_name)

    if not normalized_element_name:
        errors.append("Element name is required.")
    elif normalized_element_name not in _seed_authorable_set(store):
        errors.append(f"Element does not have a registered filler function: {normalized_element_name}")

    if not question_text.strip():
        errors.append("Question text is required.")
    if not domain.strip():
        errors.append("Domain is required.")
    if not type.strip():
        errors.append("Type is required.")
    if not instructions.strip():
        errors.append("Instructions are required. Use Generate instructions or write them manually.")

    try:
        parsed_difficulty = int(difficulty_level)
    except (TypeError, ValueError):
        parsed_difficulty = 0
        errors.append("Difficulty must be an integer.")

    option_texts = [line.strip() for line in options_text.splitlines() if line.strip()]
    answer = correct_answer.strip()
    if option_texts:
        if not answer:
            errors.append("Correct answer is required when options are provided.")
        elif answer.isdigit():
            if int(answer) >= len(option_texts):
                errors.append("Correct answer index is outside the options list.")
        elif answer not in option_texts:
            errors.append("Correct answer must match an option or be a zero-based option index.")

    if question_text.strip() and not _template_variables(question_text):
        warnings.append("No placeholders were detected. Seeds usually include placeholders such as {price}.")

    return normalized_element_name, parsed_difficulty, errors, warnings


def _sampler_config_from_request(request: SamplerRequest) -> SamplerConfig:
    return SamplerConfig(
        element_name=request.element_name or None,
        domain=request.domain or None,
        type=request.type or None,
        difficulty_level=request.difficulty_level if request.difficulty_level not in ("", None) else None,
        tags=request.tags or None,
    )


API_MAX_N = 50
# Seeding resets process-wide RNGs, so draws are serialized; one sampler is kept per manifest version.
_API_LOCK = threading.Lock()
# Candidate generation started from the page runs in a thread, one job per element; the page polls its progress.
_GEN_JOBS: dict[str, dict] = {}
_GEN_LOCK = threading.Lock()


def _gen_job(element: str, kind: str = "generate") -> dict | None:
    with _GEN_LOCK:
        job = _GEN_JOBS.get((kind, element))
        return dict(job, errors=list(job["errors"])) if job else None


def _any_job_running(element: str) -> bool:
    return any((_gen_job(element, k) or {}).get("running") for k in ("generate", "check"))


def _start_job(element: str, kind: str, total: int, model: str) -> dict:
    job = {"kind": kind, "total": total, "done": 0, "failed": 0, "passed": 0, "missed": 0, "errors": [],
           "running": True, "model": model, "started_at": time.time(), "finished_at": None}
    with _GEN_LOCK:
        _GEN_JOBS[(kind, element)] = job
    return job


def _run_check_job(store, element: str, job: dict, provider, model, runs_per_seed: int) -> None:
    def progress(summary, total):
        with _GEN_LOCK:
            job.update(total=total, done=summary["validated"], passed=summary["passed"], missed=summary["failed"],
                       failed=summary["errors"])
    def run(n, of):
        with _GEN_LOCK:
            job.update(run=n, runs=of)
    try:
        run_generated_seed_validation(store=store, element_name=element, model=model, provider=provider,
                                      runs_per_seed=runs_per_seed, statuses=("draft", "accepted", "needs_edit"),
                                      on_progress=progress, on_run=run)
    except Exception as exc:
        with _GEN_LOCK:
            job["errors"].append(str(exc)[:300])
    with _GEN_LOCK:
        job["running"] = False
        job["finished_at"] = time.time()


def _run_gen_job(store, element: str, targets: list, provider, model) -> None:
    job = _GEN_JOBS[("generate", element)]
    for target in targets:
        try:
            generate_generated_seed(store=store, element_name=element, target_metadata=target, model=model, provider=provider)
            ok = True
        except Exception as exc:  # one failed target does not stop the rest
            ok = False
            err = f"{target.get('domain', '')} · {target.get('type', '')}: {exc}"[:300]
        with _GEN_LOCK:
            job["done"] += 1
            if not ok:
                job["failed"] += 1
                job["errors"].append(err)
    with _GEN_LOCK:
        job["running"] = False
        job["finished_at"] = time.time()
_API_SAMPLER: dict = {}


def _api_sampler(manifest: dict) -> ElementSampler:
    key = (manifest.get("generated_at"), tuple(sorted(manifest.get("elements", {}))))
    if _API_SAMPLER.get("key") != key:
        _API_SAMPLER.update(key=key, sampler=ElementSampler(manifest=manifest))
    return _API_SAMPLER["sampler"]


def _api_element_names(manifest: dict) -> list[str]:
    """Elements /api/sample serves (STEER and STEER-ME): sampleable, listed on the site, not held."""
    from autosteer.release.rows import catalog

    listed = catalog()
    return sorted(
        name for name in get_sampleable_elements(manifest)
        if name in listed and listed[name].get("status") != "held"
    )


def _api_element(element_name: str | None, names: list[str]) -> None:
    if element_name is not None and element_name not in names:
        raise HTTPException(status_code=404, detail=f"Element is not available in the sampler: {element_name}")


def _api_difficulty_names(level, names: list[str], manifest: dict) -> list[str]:
    """Without an element, a difficulty filter draws from the elements that have that (relative) level."""
    if level is None:
        return names
    text = str(level).strip()
    ok = text.isdigit() and int(text) >= 1
    names = [n for n in names if ok and int(text) <= level_count(n, manifest)]
    if not names:
        top = max((level_count(n, manifest) for n in _api_element_names(manifest)), default=1)
        raise HTTPException(status_code=400, detail=(
            f"difficulty_level must be an integer from 1 to {top}, got {level!r}; levels are relative to each "
            "element (1 = its easiest)"))
    return names


def _api_raw_difficulty(element: str, level, manifest: dict):
    """The sampler's raw level of a relative difficulty_level of the API (400 if out of the element's range)."""
    if level is None:
        return None
    try:
        return to_raw(element, level, manifest)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _sample(config: SamplerConfig) -> dict:
    """Legacy response (no n, seed or template_id): {question, options: [{text, correct}], answer, metadata}.
    difficulty_level, in the filter and the metadata, is the relative level within the element."""
    manifest = load_manifest()
    names = _api_element_names(manifest)
    _api_element(config.element_name, names)
    element = config.element_name or random.choice(_api_difficulty_names(config.difficulty_level, names, manifest))
    config = SamplerConfig(
        element_name=element,
        domain=config.domain,
        type=config.type,
        difficulty_level=_api_raw_difficulty(element, config.difficulty_level, manifest),
        tags=config.tags,
    )
    try:
        with _API_LOCK:
            out = _api_sampler(manifest).sample(config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    md = out.get("metadata")
    if md and md.get("difficulty_level") is not None:
        out["metadata"] = {**md, "difficulty_level": to_relative(element, md["difficulty_level"], manifest)}
    return out


def _sample_rows(
    config: SamplerConfig, n: int | None, seed: int | None, template_id: str | None
) -> dict:
    """{"questions": [row, ...]}: dataset rows (autosteer.schemas.DatasetRow), one per question part.

    Question i is drawn with seed + i (seed random if not given). If that draw is not a valid question
    (autosteer.release.rows.row_problems), seeds derived from it are tried; every row carries the seed it was
    drawn with, so (element, template_id, seed) of a row reproduces it, here and in the exported dataset.
    """
    from autosteer.release.rows import build_rows, row_problems

    n = n or 1
    if seed is None:
        seed = secrets.randbelow(SEED_MAX - API_MAX_N)
    if seed + n - 1 > SEED_MAX:
        raise HTTPException(status_code=422, detail="seed + n - 1 must be below 2**63")
    manifest = load_manifest()
    names = _api_element_names(manifest)
    _api_element(config.element_name, names)
    filters = {k: v for k, v in (("domain", config.domain), ("type", config.type), ("tags", config.tags))
               if v is not None}
    if config.element_name:
        pick = names
        _api_raw_difficulty(config.element_name, config.difficulty_level, manifest)  # 400 before any draw
    else:
        pick = _api_difficulty_names(config.difficulty_level, names, manifest)
    rows: list[dict] = []
    with _API_LOCK:
        sampler = _api_sampler(manifest)
        for i in range(n):
            qseed = seed + i
            element = config.element_name or pick[random.Random(derive_seed("element", qseed)).randrange(len(pick))]
            raw = _api_raw_difficulty(element, config.difficulty_level, manifest)
            for attempt in range(20):
                try:
                    q = sampler.sample_question(element, qseed, template_id=template_id,
                                                **filters, **({} if raw is None else {"difficulty_level": raw}))
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                question_rows = build_rows(q, manifest=manifest)
                if not row_problems(q, question_rows):
                    break
                qseed = derive_seed(seed + i, "retry", attempt)
            else:
                raise HTTPException(status_code=500, detail=f"no valid question for {element} after 20 draws")
            rows.extend(question_rows)
    return {"questions": rows}


def _developer_status() -> dict:
    """Which AI providers this server can use (from the environment; key values are never read into a page)."""
    providers = devmode.provider_rows()
    configured = [p for p in providers if p["configured"]]
    has_azure_key = bool(os.getenv("AZURE_OPENAI_API_KEY"))
    has_azure_endpoint = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))
    allow_browser_key = os.getenv("STEER_ALLOW_BROWSER_API_KEY", "").lower() in {"1", "true", "yes", "on"}
    default_model = os.getenv("GENERATION_MODEL", "gpt-5.4")
    model_options = list(dict.fromkeys([default_model, "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]))
    web_mode = os.getenv("AUTOSTEER_WEB_MODE", os.getenv("STEER_WEB_MODE", "local")).lower()
    return {
        "mode": os.getenv("STEER_MODE", "developer"),
        "web_mode": web_mode,
        "public_mode": web_mode in {"public", "hosted", "read_only", "readonly"},
        "ai_configured": bool(configured),
        "provider": configured[0]["label"] if configured else None,
        "providers": providers,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "azure_configured": has_azure_key and has_azure_endpoint,
        "azure_missing_endpoint": has_azure_key and not has_azure_endpoint,
        "browser_key_allowed": allow_browser_key,
        "default_model": default_model,
        "models": model_options,
        "data_dir": os.getenv("AUTOSTEER_DATA_DIR", os.getenv("STEER_DATA_DIR", "")),
    }


def _public_web_mode() -> bool:
    return bool(_developer_status().get("public_mode", False))


def _ensure_local_authoring() -> None:
    if _public_web_mode():
        raise HTTPException(
            status_code=403,
            detail="Authoring is only available in the local autosteer console.",
        )


def _template_variables(*texts: str) -> str:
    variables = []
    for text in texts:
        variables.extend(re.findall(r"{.*?}", text or ""))
    return ";".join(dict.fromkeys(variables))


def _build_seed_instructions(
    metadata: dict,
    num_options: int,
    include_answer: bool,
    variables: str,
    task_description: str = "",
) -> str:
    if task_description:
        instruction = task_description + "."
    else:
        instruction = "Please give an MCQ example testing the ability " + "<insert task description here>.".upper()

    try:
        if metadata["domain"] != "base":
            instruction += " The question should be concerning " + DOMAINS[metadata["domain"]] + ". "
        else:
            instruction += " The question should not be phrased in terms of a specific domain. "
    except KeyError:
        instruction += " The question should be concerning" + " <insert domain description>. ".upper()

    if metadata.get("tags"):
        tags_instructions = []
        missing_tags = ""
        for tag in metadata["tags"].split(";"):
            try:
                tags_instructions.append(TAGS[tag])
            except KeyError:
                missing_tags += ";" + tag
        if tags_instructions:
            instruction += ". ".join(tags_instructions) + "."
        if len(missing_tags) > 1:
            instruction += " <insert tag descriptions for the following tags: ".upper() + ", ".join(missing_tags) + ">."

    if variables:
        text_variables = ", ".join(set(variables.split(";")))
        instruction += f" Do not include numbers in the example but leave them as variables as in the example below. Use the following variables in the curly braces: {text_variables}."

    if not include_answer:
        instruction += " Do not mark any option with correct_answer as True."
    if num_options == 1:
        instruction += " The MCQ should have only one option."
    elif num_options == 0:
        instruction = instruction.replace(
            " Do not mark any option with correct_answer as True.",
            " Do not mark any option with correct_answer as True and the options should be a list containing a single empty string.",
        )

    instruction += " See the following example as a guide but give a different example and write a narrative: "
    return instruction


def _draft_option_records(options_text: str, correct_answer: str) -> list[dict]:
    option_texts = [line.strip() for line in options_text.splitlines() if line.strip()]
    if not option_texts:
        return []

    correct_answer = correct_answer.strip()
    correct_index = int(correct_answer) if correct_answer.isdigit() else None
    return [
        {
            "option_id": i,
            "option_text": text,
            "correct_answer": i == correct_index or text == correct_answer,
        }
        for i, text in enumerate(option_texts)
    ]


def _seed_draft_question(question_text: str, options_text: str, correct_answer: str) -> dict:
    return {
        "question_id": 0,
        "sub_id": 0,
        "question_text": question_text.strip(),
        "options": _draft_option_records(options_text, correct_answer),
    }


def _json_to_mcq(json_object: dict) -> str:
    mcq_string = f"Question 0-{json_object['sub_id']}:\n"
    mcq_string += f"{json_object['question_text']}\n"
    correct = ""
    for i, option in enumerate(json_object["options"]):
        mcq_string += f"{chr(65 + i)}. {option['option_text']}\n"
        if option["correct_answer"]:
            correct = option["option_text"]
    if json_object["options"]:
        mcq_string += f"Correct Answer: {correct}\n"
    else:
        mcq_string += "Options: none supplied for this seed.\n"
    return mcq_string


NO_PROVIDER = "No AI provider is set up. Add a key for one in Setup (/developer/setup)."


def _seed_coverage(element: str, records: list) -> dict:
    """What the example seeds cover: a type x level grid of counts (levels relative, with the raw level each stands
    for, so an empty cell can open a new seed at it), and the counts of domains and tags."""
    mds = [getattr(r, "metadata", None) for r in records]
    scale = _level_scale(element, [getattr(m, "difficulty_level", None) for m in mds])
    raw_of = {rel: raw for raw, rel in scale.items()}
    types, cells, domains, tags = [], {}, {}, {}
    for m in mds:
        t = str(getattr(m, "type", "") or "")
        if t not in types:
            types.append(t)
        try:
            rel = scale.get(int(getattr(m, "difficulty_level", None)))
        except (TypeError, ValueError):
            rel = None
        cells[f"{t}|{rel}"] = cells.get(f"{t}|{rel}", 0) + 1
        d = str(getattr(m, "domain", "") or "")
        domains[d] = domains.get(d, 0) + 1
        for tag in str(getattr(m, "tags", "") or "").replace(",", ";").split(";"):
            if tag.strip():
                tags[tag.strip()] = tags.get(tag.strip(), 0) + 1
    levels = [{"rel": rel, "raw": raw_of[rel]} for rel in sorted(raw_of)]
    by_count = lambda c: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    type_n = {t: sum(n for k, n in cells.items() if k.split("|")[0] == t) for t in types}
    level_n = {lv["rel"]: sum(n for k, n in cells.items() if k.split("|")[1] == str(lv["rel"])) for lv in levels}
    for lv in levels:
        lv["n"] = level_n[lv["rel"]]
    return {"types": types, "levels": levels, "cells": cells, "domains": by_count(domains), "tags": by_count(tags),
            "type_counts": by_count(type_n),
            "gaps": sum(1 for t in types for lv in levels if not cells.get(f"{t}|{lv['rel']}"))}


def _seeds_context(
    store: DatasetStore,
    element: str,
    request: Request,
    status: dict,
    draft: dict | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict:
    errors = list(errors or [])
    edit = request.query_params.get("edit", "")
    if draft is None and edit:
        if edit == "new":  # a gap in the coverage grid opens with its type and level filled in
            q = request.query_params
            draft = _seed_draft(element_name=element, type=q.get("type", ""), difficulty_level=q.get("level", 0) or 0)
        else:
            qid, _, sid = edit.partition(".")
            try:
                draft = _seed_draft(
                    **store.seed_draft(element, question_id=int(qid), sub_id=int(sid or 0)),
                    saved_element_name=element,
                )
            except (FileNotFoundError, ValueError, KeyError, IndexError):
                errors.append(f"No saved seed {edit} for this element.")
    try:
        records = store.records("example_seeds", element, dataframe_type="example_questions", limit=500)
    except FileNotFoundError:
        records = []
    parts, lv_raw, legacy = None, {}, ""
    if draft is not None:
        parts = _seed_parts(store, element, records)
        scale = _level_scale(element, [getattr(getattr(r, "metadata", None), "difficulty_level", None) for r in records]
                             + [draft.get("difficulty_level")])
        lv_raw = {str(rel): raw for raw, rel in scale.items()}
        saved = str(draft.get("instructions") or "").strip()
        if saved and not draft.get("posted_parts"):
            md = {"domain": draft.get("domain", ""), "type": draft.get("type", ""),
                  "difficulty_level": draft.get("difficulty_level", ""), "tags": draft.get("tags", "")}
            built = instruction_parts.text(_compose_seed(parts, md, draft.get("question_text", ""),
                                                         draft.get("options_text", ""), draft.get("correct_answer", "")))
            legacy = saved if saved != built.strip() else ""
    return {
        "records": records,
        "coverage": _seed_coverage(element, records),
        "draft": draft,
        "parts": parts,
        "lv_raw": lv_raw,
        "legacy_instructions": legacy,
        "editing_id": f"{draft['question_id']}.{draft['sub_id']}" if draft and draft.get("question_id") != "" else "",
        "errors": errors,
        "warnings": warnings or [],
        "domains": devmode.domain_options(),
        "tags": devmode.tag_options(),
        "models": devmode.model_choices(status, store, "generation"),
    }


CANDIDATE_FILTERS = [
    ("open", "Needs review"),
    ("accepted", "Accepted"),
    ("rejected", "Rejected"),
    ("missed", "Solver missed"),
    ("error", "Filler error"),
    ("all", "All"),
]


def _candidate_context_notes(store: DatasetStore, element: str, records: list[dict]) -> None:
    """Add to each candidate record what the model was asked for (``asked``: the instructions stored for exactly its
    target domain, type, level and perspective, else the prompt saved with it) and the example seed of the same type
    and level it was written from (``seed_example``)."""
    try:
        instructions = store.load_dataset("generated_seeds", element).instruction_set
    except Exception:
        instructions = None
    try:
        seeds = store.records("example_seeds", element, dataframe_type="example_questions", limit=2000)
    except Exception:
        seeds = []
    by_kind: dict = {}
    for s in seeds:
        by_kind.setdefault((s.metadata.type, int(s.metadata.difficulty_level)), s.question_text)
    for r in records:
        md = r.get("metadata") or {}
        asked = None
        if instructions is not None and not instructions.empty:
            want = {k: md.get(k, "") for k in ("domain", "type", "difficulty_level", "tags")}
            try:
                hit = instructions.loc[store._metadata_mask(instructions, want), "instructions"]
                asked = str(hit.iloc[0]) if len(hit) else None
            except (KeyError, ValueError):
                asked = None
        if not asked:
            prompt = (r.get("review") or {}).get("prompt") or ""
            asked = prompt.strip() or None
        r["asked"] = asked
        try:
            r["seed_example"] = by_kind.get((md.get("type"), int(md.get("difficulty_level"))))
        except (TypeError, ValueError):
            r["seed_example"] = None


def _candidates_context(
    store: DatasetStore,
    element: str,
    status: dict,
    selected_domains: list[str] | None = None,
    errors: list[str] | None = None,
) -> dict:
    candidate_domains = [{"name": d, "label": DOMAINS[d]} for d in DOMAINS if d != "base"]
    selected_domains = selected_domains or [d["name"] for d in candidate_domains]
    summary = None
    targets = []
    if store.has_dataset("example_seeds", element):
        try:
            summary = example_seed_summary(store, element)
            targets = build_target_grid(store, element, domains=selected_domains)
        except Exception:
            summary = None
    records = _decorate_generated_records(generated_seed_records(store, element, limit=2000))
    for r in records:
        review_status = r.get("review", {}).get("status", "draft")
        r["groups"] = ["all"]
        if review_status in ("draft", "needs_edit"):
            r["groups"].append("open")
        elif review_status in ("accepted", "rejected"):
            r["groups"].append(review_status)
        if r["validation_status"] == "failed":
            r["groups"].append("missed")
        elif r["validation_status"] == "error":
            r["groups"].append("error")
    _candidate_context_notes(store, element, records)
    counts = {key: sum(key in r["groups"] for r in records) for key, _ in CANDIDATE_FILTERS}
    return {
        "summary": summary,
        "candidate_domains": candidate_domains,
        "selected_domains": selected_domains,
        "targets": targets,
        "records": records,
        "filters": [{"key": k, "label": label, "count": counts[k]} for k, label in CANDIDATE_FILTERS],
        "default_filter": "open" if counts["open"] else "all",
        "validation_summary": _generated_validation_summary(records),
        "gen_models": devmode.model_choices(status, store, "generation"),
        "solve_models": devmode.model_choices(status, store, "solve"),
        "has_filler": element in function_registry.functions,
        "shows_options": uses_template_options(element),
        "gen_job": _gen_job(element),
        "check_job": _gen_job(element, "check"),
        "job_running": _any_job_running(element),
        "errors": errors or [],
    }


def _template_list(element: str) -> dict:
    """The first 25 templates; the page fetches more, and searches, through /developer/e/{el}/template-rows."""
    try:
        rows, total, facets = devmode.template_page(element, 0, "", 25)
        error = None
    except Exception as exc:  # templates missing or unloadable: say why
        rows, total, facets, error = [], 0, {}, f"{type(exc).__name__}: {exc}"
    in_file = (load_manifest().get("elements", {}).get(element) or {}).get("template_question_count")
    return {"rows": rows, "total": total, "facets": facets, "list_error": error,
            "left_out": max(0, (in_file or 0) - total) if not error else 0, "in_file": in_file}


def _templates_context(store: DatasetStore, element: str, status: dict, errors: list[str] | None = None) -> dict:
    summary = None
    if has_accepted_generated_seeds(store, element):
        summary = template_generation_summary(store, element)
    has_stored = store.has_dataset("templates", element)
    in_manifest = element in load_manifest().get("elements", {})
    listing = _template_list(element) if (in_manifest or has_stored) else {"rows": [], "total": 0, "list_error": None}
    return {
        "summary": summary,
        "models": devmode.model_choices(status, store, "generation"),
        "errors": errors or [],
        **listing,
    }


def _review_context(element: str, errors: list[str] | None = None) -> dict:
    in_manifest = element in load_manifest().get("elements", {})
    record = review._record(element)
    sample, sample_error = {"picks": [], "total": 0, "n": 0}, None
    if in_manifest:
        try:
            sample = review.review_sample(element)
        except Exception as exc:
            sample_error = f"{type(exc).__name__}: {exc}"
    marks = review.marks_of(record)
    keys = [p["key"] for p in sample["picks"]]
    st = review.standing(marks, keys)
    read = record.get("sample") if isinstance(record.get("sample"), dict) and record["sample"].get("seed") == review.REVIEW_SEED else {}
    return {"record": record, "errors": errors or [], "sample": sample["picks"], "sample_error": sample_error,
            "total": sample["total"], "n": st["n"], "marks": {k: marks[k] for k in keys if k in marks},
            "n_ok": st["ok"], "n_problems": st["problems"], "n_marked": st["marked"], "bound": st["bound"],
            "read_by": read.get("read_by"), "read_at": read.get("read_at"),
            "rule": {"share": review.SAMPLE_SHARE, "min": review.SAMPLE_MIN, "max": review.SAMPLE_MAX},
            "list_error": None}


def create_app() -> FastAPI:
    app = FastAPI(title="STEER Generation")
    templates.env.globals["no_provider_text"] = NO_PROVIDER
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.on_event("startup")
    def warm_templates() -> None:
        """Load the largest elements' templates in the background, so the first visitor to one does not wait for
        it (shared by the API, the element pages and developer mode)."""
        if os.environ.get("AUTOSTEER_WARM", "1") == "0":
            return
        threading.Thread(target=lambda: _api_sampler(load_manifest()).warm(), name="warm-templates", daemon=True).start()

    @app.get("/api/elements")
    def api_elements(store: DatasetStore = Depends(get_dataset_store)) -> dict:
        elements = [
            {
                "element_name": element["element_name"],
                "display_name": element["display_name"],
                "sampleable": True,
                "summary": element["kinds"],
                "manifest": element["manifest"],
            }
            for element in _sampleable_summaries(store)
        ]
        return {"elements": elements}

    @app.get("/api/elements/{element_name}")
    def api_element(element_name: str, store: DatasetStore = Depends(get_dataset_store)) -> dict:
        _ensure_sampleable(element_name)
        manifest = load_manifest()
        entry = public_entry(element_name, manifest.get("elements", {}).get(element_name))
        return {
            **store.summarize_element(element_name),
            "display_name": display_name_for_element(element_name),
            "manifest": entry,
            "categories": element_category_summary(element_name, entry),
        }

    @app.get("/api/datasets/{kind}/{element_name}/records")
    def api_records(
        kind: str,
        element_name: str,
        dataframe_type: str = Query(default="questions", pattern="^(questions|example_questions)$"),
        limit: int = Query(default=50, ge=1, le=500),
        store: DatasetStore = Depends(get_dataset_store),
    ) -> dict:
        if kind in {"example_seeds", "generated_seeds"}:
            _ensure_seed_authorable(element_name)
        elif kind == "templates" and store.has_dataset("templates", element_name):
            pass
        else:
            _ensure_sampleable(element_name)
        if kind not in DATASET_KINDS:
            raise HTTPException(status_code=404, detail=f"Unknown dataset kind: {kind}")
        try:
            records = store.records(kind, element_name, dataframe_type=dataframe_type, limit=limit)
        except FileNotFoundError as exc:
            if kind == "example_seeds":
                records = []
            else:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"records": [record.model_dump() for record in records]}

    @app.get("/api/sample")
    def api_sample_get(
        element_name: str | None = None,
        domain: str | None = None,
        type: str | None = None,
        difficulty_level: str | None = None,
        tags: str | None = None,
        n: int | None = Query(None, ge=1, le=API_MAX_N),
        seed: int | None = Query(None, ge=0, le=SEED_MAX),
        template_id: str | None = None,
    ) -> dict:
        config = SamplerConfig(
            element_name=element_name,
            domain=domain,
            type=type,
            difficulty_level=difficulty_level,
            tags=tags,
        )
        if n is None and seed is None and template_id is None:
            return _sample(config)
        return _sample_rows(config, n, seed, template_id)

    @app.post("/api/sample")
    def api_sample_post(request: SamplerRequest) -> dict:
        config = _sampler_config_from_request(request)
        if request.n is None and request.seed is None and not request.template_id:
            return _sample(config)
        return _sample_rows(config, request.n, request.seed, request.template_id or None)

    def _render_step(request: Request, store: DatasetStore, element: str, step: str, status_code: int = 200, **extra):
        return app.state.render_step(request, store, element, step, status_code=status_code, **extra)

    def _authorable(store: DatasetStore, element_name: str) -> str:
        normalized = _normalize_element_name(element_name)
        _ensure_seed_authorable(normalized, store)
        return normalized

    @app.post("/generated-seeds/generate", include_in_schema=False)
    @app.post("/developer/e/{element}/candidates/generate", include_in_schema=False)
    def generate_generated_seeds(
        request: Request,
        element_name: str = Form(...),
        domains: list[str] = Form(default=[]),
        ai_model: str = Form(""),
        background: str = Form(""),
        store: DatasetStore = Depends(get_dataset_store),
    ):
        _ensure_local_authoring()
        normalized_element_name = _authorable(store, element_name)
        status = _developer_status()
        errors = []
        if not status["ai_configured"]:
            errors.append(NO_PROVIDER)
        if _any_job_running(normalized_element_name):
            errors.append("Candidates are being generated or checked for this element; wait for that to finish.")
        selected_domains = domains or [domain for domain in DOMAINS if domain != "base"]
        targets = build_target_grid(store, normalized_element_name, domains=selected_domains)
        if not targets:
            errors.append("No valid generation targets were found for this element.")
        if errors:
            return _render_step(request, store, normalized_element_name, "candidates", 400,
                                errors=errors, selected_domains=selected_domains)

        provider, model = devmode.resolve_model(ai_model, status)
        if background:  # from the page: run in a thread and show progress there
            _start_job(normalized_element_name, "generate", len(targets), model)
            threading.Thread(target=_run_gen_job, args=(store, normalized_element_name, targets, provider, model),
                             name=f"candidates-{normalized_element_name}", daemon=True).start()
            return RedirectResponse(url=devmode.step_url(normalized_element_name, "candidates", {}), status_code=303)
        generated_count = 0
        for target in targets:
            try:
                generate_generated_seed(
                    store=store,
                    element_name=normalized_element_name,
                    target_metadata=target,
                    model=model,
                    provider=provider,
                )
                generated_count += 1
            except Exception as exc:
                errors.append(f"{target}: {exc}")
        success = (
            f"Generated {generated_count} seed{'' if generated_count == 1 else 's'} for "
            f"{display_name_for_element(normalized_element_name)}." if generated_count else ""
        )
        return RedirectResponse(
            url=devmode.step_url(normalized_element_name, "candidates", {"success": success, "errors": errors}),
            status_code=303,
        )

    @app.get("/developer/e/{element}/candidates/progress", include_in_schema=False)
    def generation_progress(element: str, kind: str = "generate") -> dict:
        _ensure_local_authoring()
        return _gen_job(_normalize_element_name(element), kind) or {"running": False, "total": 0}

    @app.post("/generated-seeds/validate", include_in_schema=False)
    @app.post("/developer/e/{element}/candidates/check", include_in_schema=False)
    def validate_generated_seeds(
        request: Request,
        element_name: str = Form(...),
        ai_model: str = Form(""),
        runs_per_seed: int = Form(3),
        background: str = Form(""),
        store: DatasetStore = Depends(get_dataset_store),
    ):
        _ensure_local_authoring()
        normalized_element_name = _authorable(store, element_name)
        status = _developer_status()
        errors = []
        if not status["ai_configured"]:
            errors.append(NO_PROVIDER)
        if runs_per_seed < 1 or runs_per_seed > 10:
            errors.append("Runs per seed must be between 1 and 10.")
        if _any_job_running(normalized_element_name):
            errors.append("Candidates are being generated or checked for this element; wait for that to finish.")
        if errors:
            return _render_step(request, store, normalized_element_name, "candidates", 400, errors=errors)

        provider, model = devmode.resolve_model(ai_model, status)
        if background:  # from the page: run in a thread and show progress there
            job = _start_job(normalized_element_name, "check", 0, model)
            threading.Thread(target=_run_check_job, args=(store, normalized_element_name, job, provider, model, runs_per_seed),
                             name=f"solve-check-{normalized_element_name}", daemon=True).start()
            return RedirectResponse(url=devmode.step_url(normalized_element_name, "candidates", {}), status_code=303)
        try:
            summary = run_generated_seed_validation(
                store=store,
                element_name=normalized_element_name,
                model=model,
                provider=provider,
                runs_per_seed=runs_per_seed,
                statuses=("draft", "accepted", "needs_edit"),
            )
        except Exception as exc:
            return _render_step(request, store, normalized_element_name, "candidates", 400, errors=[str(exc)])

        if summary["validated"] == 0:
            errors.append("There are no draft, accepted, or needs-edit generated seeds to check.")
        message = (
            f"Checked {summary['validated']} generated seed"
            f"{'' if summary['validated'] == 1 else 's'}: "
            f"{summary['passed']} solved, {summary['failed']} missed, {summary['errors']} filler errors."
            if summary["validated"]
            else ""
        )
        return RedirectResponse(
            url=devmode.step_url(normalized_element_name, "candidates", {"success": message, "errors": errors}),
            status_code=303,
        )

    def _candidate_url(element: str, question_id: int, sub_id: int) -> str:
        return devmode.step_url(element, "candidates") + f"#c-{question_id}-{sub_id}"

    @app.post("/generated-seeds/check-one", include_in_schema=False)
    @app.post("/developer/e/{element}/candidates/check-one", include_in_schema=False)
    def check_one_generated_seed(
        element_name: str = Form(...),
        question_id: int = Form(...),
        sub_id: int = Form(0),
        runs_per_seed: int = Form(3),
        ai_model: str = Form(""),
        store: DatasetStore = Depends(get_dataset_store),
    ) -> RedirectResponse:
        _ensure_local_authoring()
        normalized_element_name = _authorable(store, element_name)
        status = _developer_status()
        if not status["ai_configured"]:
            raise HTTPException(status_code=400, detail=NO_PROVIDER)
        if runs_per_seed < 1 or runs_per_seed > 10:
            raise HTTPException(status_code=400, detail="Runs per seed must be between 1 and 10.")

        provider, model = devmode.resolve_model(
            ai_model or devmode.model_choices(status, store, "solve")["default"], status)
        try:
            run_single_generated_seed_validation(
                store=store,
                element_name=normalized_element_name,
                question_id=question_id,
                sub_id=sub_id,
                model=model,
                provider=provider,
                runs_per_seed=runs_per_seed,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(url=_candidate_url(normalized_element_name, question_id, sub_id), status_code=303)

    @app.post("/generated-seeds/review", include_in_schema=False)
    @app.post("/developer/e/{element}/candidates/review", include_in_schema=False)
    def review_generated_seed(
        element_name: str = Form(...),
        question_id: int = Form(...),
        sub_id: int = Form(0),
        status: str = Form(...),
        notes: str = Form(""),
        store: DatasetStore = Depends(get_dataset_store),
    ) -> RedirectResponse:
        _ensure_local_authoring()
        normalized_element_name = _authorable(store, element_name)
        try:
            set_review_status(
                store,
                normalized_element_name,
                question_id=question_id,
                sub_id=sub_id,
                status=status,
                notes=notes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(url=_candidate_url(normalized_element_name, question_id, sub_id), status_code=303)

    @app.post("/generated-seeds/edit", include_in_schema=False)
    @app.post("/developer/e/{element}/candidates/edit", include_in_schema=False)
    def edit_generated_seed(
        element_name: str = Form(...),
        question_id: int = Form(...),
        sub_id: int = Form(0),
        question_text: str = Form(...),
        options_text: str = Form(""),
        correct_answer: str = Form(""),
        store: DatasetStore = Depends(get_dataset_store),
    ) -> RedirectResponse:
        _ensure_local_authoring()
        normalized_element_name = _authorable(store, element_name)
        try:
            update_generated_seed(
                store,
                normalized_element_name,
                question_id=question_id,
                sub_id=sub_id,
                question_text=question_text,
                options_text=options_text,
                correct_answer=correct_answer,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse(url=_candidate_url(normalized_element_name, question_id, sub_id), status_code=303)

    @app.post("/developer/e/{element}/candidates/bulk", include_in_schema=False)
    def bulk_generated_seeds(
        element_name: str = Form(...),
        action: str = Form(...),
        ids: list[str] = Form(default=[]),
        store: DatasetStore = Depends(get_dataset_store),
    ) -> RedirectResponse:
        """Accept, reject, take back to needs review (draft) or delete several candidates at once; ids are 'qid.sid'."""
        _ensure_local_authoring()
        normalized_element_name = _authorable(store, element_name)
        if action not in ("accepted", "rejected", "draft", "delete"):
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
        done, errors = 0, []
        for rid in ids:
            qid, _, sid = rid.partition(".")
            try:
                if action == "delete":
                    delete_generated_seed(store, normalized_element_name, question_id=int(qid), sub_id=int(sid or 0))
                else:
                    set_review_status(store=store, element_name=normalized_element_name, question_id=int(qid),
                                      sub_id=int(sid or 0), status=action)
                done += 1
            except (ValueError, FileNotFoundError) as exc:
                errors.append(f"{rid}: {exc}")
        verb = {"accepted": "Accepted", "rejected": "Rejected", "draft": "Took back to needs review", "delete": "Deleted"}[action]
        return RedirectResponse(
            url=devmode.step_url(normalized_element_name, "candidates",
                                 {"success": f"{verb} {done} candidate{'' if done == 1 else 's'}." if done else "",
                                  "errors": errors}),
            status_code=303,
        )

    @app.post("/generated-seeds/delete", include_in_schema=False)
    @app.post("/developer/e/{element}/candidates/delete", include_in_schema=False)
    def delete_generated_seed_route(
        element_name: str = Form(...),
        question_id: int = Form(...),
        sub_id: int = Form(0),
        store: DatasetStore = Depends(get_dataset_store),
    ) -> RedirectResponse:
        _ensure_local_authoring()
        normalized_element_name = _authorable(store, element_name)
        try:
            delete_generated_seed(store, normalized_element_name, question_id=question_id, sub_id=sub_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return RedirectResponse(
            url=devmode.step_url(normalized_element_name, "candidates",
                                 {"success": f"Deleted candidate {question_id}{'.' + str(sub_id) if sub_id else ''}."}),
            status_code=303,
        )

    @app.post("/templates/generate", include_in_schema=False)
    @app.post("/developer/e/{element}/templates/generate", include_in_schema=False)
    def generate_templates_page(
        request: Request,
        element_name: str = Form(...),
        ai_model: str = Form(""),
        templates_per_target: int = Form(1),
        store: DatasetStore = Depends(get_dataset_store),
    ):
        _ensure_local_authoring()
        normalized_element_name = _authorable(store, element_name)
        status = _developer_status()
        errors = []
        if not has_accepted_generated_seeds(store, normalized_element_name):
            errors.append("Accept at least one seed candidate in Seed candidates before generating templates.")
        if not status["ai_configured"]:
            errors.append(NO_PROVIDER)
        if templates_per_target < 1 or templates_per_target > 25:
            errors.append("Templates per target must be between 1 and 25.")
        if errors:
            return _render_step(request, store, normalized_element_name, "templates", 400, errors=errors)

        provider, model = devmode.resolve_model(ai_model, status)
        try:
            result = generate_template_batch(
                store=store,
                element_name=normalized_element_name,
                model=model,
                provider=provider,
                templates_per_target=templates_per_target,
            )
        except Exception as exc:
            return _render_step(request, store, normalized_element_name, "templates", 400, errors=[str(exc)])

        message = (
            f"Generated {result['generated']} template"
            f"{'' if result['generated'] == 1 else 's'} for {display_name_for_element(normalized_element_name)}."
            if result["generated"]
            else ""
        )
        return RedirectResponse(
            url=devmode.step_url(normalized_element_name, "templates", {"success": message, "errors": result["errors"]}),
            status_code=303,
        )

    @app.post("/api/seeds/compose")
    def seed_compose(
        element_name: str = Form(""),
        question_text: str = Form(""),
        options_text: str = Form(""),
        correct_answer: str = Form(""),
        domain: str = Form(""),
        type: str = Form(""),
        difficulty: str = Form(""),
        tags: str = Form(""),
        part_task: str = Form(""),
        part_type: str = Form(""),
        part_level: str = Form(""),
        part_domain: str = Form(""),
        part_tag_keys: list[str] = Form([]),
        part_tag_texts: list[str] = Form([]),
        store: DatasetStore = Depends(get_dataset_store),
    ) -> dict:
        """The instructions the model gets for the seed being edited, built from the parts in the form."""
        _ensure_local_authoring()
        element = _normalize_element_name(element_name)
        try:
            raw = _seed_raw_difficulty(store, element, difficulty) if difficulty.strip() else ""
        except (TypeError, ValueError):
            raw = ""
        parts = _parts_with_form(_seed_parts(store, element), type_key=type.strip(), raw_level=raw, domain=domain.strip(),
                                 task=part_task, type_text=part_type, level_text=part_level, domain_text=part_domain,
                                 tag_keys=part_tag_keys, tag_texts=part_tag_texts)
        md = {"domain": domain.strip(), "type": type.strip(), "difficulty_level": raw, "tags": ";".join(_split_tags(tags))}
        segments = _compose_seed(parts, md, question_text, options_text, correct_answer)
        return {"segments": segments, "text": instruction_parts.text(segments)}

    @app.post("/api/seeds/instructions")
    def seed_instructions(
        question_text: str = Form(""),
        options_text: str = Form(""),
        correct_answer: str = Form(""),
        domain: str = Form(""),
        type: str = Form(""),
        difficulty_level: int = Form(0),
        tags: str = Form(""),
        instructions: str = Form(""),
    ) -> dict:
        _ensure_local_authoring()
        option_texts = [line.strip() for line in options_text.splitlines() if line.strip()]
        normalized_tags = ";".join(part.strip() for part in re.split(r"[;,]", tags) if part.strip())
        task_description = instructions.split(". ")[0].strip() if instructions.strip() else ""
        generated = _build_seed_instructions(
            {
                "domain": domain.strip(),
                "type": type.strip(),
                "difficulty_level": difficulty_level,
                "tags": normalized_tags,
            },
            num_options=len(option_texts),
            include_answer=bool(option_texts and correct_answer.strip()),
            variables=_template_variables(question_text, options_text, correct_answer),
            task_description=task_description,
        )
        return {"instructions": generated}

    @app.post("/api/seeds/verify")
    def seed_verify(
        question_text: str = Form(""),
        options_text: str = Form(""),
        correct_answer: str = Form(""),
        instructions: str = Form(""),
        ai_model: str = Form(""),
    ) -> dict:
        _ensure_local_authoring()
        status = _developer_status()
        if not status["ai_configured"]:
            raise HTTPException(
                status_code=400,
                detail=NO_PROVIDER,
            )

        draft = _seed_draft_question(question_text, options_text, correct_answer)
        model = devmode.resolve_model(ai_model, status)[1]
        has_options = bool(draft["options"])
        if has_options:
            output_contract = (
                "Return only JSON with keys: question_text, options, notes. "
                "options must be a list of objects with option_id, option_text, and correct_answer."
            )
        else:
            output_contract = (
                "Return only JSON with keys: question_text, notes. "
                "Do not include an options key, an answer key, or any option text."
            )
        system_prompt = (
            "You are checking a STEER-ME seed example for template authoring. "
            "Do not answer the original question. Generate one analogous seed example "
            "that follows the provided authoring instructions and preserves the same "
            "placeholder style. "
            + output_contract
            + " notes should briefly mention any authoring concerns."
        )
        prompt = (
            "Authoring instructions:\n"
            + (instructions.strip() or "(none provided)")
            + "\n\nSeed example to check:\n"
            + _json_to_mcq(draft)
        )
        try:
            from autosteer.json_utils import parse_model_json
            from autosteer.llm_client import LanguageModelClient

            provider, model = devmode.resolve_model(ai_model, status)
            result_text = LanguageModelClient(provider_name=provider).make_request(
                user_prompts=prompt,
                system_prompts=system_prompt,
                request_type="completion",
                return_type="string",
                response_format={"type": "json_object"},
                model=model,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result_json = None
        try:
            result_json = parse_model_json(result_text)
        except Exception:
            result_json = None

        return {
            "model": model,
            "prompt": prompt,
            "result_text": result_text,
            "result_json": result_json,
        }

    @app.post("/seeds")
    @app.post("/developer/e/{element}/seeds")
    def create_seed(
        request: Request,
        element_name: str = Form(...),
        question_id: str = Form(""),
        sub_id: int = Form(0),
        saved_element_name: str = Form(""),
        question_text: str = Form(...),
        domain: str = Form(...),
        type: str = Form(...),
        difficulty_level: str = Form(""),
        difficulty: str = Form(""),
        tags: str = Form(""),
        options_text: str = Form(""),
        correct_answer: str = Form(""),
        instructions: str = Form(""),
        ai_model: str = Form(""),
        parts_form: str = Form(""),
        part_task: str = Form(""),
        part_type: str = Form(""),
        part_level: str = Form(""),
        part_domain: str = Form(""),
        part_tag_keys: list[str] = Form([]),
        part_tag_texts: list[str] = Form([]),
        store: DatasetStore = Depends(get_dataset_store),
    ):
        _ensure_local_authoring()
        difficulty_error = None
        if difficulty.strip():  # the editor's level (1 = easiest); stored as the element's raw level
            try:
                difficulty_level = str(_seed_raw_difficulty(store, _normalize_element_name(element_name), difficulty))
            except (TypeError, ValueError) as exc:
                difficulty_error = str(exc) if "Difficulty" in str(exc) else "Difficulty must be a whole number."
                difficulty_level = "0"
        parts_errors, form_parts = [], None
        if parts_form:  # the editor sends the instruction parts; the instructions are built from them
            element_key = _normalize_element_name(element_name)
            form_parts = _parts_with_form(_seed_parts(store, element_key), type_key=type.strip(), raw_level=difficulty_level,
                                          domain=domain.strip(), task=part_task, type_text=part_type, level_text=part_level,
                                          domain_text=part_domain, tag_keys=part_tag_keys, tag_texts=part_tag_texts)
            md = {"domain": domain.strip(), "type": type.strip(), "difficulty_level": difficulty_level,
                  "tags": ";".join(_split_tags(tags))}
            segments = _compose_seed(form_parts, md, question_text, options_text, correct_answer)
            instructions = instruction_parts.text(segments)
            parts_errors = [f"Instructions: write the {'element' if s['kind'] == 'task' else s['kind']} part ({s['text']})."
                            for s in instruction_parts.missing(segments)]
        existing_question_id = int(question_id) if question_id.strip().isdigit() else None
        normalized_saved_element_name = _normalize_element_name(saved_element_name) if saved_element_name else ""
        draft = _seed_draft(
            element_name=element_name,
            question_id=question_id,
            sub_id=sub_id,
            saved_element_name=saved_element_name or element_name,
            question_text=question_text,
            domain=domain,
            type=type,
            difficulty_level=difficulty_level,
            tags=tags,
            options_text=options_text,
            correct_answer=correct_answer,
            instructions=instructions,
            ai_model=ai_model,
        )
        if parts_form:
            draft["posted_parts"] = {"task": part_task, "type": part_type, "level": part_level, "domain": part_domain,
                                     "tags": dict(zip(part_tag_keys, part_tag_texts))}
        normalized_element_name, parsed_difficulty, errors, warnings = _validate_seed_draft(
            element_name=element_name,
            question_text=question_text,
            domain=domain,
            type=type,
            difficulty_level=difficulty_level,
            options_text=options_text,
            correct_answer=correct_answer,
            instructions=instructions,
            store=store,
        )
        if not normalized_element_name:
            raise HTTPException(status_code=400, detail="Element name is required.")
        if difficulty_error:
            errors.append(difficulty_error)
        errors.extend(parts_errors)
        if (
            existing_question_id is not None
            and normalized_saved_element_name
            and normalized_element_name != normalized_saved_element_name
        ):
            errors.append("Saved seeds cannot be moved to another element from the edit form. Start a new seed instead.")
        if errors:
            return _render_step(request, store, normalized_element_name, "seeds", 400,
                                draft=draft, errors=errors, warnings=warnings)

        metadata = Metadata(
            domain=domain.strip(),
            type=type.strip(),
            difficulty_level=parsed_difficulty,
            tags=tags.strip(),
        )
        duplicates = store.seed_duplicates(
            element_name=normalized_element_name,
            question_text=question_text,
            metadata=metadata,
            question_id=existing_question_id,
            sub_id=sub_id,
        )
        exact_duplicates = [duplicate for duplicate in duplicates if duplicate["exact"]]
        if exact_duplicates:
            errors.extend(
                [
                    (
                        f"This question matches seed id {duplicate['question_id']}"
                        f"{'.' + str(duplicate['sub_id']) if duplicate['sub_id'] else ''} exactly. "
                        "Edit that seed instead of saving another copy."
                    )
                    for duplicate in exact_duplicates
                ]
            )
        else:
            warnings.extend(
                [
                    (
                        f"Likely duplicate of seed id {duplicate['question_id']}"
                        f"{'.' + str(duplicate['sub_id']) if duplicate['sub_id'] else ''} "
                        f"({int(duplicate['similarity'] * 100)}% text match)."
                    )
                    for duplicate in duplicates
                ]
            )
        if errors:
            return _render_step(request, store, normalized_element_name, "seeds", 400,
                                draft=draft, errors=errors, warnings=warnings)
        if form_parts is not None:
            tag_map = dict(zip(part_tag_keys, part_tag_texts))
            instruction_parts.save(store.data_root, normalized_element_name, task=part_task, type_key=type,
                                   type_text=part_type, level=difficulty_level, level_text=part_level,
                                   domain_key=domain, domain_text=part_domain, tags=tag_map)
        _, saved_question_id = store.save_seed(
            element_name=normalized_element_name,
            question_text=question_text.strip(),
            metadata=metadata,
            options=_parse_options(options_text, correct_answer),
            explanation="",
            instructions=instructions.strip() or None,
            question_id=existing_question_id,
            sub_id=sub_id,
        )
        return RedirectResponse(
            url=devmode.step_url(
                normalized_element_name, "seeds",
                {"success": f"Seed id {saved_question_id} saved for {display_name_for_element(normalized_element_name)}.",
                 "warnings": warnings},
                edit=f"{saved_question_id}.{sub_id}",
            ),
            status_code=303,
        )

    @app.post("/seeds/delete")
    @app.post("/developer/e/{element}/seeds/delete")
    def delete_seed(
        element_name: str = Form(...),
        question_id: int = Form(...),
        sub_id: int = Form(0),
        store: DatasetStore = Depends(get_dataset_store),
    ) -> RedirectResponse:
        _ensure_local_authoring()
        normalized_element_name = _authorable(store, element_name)
        try:
            store.delete_seed(normalized_element_name, question_id=question_id, sub_id=sub_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return RedirectResponse(
            url=devmode.step_url(normalized_element_name, "seeds",
                                 {"success": f"Deleted seed id {question_id}{'.' + str(sub_id) if sub_id else ''}."}),
            status_code=303,
        )

    @app.post("/developer/e/{element}/seeds/delete-many", include_in_schema=False)
    def delete_seeds(
        element_name: str = Form(...),
        ids: list[str] = Form(...),
        store: DatasetStore = Depends(get_dataset_store),
    ) -> RedirectResponse:
        _ensure_local_authoring()
        normalized_element_name = _authorable(store, element_name)
        keys = []
        for rid in ids:
            qid, _, sid = rid.partition(".")
            try:
                keys.append((int(qid), int(sid or 0)))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Not a seed id: {rid}")
        for qid, sid in sorted(set(keys), reverse=True):  # last first, in case ids shift
            try:
                store.delete_seed(normalized_element_name, question_id=qid, sub_id=sid)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        n = len(set(keys))
        return RedirectResponse(
            url=devmode.step_url(normalized_element_name, "seeds", {"success": f"Deleted {n} seed{'' if n == 1 else 's'}."}),
            status_code=303,
        )

    def step_context(store: DatasetStore, element: str, step: str, request: Request, status: dict, **extra) -> dict:
        if step == "seeds":
            return _seeds_context(store, element, request, status, **extra)
        if step == "candidates":
            return _candidates_context(store, element, status, **extra)
        if step == "templates":
            return _templates_context(store, element, status, **extra)
        return _review_context(element, **extra)

    review.install(app, templates, public_mode=_public_web_mode)
    devmode.install(app, templates, public_mode=_public_web_mode, status=lambda: _developer_status(),
                    step_context=step_context)

    site.install(app, templates, public_mode=_public_web_mode)
    return app


app = create_app()
