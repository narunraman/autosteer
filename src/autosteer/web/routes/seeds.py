"""Developer mode, step 1: composing, verifying, saving and deleting example seeds (and the seeds API)."""
from __future__ import annotations

import re

from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse

from autosteer import instruction_parts
from autosteer.schemas import Metadata
from autosteer.services.dataset_store import DatasetStore, get_dataset_store
from autosteer.taxonomy import (
    display_name_for_element,
)
from autosteer.web import devmode

from fastapi import APIRouter

from autosteer.web import common

router = APIRouter()


@router.post("/api/seeds/compose")
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
    common._ensure_local_authoring()
    element = common._normalize_element_name(element_name)
    try:
        raw = common._seed_raw_difficulty(store, element, difficulty) if difficulty.strip() else ""
    except (TypeError, ValueError):
        raw = ""
    parts = common._parts_with_form(common._seed_parts(store, element), type_key=type.strip(), raw_level=raw, domain=domain.strip(),
                             task=part_task, type_text=part_type, level_text=part_level, domain_text=part_domain,
                             tag_keys=part_tag_keys, tag_texts=part_tag_texts)
    md = {"domain": domain.strip(), "type": type.strip(), "difficulty_level": raw, "tags": ";".join(common._split_tags(tags))}
    segments = common._compose_seed(parts, md, question_text, options_text, correct_answer)
    return {"segments": segments, "text": instruction_parts.text(segments)}


@router.post("/api/seeds/instructions")
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
    common._ensure_local_authoring()
    option_texts = [line.strip() for line in options_text.splitlines() if line.strip()]
    normalized_tags = ";".join(part.strip() for part in re.split(r"[;,]", tags) if part.strip())
    task_description = instructions.split(". ")[0].strip() if instructions.strip() else ""
    generated = common._build_seed_instructions(
        {
            "domain": domain.strip(),
            "type": type.strip(),
            "difficulty_level": difficulty_level,
            "tags": normalized_tags,
        },
        num_options=len(option_texts),
        include_answer=bool(option_texts and correct_answer.strip()),
        variables=common._template_variables(question_text, options_text, correct_answer),
        task_description=task_description,
    )
    return {"instructions": generated}


@router.post("/api/seeds/verify")
def seed_verify(
    question_text: str = Form(""),
    options_text: str = Form(""),
    correct_answer: str = Form(""),
    instructions: str = Form(""),
    ai_model: str = Form(""),
) -> dict:
    common._ensure_local_authoring()
    status = common._developer_status()
    if not status["ai_configured"]:
        raise HTTPException(
            status_code=400,
            detail=common.NO_PROVIDER,
        )

    draft = common._seed_draft_question(question_text, options_text, correct_answer)
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
        + common._json_to_mcq(draft)
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


@router.post("/seeds")
@router.post("/developer/e/{element}/seeds")
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
    common._ensure_local_authoring()
    difficulty_error = None
    if difficulty.strip():  # the editor's level (1 = easiest); stored as the element's raw level
        try:
            difficulty_level = str(common._seed_raw_difficulty(store, common._normalize_element_name(element_name), difficulty))
        except (TypeError, ValueError) as exc:
            difficulty_error = str(exc) if "Difficulty" in str(exc) else "Difficulty must be a whole number."
            difficulty_level = "0"
    parts_errors, form_parts = [], None
    if parts_form:  # the editor sends the instruction parts; the instructions are built from them
        element_key = common._normalize_element_name(element_name)
        form_parts = common._parts_with_form(common._seed_parts(store, element_key), type_key=type.strip(), raw_level=difficulty_level,
                                      domain=domain.strip(), task=part_task, type_text=part_type, level_text=part_level,
                                      domain_text=part_domain, tag_keys=part_tag_keys, tag_texts=part_tag_texts)
        md = {"domain": domain.strip(), "type": type.strip(), "difficulty_level": difficulty_level,
              "tags": ";".join(common._split_tags(tags))}
        segments = common._compose_seed(form_parts, md, question_text, options_text, correct_answer)
        instructions = instruction_parts.text(segments)
        parts_errors = [f"Instructions: write the {'element' if s['kind'] == 'task' else s['kind']} part ({s['text']})."
                        for s in instruction_parts.missing(segments)]
    existing_question_id = int(question_id) if question_id.strip().isdigit() else None
    normalized_saved_element_name = common._normalize_element_name(saved_element_name) if saved_element_name else ""
    draft = common._seed_draft(
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
    normalized_element_name, parsed_difficulty, errors, warnings = common._validate_seed_draft(
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
        return common._render_step(request, store, normalized_element_name, "seeds", 400,
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
        return common._render_step(request, store, normalized_element_name, "seeds", 400,
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
        options=common._parse_options(options_text, correct_answer),
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


@router.post("/seeds/delete")
@router.post("/developer/e/{element}/seeds/delete")
def delete_seed(
    element_name: str = Form(...),
    question_id: int = Form(...),
    sub_id: int = Form(0),
    store: DatasetStore = Depends(get_dataset_store),
) -> RedirectResponse:
    common._ensure_local_authoring()
    normalized_element_name = common._authorable(store, element_name)
    try:
        store.delete_seed(normalized_element_name, question_id=question_id, sub_id=sub_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RedirectResponse(
        url=devmode.step_url(normalized_element_name, "seeds",
                             {"success": f"Deleted seed id {question_id}{'.' + str(sub_id) if sub_id else ''}."}),
        status_code=303,
    )


@router.post("/developer/e/{element}/seeds/delete-many", include_in_schema=False)
def delete_seeds(
    element_name: str = Form(...),
    ids: list[str] = Form(...),
    store: DatasetStore = Depends(get_dataset_store),
) -> RedirectResponse:
    common._ensure_local_authoring()
    normalized_element_name = common._authorable(store, element_name)
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
