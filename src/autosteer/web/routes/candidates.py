"""Developer mode, step 2: generating, checking, reviewing and editing seed candidates."""
from __future__ import annotations

import threading

from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse

from autosteer.constants import DOMAINS
from autosteer.services.dataset_store import DatasetStore, get_dataset_store
from autosteer.services.generated_seed_service import (
    build_target_grid,
    delete_generated_seed,
    set_review_status,
    update_generated_seed,
)
from autosteer.services.generated_seed_validation import (
    run_generated_seed_validation,
    run_single_generated_seed_validation,
)
from autosteer.taxonomy import (
    display_name_for_element,
)
from autosteer.web import devmode

from fastapi import APIRouter

from autosteer.web import common

router = APIRouter()


@router.post("/generated-seeds/generate", include_in_schema=False)
@router.post("/developer/e/{element}/candidates/generate", include_in_schema=False)
def generate_generated_seeds(
    request: Request,
    element_name: str = Form(...),
    domains: list[str] = Form(default=[]),
    ai_model: str = Form(""),
    background: str = Form(""),
    store: DatasetStore = Depends(get_dataset_store),
):
    common._ensure_local_authoring()
    normalized_element_name = common._authorable(store, element_name)
    status = common._developer_status()
    errors = []
    if not status["ai_configured"]:
        errors.append(common.NO_PROVIDER)
    if common._any_job_running(normalized_element_name):
        errors.append("Candidates are being generated or checked for this element; wait for that to finish.")
    selected_domains = domains or [domain for domain in DOMAINS if domain != "base"]
    targets = build_target_grid(store, normalized_element_name, domains=selected_domains)
    if not targets:
        errors.append("No valid generation targets were found for this element.")
    if errors:
        return common._render_step(request, store, normalized_element_name, "candidates", 400,
                            errors=errors, selected_domains=selected_domains)

    provider, model = devmode.resolve_model(ai_model, status)
    if background:  # from the page: run in a thread and show progress there
        common._start_job(normalized_element_name, "generate", len(targets), model)
        threading.Thread(target=common._run_gen_job, args=(store, normalized_element_name, targets, provider, model),
                         name=f"candidates-{normalized_element_name}", daemon=True).start()
        return RedirectResponse(url=devmode.step_url(normalized_element_name, "candidates", {}), status_code=303)
    generated_count = 0
    for target in targets:
        try:
            common.generate_generated_seed(
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


@router.get("/developer/e/{element}/candidates/progress", include_in_schema=False)
def generation_progress(element: str, kind: str = "generate") -> dict:
    common._ensure_local_authoring()
    return common._gen_job(common._normalize_element_name(element), kind) or {"running": False, "total": 0}


@router.post("/generated-seeds/validate", include_in_schema=False)
@router.post("/developer/e/{element}/candidates/check", include_in_schema=False)
def validate_generated_seeds(
    request: Request,
    element_name: str = Form(...),
    ai_model: str = Form(""),
    runs_per_seed: int = Form(3),
    background: str = Form(""),
    store: DatasetStore = Depends(get_dataset_store),
):
    common._ensure_local_authoring()
    normalized_element_name = common._authorable(store, element_name)
    status = common._developer_status()
    errors = []
    if not status["ai_configured"]:
        errors.append(common.NO_PROVIDER)
    if runs_per_seed < 1 or runs_per_seed > 10:
        errors.append("Runs per seed must be between 1 and 10.")
    if common._any_job_running(normalized_element_name):
        errors.append("Candidates are being generated or checked for this element; wait for that to finish.")
    if errors:
        return common._render_step(request, store, normalized_element_name, "candidates", 400, errors=errors)

    provider, model = devmode.resolve_model(ai_model, status)
    if background:  # from the page: run in a thread and show progress there
        job = common._start_job(normalized_element_name, "check", 0, model)
        threading.Thread(target=common._run_check_job, args=(store, normalized_element_name, job, provider, model, runs_per_seed),
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
        return common._render_step(request, store, normalized_element_name, "candidates", 400, errors=[str(exc)])

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


@router.post("/generated-seeds/check-one", include_in_schema=False)
@router.post("/developer/e/{element}/candidates/check-one", include_in_schema=False)
def check_one_generated_seed(
    element_name: str = Form(...),
    question_id: int = Form(...),
    sub_id: int = Form(0),
    runs_per_seed: int = Form(3),
    ai_model: str = Form(""),
    store: DatasetStore = Depends(get_dataset_store),
) -> RedirectResponse:
    common._ensure_local_authoring()
    normalized_element_name = common._authorable(store, element_name)
    status = common._developer_status()
    if not status["ai_configured"]:
        raise HTTPException(status_code=400, detail=common.NO_PROVIDER)
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
    return RedirectResponse(url=common._candidate_url(normalized_element_name, question_id, sub_id), status_code=303)


@router.post("/generated-seeds/review", include_in_schema=False)
@router.post("/developer/e/{element}/candidates/review", include_in_schema=False)
def review_generated_seed(
    element_name: str = Form(...),
    question_id: int = Form(...),
    sub_id: int = Form(0),
    status: str = Form(...),
    notes: str = Form(""),
    store: DatasetStore = Depends(get_dataset_store),
) -> RedirectResponse:
    common._ensure_local_authoring()
    normalized_element_name = common._authorable(store, element_name)
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
    return RedirectResponse(url=common._candidate_url(normalized_element_name, question_id, sub_id), status_code=303)


@router.post("/generated-seeds/edit", include_in_schema=False)
@router.post("/developer/e/{element}/candidates/edit", include_in_schema=False)
def edit_generated_seed(
    element_name: str = Form(...),
    question_id: int = Form(...),
    sub_id: int = Form(0),
    question_text: str = Form(...),
    options_text: str = Form(""),
    correct_answer: str = Form(""),
    store: DatasetStore = Depends(get_dataset_store),
) -> RedirectResponse:
    common._ensure_local_authoring()
    normalized_element_name = common._authorable(store, element_name)
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
    return RedirectResponse(url=common._candidate_url(normalized_element_name, question_id, sub_id), status_code=303)


@router.post("/developer/e/{element}/candidates/bulk", include_in_schema=False)
def bulk_generated_seeds(
    element_name: str = Form(...),
    action: str = Form(...),
    ids: list[str] = Form(default=[]),
    store: DatasetStore = Depends(get_dataset_store),
) -> RedirectResponse:
    """Accept, reject, take back to needs review (draft) or delete several candidates at once; ids are 'qid.sid'."""
    common._ensure_local_authoring()
    normalized_element_name = common._authorable(store, element_name)
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


@router.post("/generated-seeds/delete", include_in_schema=False)
@router.post("/developer/e/{element}/candidates/delete", include_in_schema=False)
def delete_generated_seed_route(
    element_name: str = Form(...),
    question_id: int = Form(...),
    sub_id: int = Form(0),
    store: DatasetStore = Depends(get_dataset_store),
) -> RedirectResponse:
    common._ensure_local_authoring()
    normalized_element_name = common._authorable(store, element_name)
    try:
        delete_generated_seed(store, normalized_element_name, question_id=question_id, sub_id=sub_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RedirectResponse(
        url=devmode.step_url(normalized_element_name, "candidates",
                             {"success": f"Deleted candidate {question_id}{'.' + str(sub_id) if sub_id else ''}."}),
        status_code=303,
    )
