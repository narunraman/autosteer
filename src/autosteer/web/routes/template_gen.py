"""Developer mode, step 3: generating templates from accepted seed candidates."""
from __future__ import annotations


from fastapi import Depends, Form, Request
from fastapi.responses import RedirectResponse

from autosteer.services.dataset_store import DatasetStore, get_dataset_store
from autosteer.services.template_generation_service import (
    has_accepted_generated_seeds,
    generate_template_batch,
)
from autosteer.taxonomy import (
    display_name_for_element,
)
from autosteer.web import devmode

from fastapi import APIRouter

from autosteer.web import common

router = APIRouter()


@router.post("/templates/generate", include_in_schema=False)
@router.post("/developer/e/{element}/templates/generate", include_in_schema=False)
def generate_templates_page(
    request: Request,
    element_name: str = Form(...),
    ai_model: str = Form(""),
    templates_per_target: int = Form(1),
    store: DatasetStore = Depends(get_dataset_store),
):
    common._ensure_local_authoring()
    normalized_element_name = common._authorable(store, element_name)
    status = common._developer_status()
    errors = []
    if not has_accepted_generated_seeds(store, normalized_element_name):
        errors.append("Accept at least one seed candidate in Seed candidates before generating templates.")
    if not status["ai_configured"]:
        errors.append(common.NO_PROVIDER)
    if templates_per_target < 1 or templates_per_target > 25:
        errors.append("Templates per target must be between 1 and 25.")
    if errors:
        return common._render_step(request, store, normalized_element_name, "templates", 400, errors=errors)

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
        return common._render_step(request, store, normalized_element_name, "templates", 400, errors=[str(exc)])

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
