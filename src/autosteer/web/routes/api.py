"""The JSON API: /health, /api/elements, /api/datasets/... and /api/sample."""
from __future__ import annotations


from fastapi import Depends, HTTPException, Query

from autosteer.sampler import SamplerConfig, load_manifest
from autosteer.sampler.difficulty import public_entry
from autosteer.sampler.generator import SEED_MAX
from autosteer.schemas import SamplerRequest
from autosteer.services.dataset_store import DATASET_KINDS, DatasetStore, get_dataset_store
from autosteer.taxonomy import (
    display_name_for_element,
    element_category_summary,
)

from fastapi import APIRouter

from autosteer.web import common

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/api/elements")
def api_elements(store: DatasetStore = Depends(get_dataset_store)) -> dict:
    elements = [
        {
            "element_name": element["element_name"],
            "display_name": element["display_name"],
            "sampleable": True,
            "summary": element["kinds"],
            "manifest": element["manifest"],
        }
        for element in common._sampleable_summaries(store)
    ]
    return {"elements": elements}


@router.get("/api/elements/{element_name}")
def api_element(element_name: str, store: DatasetStore = Depends(get_dataset_store)) -> dict:
    common._ensure_sampleable(element_name)
    manifest = load_manifest()
    entry = public_entry(element_name, manifest.get("elements", {}).get(element_name))
    return {
        **store.summarize_element(element_name),
        "display_name": display_name_for_element(element_name),
        "manifest": entry,
        "categories": element_category_summary(element_name, entry),
    }


@router.get("/api/datasets/{kind}/{element_name}/records")
def api_records(
    kind: str,
    element_name: str,
    dataframe_type: str = Query(default="questions", pattern="^(questions|example_questions)$"),
    limit: int = Query(default=50, ge=1, le=500),
    store: DatasetStore = Depends(get_dataset_store),
) -> dict:
    if kind in {"example_seeds", "generated_seeds"}:
        common._ensure_seed_authorable(element_name)
    elif kind == "templates" and store.has_dataset("templates", element_name):
        pass
    else:
        common._ensure_sampleable(element_name)
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


@router.get("/api/sample")
def api_sample_get(
    element_name: str | None = None,
    domain: str | None = None,
    type: str | None = None,
    difficulty_level: str | None = None,
    tags: str | None = None,
    n: int | None = Query(None, ge=1, le=common.API_MAX_N),
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
        return common._sample(config)
    return common._sample_rows(config, n, seed, template_id)


@router.post("/api/sample")
def api_sample_post(request: SamplerRequest) -> dict:
    config = common._sampler_config_from_request(request)
    if request.n is None and request.seed is None and not request.template_id:
        return common._sample(config)
    return common._sample_rows(config, request.n, request.seed, request.template_id or None)
