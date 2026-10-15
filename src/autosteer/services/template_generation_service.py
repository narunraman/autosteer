from __future__ import annotations

import json
import random

import pandas as pd

from autosteer import steer_dataset as sd
from autosteer.constants import TAGS
from autosteer.json_utils import parse_model_json
from autosteer.llm_client import LanguageModelClient
from autosteer.services.dataset_store import DatasetStore
from autosteer.services.generated_seed_service import (
    GENERATION_PROMPT,
    build_target_instructions,
    generated_seed_records,
    load_review_state,
)


def _review_key(question_id: int, sub_id: int = 0) -> str:
    return f"{int(question_id)}:{int(sub_id)}"


def _metadata_dict(row: dict | pd.Series) -> dict:
    return {
        "domain": str(row["domain"]),
        "type": str(row["type"]),
        "difficulty_level": int(row["difficulty_level"]),
        "tags": "" if pd.isna(row.get("tags", "")) else str(row.get("tags", "")),
    }


def _accepted_question_ids(store: DatasetStore, element_name: str) -> set[int]:
    review = load_review_state(store, element_name)
    accepted = set()
    for key, entry in review.items():
        if not isinstance(entry, dict) or entry.get("status") != "accepted":
            continue
        question_id, _sub_id = key.split(":", 1)
        accepted.add(int(question_id))
    return accepted


def has_accepted_generated_seeds(store: DatasetStore, element_name: str) -> bool:
    """Whether the element has an accepted seed candidate (reads its review state only)."""
    return store.has_dataset("generated_seeds", element_name) and bool(_accepted_question_ids(store, element_name))


def elements_with_accepted_generated_seeds(store: DatasetStore) -> list[str]:
    elements = []
    for element_name in store.list_elements("generated_seeds"):
        try:
            dataset = accepted_generated_seed_dataset(store, element_name)
        except FileNotFoundError:
            continue
        if not dataset.example_questions.empty:
            elements.append(element_name)
    return sorted(elements)


def accepted_generated_seed_dataset(store: DatasetStore, element_name: str) -> sd.Dataset:
    source = store.load_dataset("generated_seeds", element_name)
    accepted_ids = _accepted_question_ids(store, element_name)
    dataset = sd.Dataset(element_name=element_name)
    if not accepted_ids:
        dataset.instruction_set = source.instruction_set.copy()
        return dataset

    question_mask = source.example_questions["question_id"].astype(int).isin(accepted_ids)
    option_mask = source.example_options["question_id"].astype(int).isin(accepted_ids)
    dataset.example_questions = source.example_questions.loc[question_mask].copy().reset_index(drop=True)
    dataset.example_options = source.example_options.loc[option_mask].copy().reset_index(drop=True)
    dataset.instruction_set = source.instruction_set.copy().reset_index(drop=True)
    sd.reindex(dataset, in_place=True)
    return dataset


def build_template_targets(
    store: DatasetStore,
    element_name: str,
    tags: list[str] | None = None,
) -> list[dict]:
    dataset = accepted_generated_seed_dataset(store, element_name)
    if dataset.example_questions.empty:
        return []

    tag_values = tags or list(TAGS.keys())
    metadata_rows = (
        dataset.example_questions[["domain", "type", "difficulty_level"]]
        .drop_duplicates()
        .sort_values(["domain", "type", "difficulty_level"])
    )
    targets = []
    for _, row in metadata_rows.iterrows():
        for tag in tag_values:
            targets.append(
                {
                    "domain": str(row["domain"]),
                    "type": str(row["type"]),
                    "difficulty_level": int(row["difficulty_level"]),
                    "tags": str(tag),
                }
            )
    return targets


def _normalize_seed_template(display_json: dict) -> dict:
    display_json = dict(display_json)
    for key in list(display_json):
        if not key.startswith("options_"):
            continue
        suffix = key.split("_", 1)[1]
        options = display_json.get(key)
        if not isinstance(options, list):
            display_json[key] = []
        elif len(options) == 1 and str(options[0]).strip().lower() in {"", "no options", "no options found"}:
            display_json[key] = []
        if not display_json[key]:
            display_json[f"answer_{suffix}"] = ""
    return display_json


def _ordered_display_json(display_json: dict) -> dict:
    ordered_keys = ["metadata", "instructions"]
    question_keys = sorted(key for key in display_json if key.startswith("question_"))
    for question_key in question_keys:
        suffix = question_key.split("_", 1)[1]
        ordered_keys.extend([question_key, f"options_{suffix}", f"answer_{suffix}"])
    return {key: display_json[key] for key in ordered_keys if key in display_json}


def seed_templates_for_template_target(
    store: DatasetStore,
    element_name: str,
    target_metadata: dict,
) -> list[dict]:
    dataset = accepted_generated_seed_dataset(store, element_name)
    if dataset.example_questions.empty:
        raise ValueError(f"No accepted generated seeds found for {element_name}.")

    command = sd.build_command(
        {
            "domain": target_metadata["domain"],
            "type": target_metadata["type"],
            "difficulty_level": int(target_metadata["difficulty_level"]),
        }
    )
    selected = dataset.query(command, dataframe_type="example_questions", return_type="dataset")
    seed_templates = selected.to_display_json()
    if not seed_templates:
        raise ValueError(f"No accepted generated seed matches target metadata: {target_metadata}.")

    for seed_template in seed_templates:
        instructions_df = sd.get_instructions(dataset.instruction_set, seed_template["metadata"], get_closest_match=True)
        if instructions_df is None or instructions_df.empty:
            raise ValueError(f"No instructions found for seed metadata: {seed_template['metadata']}")
        seed_template["instructions"] = str(instructions_df.iloc[0]["instructions"])
    return [_normalize_seed_template(seed_template) for seed_template in seed_templates]


def build_template_generation_prompt(
    store: DatasetStore,
    element_name: str,
    target_metadata: dict,
) -> tuple[str, str, list[dict]]:
    target_metadata = _metadata_dict(target_metadata)
    seed_templates = seed_templates_for_template_target(store, element_name, target_metadata)
    target_instructions = build_target_instructions(seed_templates, target_metadata)
    starting_content = json.dumps({"metadata": target_metadata, "instructions": target_instructions})
    seed_template_text = "\n".join(json.dumps(_ordered_display_json(seed)) for seed in seed_templates)
    prompt = GENERATION_PROMPT.replace("{seed_templates}", seed_template_text).replace(
        "{starting_content}",
        starting_content,
    )
    return prompt, target_instructions, seed_templates


def _coerce_display_json(response_text: str, target_metadata: dict, instructions: str) -> list[dict]:
    parsed = parse_model_json(response_text)
    if isinstance(parsed, dict):
        parsed_items = [parsed]
    elif isinstance(parsed, list):
        parsed_items = parsed
    else:
        raise ValueError("Template generation response must be a JSON object or list of JSON objects.")

    display_items = []
    for item in parsed_items:
        if not isinstance(item, dict):
            raise ValueError("Template generation response contained a non-object item.")
        item = dict(item)
        if not any(key.startswith("question_") for key in item):
            raise ValueError("Template generation response did not contain a question_1 field.")
        item["metadata"] = _metadata_dict(target_metadata)
        item["instructions"] = instructions
        display_items.append(item)
    return display_items


def generate_template_batch(
    store: DatasetStore,
    element_name: str,
    model: str,
    provider: str,
    templates_per_target: int = 1,
) -> dict:
    if templates_per_target < 1:
        raise ValueError("templates_per_target must be at least 1.")

    targets = build_template_targets(store, element_name)
    if not targets:
        raise ValueError(f"No accepted generated seeds found for {element_name}.")

    client = LanguageModelClient(provider_name=provider)
    template_dataset = store.get_or_create_dataset("templates", element_name)
    generated = []
    errors = []

    for target in targets:
        try:
            prompt, instructions, _seed_templates = build_template_generation_prompt(store, element_name, target)
        except Exception as exc:
            errors.append(f"{target}: {exc}")
            continue

        for _ in range(templates_per_target):
            try:
                response_text = client.make_request(
                    user_prompts=prompt,
                    request_type="generate_question",
                    return_type="string",
                    model=model,
                )
                display_items = _coerce_display_json(response_text, target, instructions)
                dataset_json = sd.display_to_dataset_json(display_items, ignore_options=True)
                template_dataset.append(dataset_json, ignore_options=True)
                generated.extend(display_items)
            except Exception as exc:
                errors.append(f"{target}: {exc}")

    if generated:
        store.save_dataset("templates", element_name, template_dataset)

    return {
        "generated": len(generated),
        "targets": len(targets),
        "templates_per_target": templates_per_target,
        "errors": errors,
        "preview": random.choice(generated) if generated else None,
    }


def template_metadata_counts(store: DatasetStore, element_name: str) -> list[dict]:
    if not store.has_dataset("templates", element_name):
        return []
    dataset = store.load_dataset("templates", element_name)
    if dataset.questions.empty:
        return []
    counts = (
        dataset.questions.groupby(sd.metadata_keys)
        .size()
        .reset_index(name="count")
        .sort_values(["domain", "type", "difficulty_level", "tags"])
    )
    return counts.to_dict("records")


def random_template_preview(store: DatasetStore, element_name: str) -> dict | None:
    if not store.has_dataset("templates", element_name):
        return None
    records = store.records("templates", element_name, dataframe_type="questions", limit=500)
    if not records:
        return None
    return random.choice(records).model_dump()


def template_generation_summary(store: DatasetStore, element_name: str) -> dict:
    generated_records = generated_seed_records(store, element_name, limit=1000)
    accepted_records = [
        record
        for record in generated_records
        if record.get("review", {}).get("status", "draft") == "accepted"
    ]
    targets = build_template_targets(store, element_name) if accepted_records else []

    validation_counts = {"solved": 0, "missed": 0, "filler_errors": 0, "unchecked": 0}
    for record in accepted_records:
        checks = record.get("review", {}).get("checks", {})
        solve_check = checks.get("solve_check") or checks.get("ai_validation") if isinstance(checks, dict) else None
        status = solve_check.get("status") if solve_check else "not_run"
        if status == "passed":
            validation_counts["solved"] += 1
        elif status == "failed":
            validation_counts["missed"] += 1
        elif status == "error":
            validation_counts["filler_errors"] += 1
        else:
            validation_counts["unchecked"] += 1

    existing_template_count = 0
    existing_metadata_count = 0
    if store.has_dataset("templates", element_name):
        dataset = store.load_dataset("templates", element_name)
        existing_template_count = int(len(dataset.questions))
        existing_metadata_count = int(len(dataset.metadata("questions"))) if not dataset.questions.empty else 0

    return {
        "generated_seed_count": len(generated_records),
        "accepted_seed_count": len(accepted_records),
        "target_count": len(targets),
        "perspective_count": len(TAGS),
        "existing_template_count": existing_template_count,
        "existing_metadata_count": existing_metadata_count,
        "validation_counts": validation_counts,
        "metadata_counts": template_metadata_counts(store, element_name),
    }
