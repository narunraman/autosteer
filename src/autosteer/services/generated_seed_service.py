from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from autosteer import instruction_parts
from autosteer import steer_dataset as sd
from autosteer.constants import DOMAINS, TAGS
from autosteer.schemas import Metadata
from autosteer.services.dataset_store import DatasetStore


GENERATION_PROMPT = (
    "Below, you will find some metadata that corresponds to a set of instructions "
    "and a template MCQ example that abides by these instructions.\n"
    "{seed_templates}\n\n"
    "Given the following metadata and instructions please write a template MCQ "
    "that abides by the following instructions and metadata. Make sure the template "
    "fits the following set of metadata and instructions. \n"
    "{starting_content}"
)

REVIEW_STATUSES = {"draft", "accepted", "rejected", "needs_edit"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metadata_dict(metadata: dict | Metadata) -> dict:
    if isinstance(metadata, Metadata):
        return metadata.model_dump()
    return {
        "domain": str(metadata.get("domain", "")),
        "type": str(metadata.get("type", "")),
        "difficulty_level": int(metadata.get("difficulty_level", 0)),
        "tags": "" if pd.isna(metadata.get("tags", "")) else str(metadata.get("tags", "")),
    }


def _review_key(question_id: int, sub_id: int = 0) -> str:
    return f"{int(question_id)}:{int(sub_id)}"


def _review_path(store: DatasetStore, element_name: str) -> Path:
    return store.dataset_path("generated_seeds", element_name) / "review.json"


def load_review_state(store: DatasetStore, element_name: str) -> dict:
    path = _review_path(store, element_name)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_review_state(store: DatasetStore, element_name: str, review: dict) -> None:
    path = _review_path(store, element_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(review, f, indent=2, sort_keys=True)


def set_review_status(
    store: DatasetStore,
    element_name: str,
    question_id: int,
    sub_id: int,
    status: str,
    notes: str = "",
) -> dict:
    if status not in REVIEW_STATUSES:
        raise ValueError(f"Unknown review status: {status}")
    review = load_review_state(store, element_name)
    key = _review_key(question_id, sub_id)
    entry = review.setdefault(key, {})
    entry["status"] = status
    entry["notes"] = notes
    entry["updated_at"] = _now_iso()
    save_review_state(store, element_name, review)
    return entry


def elements_with_example_seeds(store: DatasetStore) -> list[str]:
    elements = []
    for element_name in store.list_elements("example_seeds"):
        try:
            dataset = store.load_dataset("example_seeds", element_name)
        except FileNotFoundError:
            continue
        if not dataset.example_questions.empty:
            elements.append(element_name)
    return sorted(elements)


def _unique_in_order(values) -> list:
    seen = set()
    unique = []
    for value in values:
        cleaned = "" if pd.isna(value) else value
        key = str(cleaned)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def example_seed_summary(store: DatasetStore, element_name: str) -> dict:
    dataset = store.load_dataset("example_seeds", element_name)
    metadata_df = sd.get_metadata_df(dataset.example_questions)
    tags = [str(tag) for tag in _unique_in_order(metadata_df["tags"].tolist())]
    selected_tag = tags[0] if tags else ""
    type_difficulty_pairs = (
        metadata_df[["type", "difficulty_level"]]
        .drop_duplicates()
        .sort_values(["type", "difficulty_level"])
        .to_dict("records")
    )
    return {
        "element_name": element_name,
        "example_count": int(len(dataset.example_questions)),
        "instruction_count": int(len(dataset.instruction_set)),
        "domains": [str(value) for value in _unique_in_order(metadata_df["domain"].tolist())],
        "types": [str(value) for value in _unique_in_order(metadata_df["type"].tolist())],
        "difficulty_levels": [int(value) for value in _unique_in_order(metadata_df["difficulty_level"].tolist())],
        "tags": tags,
        "selected_tag": selected_tag,
        "type_difficulty_pairs": type_difficulty_pairs,
    }


def build_target_grid(
    store: DatasetStore,
    element_name: str,
    domains: list[str] | None = None,
    tags: str | None = None,
) -> list[dict]:
    summary = example_seed_summary(store, element_name)
    selected_domains = domains or [domain for domain in DOMAINS if domain != "base"]
    selected_tag = summary["selected_tag"] if tags is None else tags
    targets = []
    for domain in selected_domains:
        if domain == "base":
            continue
        for pair in summary["type_difficulty_pairs"]:
            targets.append(
                {
                    "domain": domain,
                    "type": str(pair["type"]),
                    "difficulty_level": int(pair["difficulty_level"]),
                    "tags": selected_tag,
                }
            )
    return targets


def _extract_variables(display_json: dict) -> str:
    text = display_json.get("question_1", "")
    text += "\n".join(str(option) for option in display_json.get("options_1", []))
    text += str(display_json.get("answer_1", ""))
    return ";".join(dict.fromkeys(re.findall(r"{.*?}", text)))


# The only elements whose fillers read the option text of their templates; every other filler builds the options
# itself, so a model is never asked to write options for them. Found by filling every element's templates with the
# option text blanked (10 seeds each, October 2026): only these five changed (they fail to fill).
TEMPLATE_OPTION_ELEMENTS = frozenset({
    "backward_induction", "certainty_effect", "completeness", "completeness_risk", "reflection_effect",
})


def uses_template_options(element_name: str) -> bool:
    return element_name in TEMPLATE_OPTION_ELEMENTS


def _meaningful_options(options: list | None) -> list[str]:
    if not isinstance(options, list):
        return []
    return [
        str(option)
        for option in options
        if str(option).strip() and not str(option).strip().lower().startswith("no options")
    ]


def _display_json_has_options(display_json: dict) -> bool:
    option_keys = [key for key in display_json if key.startswith("options_")]
    return any(_meaningful_options(display_json.get(key)) for key in option_keys)


def _normalize_generated_options(display_json: dict, has_options: bool) -> dict:
    display_json = dict(display_json)
    if has_options:
        return display_json
    question_keys = sorted(key for key in display_json if key.startswith("question_"))
    for question_key in question_keys:
        suffix = question_key.split("_", 1)[1]
        display_json[f"options_{suffix}"] = []
        display_json[f"answer_{suffix}"] = ""
    return display_json


def _parts_head(parts: dict, target_metadata: dict) -> str:
    """Task, domain, type, level and perspective text for the target, from the element's instruction parts
    (the target's own domain and perspective; missing parts are left out)."""
    segments = instruction_parts.compose(parts, target_metadata, variables=[], num_options=1, include_answer=True)
    keep = [s for s in segments if not s["missing"] and (s["kind"] != "auto" or s["key"] == "domain")]
    return "".join(s["text"] for s in keep).strip()


def build_target_instructions(seed_templates: list[dict], target_metadata: dict, parts: dict | None = None) -> str:
    if not seed_templates:
        raise ValueError("At least one seed template is required.")
    seed = seed_templates[0]
    options = _meaningful_options(seed.get("options_1", []))
    answer = seed.get("answer_1", "")
    num_options = len(options)
    include_answer = bool(str(answer).strip()) and answer in options
    task_description = seed.get("instructions", "").split(". ")[0].strip()
    variables = _extract_variables(seed)

    instruction = task_description + "." if task_description else "Please give an MCQ example testing the ability."
    if parts and parts.get("task"):
        # the element's instruction parts: its task, type and level text, with the target's domain and perspective
        instruction = _parts_head(parts, target_metadata)
    elif target_metadata["domain"] != "base":
        instruction += " The question should be concerning " + DOMAINS.get(
            target_metadata["domain"],
            "<INSERT DOMAIN DESCRIPTION>",
        ) + ". "
    else:
        instruction += " The question should not be phrased in terms of a specific domain. "

    if target_metadata.get("tags") and not (parts and parts.get("task")):
        tag_instructions = []
        missing_tags = []
        for tag in str(target_metadata["tags"]).split(";"):
            if not tag:
                continue
            if tag in TAGS:
                tag_instructions.append(TAGS[tag])
            else:
                missing_tags.append(tag)
        if tag_instructions:
            instruction += ". ".join(tag_instructions) + "."
        if missing_tags:
            instruction += " <INSERT TAG DESCRIPTIONS FOR: " + ", ".join(missing_tags) + ">."

    if variables:
        instruction += (
            " Do not include numbers in the example but leave them as variables as in the example below. "
            "Use the following variables in the curly braces: "
            + ", ".join(sorted(set(variables.split(";"))))
            + "."
        )

    if num_options == 0:
        pass  # no options: the example JSON has no options or answer keys, so none are asked for
    elif not include_answer:
        instruction += " Leave the answer as an empty string."
    if num_options == 1:
        instruction += " The MCQ should have only one option."

    instruction += " See the following example as a guide but give a different example and write a narrative: "
    return re.sub(r"<.*?>", "", instruction)


def _without_empty_options(display_json: dict) -> dict:
    """A seed with no options shows none in the prompt: its options_N / answer_N keys are left out, so the model is not
    asked for empty ones (the reply is normalized back to empty options when it is read)."""
    if any(display_json.get(k) for k in display_json if k.startswith(("options_", "answer_"))):
        return display_json
    return {k: v for k, v in display_json.items() if not k.startswith(("options_", "answer_"))}


def _ordered_display_json(display_json: dict) -> dict:
    ordered_keys = ["metadata", "instructions"]
    question_keys = sorted(key for key in display_json if key.startswith("question_"))
    for question_key in question_keys:
        suffix = question_key.split("_", 1)[1]
        ordered_keys.extend([question_key, f"options_{suffix}", f"answer_{suffix}"])
    return {key: display_json[key] for key in ordered_keys if key in display_json}


def get_seed_templates_for_target(store: DatasetStore, element_name: str, target_metadata: dict) -> list[dict]:
    dataset = store.load_dataset("example_seeds", element_name)
    command = sd.build_command(
        {
            "type": target_metadata["type"],
            "difficulty_level": int(target_metadata["difficulty_level"]),
        }
    )
    selected_dataset = dataset.query(command, dataframe_type="example_questions", return_type="dataset")
    seed_templates = selected_dataset.to_display_json()
    if not seed_templates:
        raise ValueError(
            f"No example seeds for type={target_metadata['type']} "
            f"and difficulty_level={target_metadata['difficulty_level']}."
        )
    for seed_template in seed_templates:
        instructions_df = sd.get_instructions(dataset.instruction_set, seed_template["metadata"], get_closest_match=True)
        if instructions_df is None or instructions_df.empty:
            raise ValueError(f"No instructions found for seed metadata: {seed_template['metadata']}")
        seed_template["instructions"] = str(instructions_df.iloc[0]["instructions"])
    return seed_templates


def build_generation_prompt(store: DatasetStore, element_name: str, target_metadata: dict) -> tuple[str, str, list[dict]]:
    target_metadata = _metadata_dict(target_metadata)
    seed_templates = get_seed_templates_for_target(store, element_name, target_metadata)
    if not uses_template_options(element_name):  # the filler writes the options: show and ask for none
        seed_templates = [_normalize_generated_options(seed, False) for seed in seed_templates]
    parts = instruction_parts.load(store.data_root, element_name)
    target_instructions = build_target_instructions(seed_templates, target_metadata, parts if parts["saved"] else None)
    starting_content = json.dumps({"metadata": target_metadata, "instructions": target_instructions})
    seed_template_text = "\n".join(
        json.dumps(_without_empty_options(_ordered_display_json(_normalize_generated_options(seed, _display_json_has_options(seed)))))
        for seed in seed_templates
    )
    prompt = GENERATION_PROMPT.replace("{seed_templates}", seed_template_text).replace(
        "{starting_content}",
        starting_content,
    )
    return prompt, target_instructions, seed_templates


def validate_display_seed(display_json: dict, target_metadata: dict) -> dict:
    errors = []
    warnings = []
    target_metadata = _metadata_dict(target_metadata)
    if not isinstance(display_json, dict):
        return {"passed": False, "errors": ["Generated seed is not a JSON object."], "warnings": warnings}

    metadata = display_json.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("Generated seed is missing metadata.")
    else:
        for key, expected in target_metadata.items():
            actual = metadata.get(key)
            if key == "difficulty_level":
                try:
                    actual = int(actual)
                except (TypeError, ValueError):
                    pass
            if actual != expected:
                errors.append(f"Metadata mismatch for {key}: expected {expected!r}, got {actual!r}.")

    question_keys = sorted(key for key in display_json if key.startswith("question_"))
    if not question_keys:
        errors.append("Generated seed has no question_1 field.")
    for question_key in question_keys:
        suffix = question_key.split("_", 1)[1]
        question = display_json.get(question_key)
        options = display_json.get(f"options_{suffix}")
        answer = display_json.get(f"answer_{suffix}")
        if not isinstance(question, str) or not question.strip():
            errors.append(f"{question_key} must be a non-empty string.")
        if not isinstance(options, list):
            errors.append(f"options_{suffix} must be a list.")
        elif answer not in options and answer != "":
            errors.append(f"answer_{suffix} must be an option value or an empty string.")
        if question and not re.findall(r"{.*?}", question):
            warnings.append(f"{question_key} has no placeholders.")

    return {"passed": not errors, "errors": errors, "warnings": warnings}


def append_generated_display_seed(
    store: DatasetStore,
    element_name: str,
    display_json: dict,
    target_metadata: dict,
    instructions: str,
    model: str = "",
    prompt: str = "",
    has_options: bool = True,
) -> tuple[int, dict]:
    target_metadata = _metadata_dict(target_metadata)
    display_json = dict(display_json)
    display_json["metadata"] = target_metadata
    display_json["instructions"] = instructions
    # a reply without options (a seed with none asks for none) is a candidate without options
    has_options = has_options and _display_json_has_options(display_json)
    display_json = _normalize_generated_options(display_json, has_options)
    validation = validate_display_seed(display_json, target_metadata)

    dataset = store.get_or_create_dataset("generated_seeds", element_name)
    dataset_json = sd.display_to_dataset_json(display_json)
    dataset.append(dataset_json, dataframe_type="example_questions", ignore_options=not has_options)
    saved_question_id = int(dataset.example_questions["question_id"].max())
    store._upsert_instructions(dataset, target_metadata, target_metadata, instructions)
    store.save_dataset("generated_seeds", element_name, dataset)

    review = load_review_state(store, element_name)
    review[_review_key(saved_question_id, 0)] = {
        "status": "draft",
        "target_metadata": target_metadata,
        "checks": {"structure": validation},
        "model": model,
        "prompt": prompt,
        "has_options": has_options,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "notes": "",
    }
    save_review_state(store, element_name, review)
    return saved_question_id, validation


def generate_generated_seed(
    store: DatasetStore,
    element_name: str,
    target_metadata: dict,
    model: str,
    provider: str,
) -> tuple[int, dict]:
    from autosteer.json_utils import parse_model_json
    from autosteer.llm_client import LanguageModelClient

    prompt, instructions, seed_templates = build_generation_prompt(store, element_name, target_metadata)
    has_options = uses_template_options(element_name) and any(_display_json_has_options(seed) for seed in seed_templates)
    generation_text = LanguageModelClient(provider_name=provider).make_request(
        user_prompts=prompt,
        request_type="generate_seed",
        return_type="string",
        model=model,
    )
    display_json = parse_model_json(generation_text)
    return append_generated_display_seed(
        store=store,
        element_name=element_name,
        display_json=display_json,
        target_metadata=target_metadata,
        instructions=instructions,
        model=model,
        prompt=prompt,
        has_options=has_options,
    )


def generated_seed_records(store: DatasetStore, element_name: str, limit: int = 100) -> list[dict]:
    try:
        records = store.records("generated_seeds", element_name, dataframe_type="example_questions", limit=limit)
    except FileNotFoundError:
        return []
    review = load_review_state(store, element_name)
    result = []
    for record in records:
        item = record.model_dump()
        item["review"] = review.get(_review_key(record.question_id, record.sub_id), {"status": "draft"})
        result.append(item)
    return result


def _option_rows_from_text(question_id: int, sub_id: int, options_text: str, correct_answer: str) -> list[dict]:
    option_texts = [line.strip() for line in options_text.splitlines() if line.strip()]
    if not option_texts:
        return []

    correct_answer = correct_answer.strip()
    correct_index = int(correct_answer) if correct_answer.isdigit() else None
    return [
        {
            "question_id": int(question_id),
            "sub_id": int(sub_id),
            "option_id": option_id,
            "option_text": option_text,
            "correct_answer": option_id == correct_index or option_text == correct_answer,
        }
        for option_id, option_text in enumerate(option_texts)
    ]


def update_generated_seed(
    store: DatasetStore,
    element_name: str,
    question_id: int,
    sub_id: int,
    question_text: str,
    options_text: str = "",
    correct_answer: str = "",
) -> dict:
    dataset = store.load_dataset("generated_seeds", element_name)
    question_mask = (
        (dataset.example_questions["question_id"] == question_id) &
        (dataset.example_questions["sub_id"] == sub_id)
    )
    if not question_mask.any():
        raise FileNotFoundError(f"No generated seed found for {element_name} id {question_id}.{sub_id}")

    review = load_review_state(store, element_name)
    key = _review_key(question_id, sub_id)
    review_entry = review.setdefault(key, {"status": "draft"})

    option_mask = (
        (dataset.example_options["question_id"] == question_id) &
        (dataset.example_options["sub_id"] == sub_id)
    )
    has_options = bool(review_entry.get("has_options", option_mask.any()))
    question_text = question_text.strip()
    if not question_text:
        raise ValueError("Generated seed question text is required.")

    metadata = {
        key: dataset.example_questions.loc[question_mask, key].iloc[0]
        for key in sd.metadata_keys
    }
    dataset.example_questions.loc[question_mask, "question_text"] = question_text

    dataset.example_options = dataset.example_options.loc[~option_mask].copy()
    option_rows = _option_rows_from_text(question_id, sub_id, options_text, correct_answer) if has_options else []
    if option_rows:
        options_df = pd.DataFrame(option_rows).astype(
            {
                "question_id": int,
                "sub_id": int,
                "option_id": int,
                "option_text": str,
                "correct_answer": bool,
            }
        )
        dataset.example_options = pd.concat([dataset.example_options, options_df], ignore_index=True)

    display_json = {
        "question_1": question_text,
        "options_1": [row["option_text"] for row in option_rows] if has_options else [],
        "answer_1": next((row["option_text"] for row in option_rows if row["correct_answer"]), ""),
        "metadata": _metadata_dict(metadata),
    }
    validation = validate_display_seed(display_json, metadata)

    checks = review_entry.setdefault("checks", {})
    previous_result = checks.get("solve_check") or checks.get("ai_validation")
    if previous_result:
        checks.setdefault("solve_check_history", []).append(previous_result)
    checks.pop("solve_check", None)
    checks.pop("ai_validation", None)
    checks["structure"] = validation
    review_entry["has_options"] = has_options
    review_entry["updated_at"] = _now_iso()

    store.save_dataset("generated_seeds", element_name, dataset)
    save_review_state(store, element_name, review)
    return validation


def delete_generated_seed(store: DatasetStore, element_name: str, question_id: int, sub_id: int = 0) -> None:
    dataset = store.load_dataset("generated_seeds", element_name)
    question_mask = (
        (dataset.example_questions["question_id"] == question_id) &
        (dataset.example_questions["sub_id"] == sub_id)
    )
    if not question_mask.any():
        raise FileNotFoundError(f"No generated seed found for {element_name} id {question_id}.{sub_id}")
    option_mask = (
        (dataset.example_options["question_id"] == question_id) &
        (dataset.example_options["sub_id"] == sub_id)
    )
    dataset.example_questions = dataset.example_questions.loc[~question_mask].copy()
    dataset.example_options = dataset.example_options.loc[~option_mask].copy()
    store.save_dataset("generated_seeds", element_name, dataset)

    review = load_review_state(store, element_name)
    review.pop(_review_key(question_id, sub_id), None)
    save_review_state(store, element_name, review)
