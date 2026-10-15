from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from numbers import Integral
from string import ascii_uppercase
from typing import Any

from autosteer.filler_tools import function_registry
from autosteer.filler_tools.fill_registry import validate_filler_output
from autosteer.formatting import exist_duplicate_options, format_fill_values, format_value
from autosteer.services.dataset_store import DatasetStore
from autosteer.services.generated_seed_service import (
    _review_key,
    load_review_state,
    save_review_state,
)


SOLVER_SYSTEM_PROMPT = (
    "You are validating a multiple-choice economics question. Solve the question "
    "and return only JSON with this schema: "
    '{"answers": ["A"], "confidence": "high|medium|low", "explanation": "short markdown explanation"}. '
    "Use one capital letter for each question part. Do not return option text as the answer."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_nested_list(values: list) -> list:
    if values and isinstance(values[0], list) and len(values[0]) == 1 and isinstance(values[0][0], list):
        return [item[0] for item in values]
    return values


def _normalize_options_and_answers(options_list: list, correct_answer: list) -> tuple[list[list], list]:
    return _normalize_nested_list(options_list), _normalize_nested_list(correct_answer)


def _has_duplicate_options(options_list: list) -> bool:
    normalized = _normalize_nested_list(options_list)
    return exist_duplicate_options(normalized)


def _answer_text(options: list[str], correct_answer: Any) -> str:
    if isinstance(correct_answer, list):
        if all(isinstance(value, (bool, Integral)) for value in correct_answer):
            bool_mask = [bool(value) for value in correct_answer]
            if True not in bool_mask:
                raise ValueError("correct_answer mask does not mark any option as correct.")
            index = bool_mask.index(True)
            if index >= len(options):
                raise ValueError("correct_answer mask points outside the options list.")
            return options[index]
        if len(correct_answer) == 1:
            answer = str(format_value(correct_answer[0]))
            if answer in options:
                return answer
        raise ValueError("correct_answer list must be a one-hot mask or a single answer value.")
    answer = str(format_value(correct_answer))
    if answer not in options:
        raise ValueError(f"correct_answer {answer!r} is not in the options list.")
    return answer


def _correct_letter(options: list[str], answer: str) -> str:
    try:
        return ascii_uppercase[options.index(answer)]
    except ValueError as exc:
        raise ValueError(f"Correct answer {answer!r} is not in the filled options.") from exc


def _filled_option_text(value: Any, fill_values: dict) -> str:
    return str(format_value(value)).format(**fill_values)


def _filled_seed_from_record(record: dict, filler_result: dict) -> dict:
    fill_values = deepcopy(filler_result)
    if ("options_list" in fill_values) ^ ("correct_answer" in fill_values):
        raise ValueError("Filler must return both options_list and correct_answer, or neither.")

    if "options_list" in fill_values:
        options_list, correct_answers = _normalize_options_and_answers(
            fill_values.pop("options_list"),
            fill_values.pop("correct_answer"),
        )
        sub_index = int(record.get("sub_id", 0))
        if sub_index >= len(options_list):
            sub_index = 0
        raw_options = options_list[sub_index]
        raw_answer = correct_answers[sub_index]
    else:
        raw_options = [option["option_text"] for option in record.get("options", [])]
        correct_options = [
            option["option_text"]
            for option in record.get("options", [])
            if option.get("correct_answer")
        ]
        if not correct_options:
            raise ValueError("Seed has no marked correct answer and the filler did not provide one.")
        raw_answer = correct_options[0]

    fill_values = format_fill_values(fill_values)
    question_text = str(record["question_text"]).format(**fill_values)
    options = [_filled_option_text(option, fill_values) for option in raw_options]
    if not options or all(not option.strip() for option in options):
        raise ValueError("Filled seed has no options to validate.")
    answer = _answer_text(options, raw_answer)
    letter = _correct_letter(options, answer)
    return {
        "question": question_text,
        "options": options,
        "answer": answer,
        "answer_letter": letter,
    }


def _mcq_prompt(filled_seed: dict) -> str:
    lines = ["Question:", filled_seed["question"], "", "Options:"]
    for index, option in enumerate(filled_seed["options"]):
        lines.append(f"{ascii_uppercase[index]}. {option}")
    return "\n".join(lines)


def parse_solver_payload(response_text: str) -> dict:
    from autosteer.json_utils import parse_model_json

    return parse_model_json(response_text)


def parse_solver_answers(response_text: str) -> list[str]:
    payload = parse_solver_payload(response_text)

    if "answers" in payload:
        answers = payload["answers"]
    elif "answer" in payload:
        answers = [payload["answer"]]
    else:
        raise ValueError("Solver JSON must include 'answers' or 'answer'.")

    if not isinstance(answers, list) or not answers:
        raise ValueError("Solver answers must be a non-empty list.")
    return [str(answer).strip().upper()[:1] for answer in answers]


def call_solver_model(prompt: str, model: str, provider: str) -> str:
    from autosteer.llm_client import LanguageModelClient

    return LanguageModelClient(provider_name=provider).make_request(
        user_prompts=prompt,
        system_prompts=SOLVER_SYSTEM_PROMPT,
        request_type="completion",
        return_type="string",
        model=model,
        response_format={"type": "json_object"},
    )


def _record_dict(record) -> dict:
    item = record.model_dump() if hasattr(record, "model_dump") else dict(record)
    item["metadata"] = dict(item["metadata"])
    item["options"] = [dict(option) for option in item.get("options", [])]
    return item


def validate_generated_seed_record(
    element_name: str,
    record: dict,
    model: str,
    provider: str,
    runs_per_seed: int = 3,
    on_run=None,
) -> dict:
    filler = function_registry.get(element_name)
    runs = []
    passed = 0

    for run_index in range(runs_per_seed):
        if on_run:
            on_run(run_index + 1, runs_per_seed)
        run = {"run": run_index + 1}
        response_text = None
        try:
            filler_result = filler(record["metadata"])
            if filler_result is None:
                raise RuntimeError("Filler returned None.")
            validate_filler_output(element_name, filler_result)

            dedup_attempts = 0
            while "options_list" in filler_result and _has_duplicate_options(filler_result["options_list"]):
                filler_result = filler(record["metadata"])
                dedup_attempts += 1
                if dedup_attempts >= 10:
                    raise ValueError("Filler repeatedly returned duplicate options.")

            filled_seed = _filled_seed_from_record(record, filler_result)
            prompt = _mcq_prompt(filled_seed)
            response_text = call_solver_model(prompt=prompt, model=model, provider=provider)
            solver_payload = parse_solver_payload(response_text)
            predicted = parse_solver_answers(response_text)
            expected = [filled_seed["answer_letter"]]
            is_correct = predicted == expected
            passed += int(is_correct)
            run.update(
                {
                    "status": "passed" if is_correct else "failed",
                    "question": filled_seed["question"],
                    "options": filled_seed["options"],
                    "answer": filled_seed["answer"],
                    "answer_letter": filled_seed["answer_letter"],
                    "expected": expected,
                    "predicted": predicted,
                    "solver_explanation": str(solver_payload.get("explanation", "")),
                    "response": response_text,
                }
            )
        except Exception as exc:
            run.update({"status": "error", "error": str(exc)})
            if response_text is not None:
                run["response"] = response_text
        runs.append(run)

    if all(run["status"] == "passed" for run in runs):
        status = "passed"
    elif any(run["status"] == "error" for run in runs):
        status = "error"
    else:
        status = "failed"

    return {
        "status": status,
        "passed_runs": passed,
        "total_runs": runs_per_seed,
        "model": model,
        "validated_at": _now_iso(),
        "runs": runs,
    }


def _save_solve_check(review: dict, question_id: int, sub_id: int, result: dict) -> None:
    key = _review_key(question_id, sub_id)
    review_entry = review.setdefault(key, {"status": "draft"})
    checks = review_entry.setdefault("checks", {})
    previous_result = checks.get("solve_check") or checks.get("ai_validation")
    if previous_result:
        checks.setdefault("solve_check_history", []).append(previous_result)
    checks["solve_check"] = result
    checks.pop("ai_validation", None)
    review_entry["updated_at"] = _now_iso()


def run_single_generated_seed_validation(
    store: DatasetStore,
    element_name: str,
    question_id: int,
    sub_id: int,
    model: str,
    provider: str,
    runs_per_seed: int = 3,
) -> dict:
    if element_name not in function_registry:
        raise ValueError(f"Element does not have a registered filler function: {element_name}")

    selected_record = None
    for record in store.records("generated_seeds", element_name, dataframe_type="example_questions", limit=10_000):
        item = _record_dict(record)
        if item["question_id"] == question_id and item["sub_id"] == sub_id:
            selected_record = item
            break
    if selected_record is None:
        raise FileNotFoundError(f"No generated seed found for {element_name} id {question_id}.{sub_id}")

    result = validate_generated_seed_record(
        element_name=element_name,
        record=selected_record,
        model=model,
        provider=provider,
        runs_per_seed=runs_per_seed,
    )
    review = load_review_state(store, element_name)
    _save_solve_check(review, question_id, sub_id, result)
    save_review_state(store, element_name, review)
    return result


def run_generated_seed_validation(
    store: DatasetStore,
    element_name: str,
    model: str,
    provider: str,
    runs_per_seed: int = 3,
    statuses: tuple[str, ...] = ("draft", "accepted", "needs_edit"),
    on_progress=None,
    on_run=None,
) -> dict:
    """Solve-check the candidates in the given review states. Each result is saved as soon as it is in (on the review
    state as it is then, so decisions made meanwhile are kept); on_progress(summary, total) follows along."""
    if element_name not in function_registry:
        raise ValueError(f"Element does not have a registered filler function: {element_name}")

    review = load_review_state(store, element_name)
    selected_records = []
    for record in store.records("generated_seeds", element_name, dataframe_type="example_questions", limit=10_000):
        item = _record_dict(record)
        key = _review_key(item["question_id"], item["sub_id"])
        review_entry = review.get(key, {"status": "draft"})
        if review_entry.get("status", "draft") in statuses:
            selected_records.append(item)

    if not selected_records:
        return {
            "validated": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "message": "No generated seeds matched the selected review statuses.",
        }

    summary = {"validated": 0, "passed": 0, "failed": 0, "errors": 0}
    if on_progress:
        on_progress(summary, len(selected_records))
    for record in selected_records:
        result = validate_generated_seed_record(
            element_name=element_name,
            record=record,
            model=model,
            provider=provider,
            runs_per_seed=runs_per_seed,
            on_run=on_run,
        )
        review = load_review_state(store, element_name)
        _save_solve_check(review, record["question_id"], record["sub_id"], result)
        save_review_state(store, element_name, review)
        summary["validated"] += 1
        if result["status"] == "passed":
            summary["passed"] += 1
        elif result["status"] == "failed":
            summary["failed"] += 1
        else:
            summary["errors"] += 1
        if on_progress:
            on_progress(summary, len(selected_records))
    return summary
