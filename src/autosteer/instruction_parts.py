"""Instructions for the model, kept as parts that are written once and reused.

A seed's instructions tell the model what to write. Most of the text is shared well beyond one seed, so it is kept
as parts and the instructions are built from them:

    task       per element               what the element tests ("Please give an MCQ example testing the ability to ...")
    type       per element and type      what this kind of question asks
    level      per element and level     what sets this difficulty level apart (optional)
    domain     per domain, all elements  how to set a question in the domain ("financial settings")
    tag        per tag                   a perspective (first_person, ...) is shared by all elements; an element's own
                                         tags (increase, round1, ...) are kept with the element; optional

and the rest is filled in: the variables to keep in curly braces, what to do with options and answers, and the
closing line that introduces the example. The built text is what is stored with the seed (instruction_set) and what
the generation pipeline sends, so seed candidates for another domain or perspective get the same task, type and
level text with that domain's and perspective's parts.

Files: <data>/example_seeds/<element>/instruction_parts.json (task, types, levels) and
<data>/instruction_parts.json (domains, tags; the built-in DOMAINS and TAGS are the defaults).
"""
from __future__ import annotations

import json
from pathlib import Path

from autosteer.constants import DOMAINS, TAGS

KINDS = ("task", "type", "level", "domain", "tag")


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def element_file(data_root, element: str) -> Path:
    return Path(data_root) / "example_seeds" / element / "instruction_parts.json"


def shared_file(data_root) -> Path:
    return Path(data_root) / "instruction_parts.json"


def load(data_root, element: str) -> dict:
    """{"task": str, "types": {type: str}, "levels": {raw: str}, "domains": {...}, "tags": {...}, "saved": bool}"""
    el = _read(element_file(data_root, element))
    shared = _read(shared_file(data_root))
    shared_tags = {**TAGS, **shared.get("tags", {})}
    return {
        "task": el.get("task", ""),
        "types": dict(el.get("types", {})),
        "levels": {str(k): v for k, v in el.get("levels", {}).items()},
        "domains": {**DOMAINS, **shared.get("domains", {})},
        "tags": {**el.get("tags", {}), **shared_tags},
        "shared_tags": sorted(shared_tags),
        "saved": bool(el),
    }


def save(data_root, element: str, *, task: str, type_key: str, type_text: str, level, level_text: str,
         domain_key: str, domain_text: str, tags: dict[str, str]) -> None:
    """Write the parts edited in the seed editor: the element's, and the shared domain and tag texts that differ
    from the built-in ones."""
    path = element_file(data_root, element)
    el = _read(path)
    el["task"] = task.strip()
    types, levels, own_tags = el.setdefault("types", {}), el.setdefault("levels", {}), el.setdefault("tags", {})
    shared_keys = set(TAGS) | set(_read(shared_file(data_root)).get("tags", {}))
    for key, text in tags.items():
        key, text = key.strip(), text.strip()
        if key and key not in shared_keys:
            if text:
                own_tags[key] = text
            else:
                own_tags.pop(key, None)
    if type_key.strip():
        if type_text.strip():
            types[type_key.strip()] = type_text.strip()
        else:
            types.pop(type_key.strip(), None)
    if str(level).strip():
        if level_text.strip():
            levels[str(level).strip()] = level_text.strip()
        else:
            levels.pop(str(level).strip(), None)
    _write(path, el)

    spath = shared_file(data_root)
    shared = _read(spath)
    changed = False
    for kind, key, text, builtin in [("domains", domain_key, domain_text, DOMAINS)] + \
            [("tags", k, v, TAGS) for k, v in tags.items() if k.strip() in shared_keys]:
        key, text = key.strip(), text.strip()
        if not key or key == "base":
            continue
        bucket = shared.setdefault(kind, {})
        if text and text != builtin.get(key):
            changed |= bucket.get(key) != text
            bucket[key] = text
        elif key in bucket:
            del bucket[key]
            changed = True
    if changed:
        _write(spath, shared)


def _sentence(text: str) -> str:
    text = " ".join(text.split())
    return text if not text or text[-1] in ".!?" else text + "."


def compose(parts: dict, metadata: dict, *, variables: list[str], num_options: int, include_answer: bool,
            options_mode: str = "seed") -> list[dict]:
    """The instructions as segments [{kind, key, text, missing}], in order. A missing part has missing=True and a
    label in place of text; text() leaves those out. options_mode "seed" is the seed's own JSON (answers marked in
    the options), "none" asks for no options at all (elements whose options the filler writes)."""
    out = []

    def add(kind, key, text, missing=False):
        out.append({"kind": kind, "key": key, "text": text, "missing": missing})

    task = parts.get("task", "").strip()
    add("task", "", _sentence(task) if task else "What this element tests", missing=not task)

    domain = str(metadata.get("domain", "")).strip()
    if domain == "base":
        add("auto", "domain", " The question should not be phrased in terms of a specific domain.")
    elif domain:
        phrase = parts.get("domains", {}).get(domain, "").strip()
        add("domain", domain, f" The question should be concerning {phrase.rstrip('.')}." if phrase else f"How to set a question in {domain}",
            missing=not phrase)

    qtype = str(metadata.get("type", "")).strip()
    if qtype and parts.get("types", {}).get(qtype, "").strip():
        add("type", qtype, " " + _sentence(parts["types"][qtype]))

    level = str(metadata.get("difficulty_level", "")).strip()
    if level and parts.get("levels", {}).get(level, "").strip():
        add("level", level, " " + _sentence(parts["levels"][level]))

    for tag in [t.strip() for t in str(metadata.get("tags", "") or "").replace(",", ";").split(";") if t.strip()]:
        text = parts.get("tags", {}).get(tag, "").strip()
        if text:
            add("tag", tag, " " + _sentence(text))

    if variables:
        add("auto", "variables", " Do not include numbers in the example but leave them as variables as in the example below. "
            f"Use the following variables in the curly braces: {', '.join(dict.fromkeys(variables))}.")
    if options_mode == "none":
        pass  # the filler writes the options: the example has none and none are asked for
    elif num_options:  # a seed without options asks for none: the expected JSON has no options either
        if not include_answer:
            add("auto", "options", " Do not mark any option with correct_answer as True.")
        if num_options == 1:
            add("auto", "options", " The MCQ should have only one option.")
    add("auto", "closing", " See the following example as a guide but give a different example and write a narrative: ")
    return out


def text(segments: list[dict]) -> str:
    return "".join(s["text"] for s in segments if not s["missing"]).strip()


def missing(segments: list[dict]) -> list[dict]:
    return [s for s in segments if s["missing"]]


def default_task(instruction_texts) -> str:
    """For an element with no saved parts: the most common first sentence of its hand-written instructions
    (what candidate generation has used as the task so far)."""
    firsts = [str(t).split(". ")[0].strip() for t in instruction_texts if str(t).strip()]
    return max(set(firsts), key=firsts.count) if firsts else ""
