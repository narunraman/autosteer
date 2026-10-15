"""Forgiving parsing of JSON produced by language models."""
import json
import re

_FENCE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n?(.*?)```", re.DOTALL)
_STRING_OR_TRAILING_COMMA = re.compile(r'"(?:\\.|[^"\\])*"|,(?=\s*[\]}])')
_DECODER = json.JSONDecoder()


def _strip_trailing_commas(text):
    """Remove commas directly before a closing bracket, leaving string contents alone."""
    out, last = [], 0
    for m in _STRING_OR_TRAILING_COMMA.finditer(text):
        if m.group() == ",":
            out.append(text[last:m.start()])
            last = m.end()
    out.append(text[last:])
    return "".join(out)


def _first_json(text):
    for start, ch in enumerate(text):
        if ch in "{[":
            try:
                return _DECODER.raw_decode(text[start:])[0]
            except ValueError:
                continue
    return None


def parse_model_json(text):
    """Parse the first complete JSON object (or array) in model output.

    Accepts markdown code fences, leading/trailing prose and trailing commas.
    Raises ValueError with a short excerpt of the output when nothing parses.
    """
    if not isinstance(text, str):
        raise ValueError(f"Model output was not text: {type(text).__name__}")
    candidates = [text.strip()] + [m.strip() for m in _FENCE.findall(text)]
    for candidate in candidates:
        for variant in (candidate, _strip_trailing_commas(candidate)):
            try:
                return json.loads(variant)
            except ValueError:
                pass
            found = _first_json(variant)
            if found is not None:
                return found
    excerpt = " ".join(text.split())
    if len(excerpt) > 200:
        excerpt = excerpt[:200] + "..."
    raise ValueError(f"Could not find valid JSON in model output: {excerpt!r}")
