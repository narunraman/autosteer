"""Templates written by hand in code (not in the data directory).

Some elements have no usable template set in the data directory (the old pickles are lost, ill-posed or need a
story that is generated together with its numbers). Their templates are written here, in the filler module of the
element, and registered with `register`. The sampler then builds the element's template dataset from the code
instead of reading `<data dir>/templates/<element>/`, and the manifest lists the element like any other.

A template is a list of parts (question texts with {placeholders}) plus metadata (type, domain, difficulty, tags)
and a free `spec` dict for the filler. Fillers find the spec of the template they are filling with `spec_for`
(the sampler passes the joined part texts as metadata['question_text']). Options always come from the filler
(Mode A), so code templates have no option rows.
"""
from __future__ import annotations

import dataclasses

from dataclasses import dataclass, field

import pandas as pd

_REGISTRY: dict[str, list['CodeTemplate']] = {}
_BY_TEXT: dict[str, dict[str, dict]] = {}
_COMPOSITES: dict[str, dict[str, str]] = {}


@dataclass(frozen=True)
class CodeTemplate:
    parts: tuple[str, ...]
    type: str
    domain: str
    tags: str = ''
    difficulty_level: int = 0
    spec: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def text(self) -> str:
        return ' '.join(self.parts)


def register(element: str, templates: list[CodeTemplate]) -> None:
    """Set the code templates of an element (replacing any template set in the data directory)."""
    if element in _REGISTRY:
        raise KeyError(f'code templates for {element!r} already registered')
    from .utilities.story_utils import articles, possessive_pronouns
    # "Maria invests Maria's savings" -> "Maria invests her savings"; "a online" -> "an online"
    templates = [dataclasses.replace(t, parts=tuple(articles(possessive_pronouns(p)) for p in t.parts))
                 for t in templates]
    texts = [t.text for t in templates]
    if len(set(texts)) != len(texts):
        dup = next(t for t in texts if texts.count(t) > 1)
        raise ValueError(f'{element}: duplicate code template {dup[:80]!r}')
    _REGISTRY[element] = list(templates)
    _BY_TEXT[element] = {t.text: t.spec for t in templates}


def elements() -> list[str]:
    return sorted(_REGISTRY)


def has_code_templates(element: str) -> bool:
    return element in _REGISTRY


def templates(element: str) -> list[CodeTemplate]:
    return list(_REGISTRY[element])


def spec_for(element: str, question_text: str) -> dict:
    """The spec of the code template whose joined part texts are `question_text`."""
    try:
        return _BY_TEXT[element][question_text]
    except KeyError:
        raise KeyError(f'{element}: no code template with text {question_text[:80]!r}') from None


def dataset(element: str):
    """A steer_dataset.Dataset holding the element's code templates (test pool only; no few-shot examples)."""
    from autosteer.steer_dataset import Dataset

    rows = []
    for qid, t in enumerate(_REGISTRY[element]):
        for sub_id, text in enumerate(t.parts):
            rows.append({'question_id': qid, 'sub_id': sub_id, 'question_text': text, 'domain': t.domain,
                         'type': t.type, 'difficulty_level': int(t.difficulty_level), 'tags': t.tags})
    d = Dataset(element)
    d.questions = pd.DataFrame(rows).astype({'question_id': int, 'sub_id': int, 'difficulty_level': int})
    d.options = d._init_options_df()
    return d


# ----------------------------------------------------------------------------------------------------------------
# Composite elements: one element whose types are other elements' template sets (decision of 5 October 2026, used to
# merge comp_stat_cobb and comp_stat_leonteif). The sampler builds its template dataset by concatenating the sources'
# corrected template datasets (questions and few-shot examples), with `type` set to the source's type name and
# question ids renumbered in source order. The element's filler dispatches on `type`.
# ----------------------------------------------------------------------------------------------------------------
def register_composite(element: str, parts: dict[str, str]) -> None:
    """parts: type name -> source element (whose templates and filler it uses)."""
    if element in _COMPOSITES or element in _REGISTRY:
        raise KeyError(f'templates for {element!r} already registered')
    _COMPOSITES[element] = dict(parts)


def composites() -> list[str]:
    return sorted(_COMPOSITES)


def is_composite(element: str) -> bool:
    return element in _COMPOSITES


def composite_parts(element: str) -> dict[str, str]:
    return dict(_COMPOSITES[element])


def composite_dataset(element: str, load):
    """The concatenated dataset; load(source) returns a source's corrected template Dataset."""
    from autosteer.steer_dataset import Dataset

    out = Dataset(element)
    frames = {'questions': [], 'options': [], 'example_questions': [], 'example_options': []}
    offsets = {'questions': 0, 'example_questions': 0}
    for qtype, source in _COMPOSITES[element].items():
        d = load(source)
        for q_attr, o_attr in (('questions', 'options'), ('example_questions', 'example_options')):
            q, o = getattr(d, q_attr), getattr(d, o_attr)
            if q is None or len(q) == 0:
                continue
            if 'question_id' not in q.columns:
                q = q.reset_index()
            ids = sorted(int(i) for i in q['question_id'].unique())
            remap = {old: offsets[q_attr] + k for k, old in enumerate(ids)}
            q = q.copy()
            q['question_id'] = q['question_id'].map(remap).astype(int)
            q['type'] = qtype
            frames[q_attr].append(q)
            if o is not None and len(o):
                if 'question_id' not in o.columns:
                    o = o.reset_index()
                o = o[o['question_id'].isin(remap)].copy()
                o['question_id'] = o['question_id'].map(remap).astype(int)
                frames[o_attr].append(o)
            offsets[q_attr] += len(ids)
    for attr, parts in frames.items():
        if parts:
            setattr(out, attr, pd.concat(parts, ignore_index=True))
    return out
