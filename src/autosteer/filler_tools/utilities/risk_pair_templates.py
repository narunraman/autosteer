"""Shared template machinery for the two-part risky-choice elements (certainty_effect, reflection_effect).

A builder supplies scenarios (story + option patterns) and a `render(scenario, ctx_index, tok, rng)` function
that returns the templates of one expansion as a list of Item(type, tags, parts); each Part is
(question_text, options, correct) with options listed sure side first. `build_dataset` expands
scenario x context x perspective; items are grouped by type (in first-seen order) and numbered consecutively,
so adding a new type does not renumber the existing ones. Types used:
  two_part       consistency-graded pair (sub_id 0 and 1), no option marked correct
  stated_prior   one question that states the part-0 choice and asks the part-1 decision; the option implied
                 by expected utility is marked correct
A single-question variant with combination options only needs another Item from `render`.

Option patterns use AMT for an amount placeholder and 'a PROB' for an article + probability placeholder
(`opt` turns 'a PROB' into '{p_art} {p_pct}' so the filler can choose 'a' / 'an').
"""
from __future__ import annotations

import os
import random
from typing import Callable, NamedTuple

import pandas as pd

MEN = ['James', 'Carlos', 'Wei', 'Ahmed', 'Daniel', 'Ravi', 'Kofi', 'Mateo', 'Hiroshi', 'Liam']
WOMEN = ['Maria', 'Aisha', 'Priya', 'Sofia', 'Emily', 'Mei', 'Amara', 'Lucia', 'Yuki', 'Hannah']
PERSPECTIVES = ['first_person', 'second_person', 'third_person;man', 'third_person;woman', 'third_person;anon']


class Part(NamedTuple):
    question: str
    options: list  # option templates, sure side first
    correct: list | None = None  # per-option correct flags (None = consistency-graded, none correct)


class Item(NamedTuple):
    type: str  # template type, e.g. 'two_part' / 'stated_prior'
    tags: str  # extra ';'-separated tags appended to 'forced_choice;implicit;<perspective>' ('' for none)
    parts: list  # list[Part], one per sub_id


def persona(kind: str, role: str, name: str | None) -> dict:
    """Grammar tokens: S / s1 = first mention (sentence start / mid-sentence), be, have, pos, obj, Q."""
    if kind == 'first_person':
        return dict(S='I', s1='I', be='am', have='have', pos='my', obj='me', Q='should I', was='was', sub='I')
    if kind == 'second_person':
        return dict(S='You', s1='you', be='are', have='have', pos='your', obj='you', Q='should you', was='were',
                    sub='you')
    if kind in ('third_person;man', 'third_person;woman'):
        pos, obj, sub = ('his', 'him', 'he') if kind == 'third_person;man' else ('her', 'her', 'she')
        return dict(S=name, s1=name, be='is', have='has', pos=pos, obj=obj, Q=f'should {name}', was='was', sub=sub)
    art = 'an' if role[0] in 'aeiou' else 'a'
    return dict(S=f'{art.capitalize()} {role}', s1=f'{art} {role}', be='is', have='has', pos='their', obj='them',
                Q=f'should the {role}', was='was', sub='they')


def now_context(ctx: str, tok: dict, role: str) -> str:
    """A context sentence introduced by 'Now, ' after the subject was already mentioned
    ('Now, the investor has ...', 'Now, at a charity gala, you can ...')."""
    s1 = f'the {role}' if tok['sub'] == 'they' else tok['s1']
    filled = fill_context(ctx, dict(tok, S=s1, s1=s1))
    if not filled.startswith(s1):
        filled = filled[0].lower() + filled[1:]
    return f'Now, {filled}'



def opt(pattern: str, prob: str | None, amount: str) -> str:
    """Option template from a pattern: AMT -> {amount}; 'a PROB' -> '{<prob>_art} {<prob>_pct}'."""
    out = pattern.replace('AMT', '{' + amount + '}')
    if prob is None:
        assert 'PROB' not in out, pattern
        return out
    art = prob.replace('_pct', '_art')
    assert out.count('a PROB') == 1, pattern
    return out.replace('a PROB', '{' + art + '} {' + prob + '}')


def fill_context(ctx: str, tok: dict) -> str:
    # mid-sentence first/second-person subjects: "..., I receives" -> "I receive"
    return (ctx % tok).replace('I receives', 'I receive').replace('you receives', 'you receive')


Render = Callable[[dict, int, dict, random.Random], list]


def build_frames(scenarios: list[dict], render: Render, rng: random.Random, difficulty: int):
    """(questions, options) DataFrames in the Dataset layout."""
    by_type: dict[str, list] = {}
    for sc in scenarios:
        for ci in range(len(sc['ctx'])):
            for kind in PERSPECTIVES:
                name = rng.choice(MEN if kind.endswith(';man') else WOMEN) if 'man' in kind else None
                tok = persona(kind, sc['role'], name)
                for item in render(sc, ci, tok, rng):
                    by_type.setdefault(item.type, []).append((sc, kind, item))
    qrows, orows = [], []
    qid = 0
    for entries in by_type.values():
        for sc, kind, item in entries:
            tags = f'forced_choice;implicit;{kind}' + (f';{item.tags}' if item.tags else '')
            for sub, part in enumerate(item.parts):
                qrows.append(dict(question_id=qid, sub_id=sub, question_text=part.question, explanation=None,
                                  domain=sc['domain'], type=item.type, difficulty_level=difficulty, tags=tags))
                correct = part.correct or [False] * len(part.options)
                for oid, (o, c) in enumerate(zip(part.options, correct)):
                    orows.append(dict(question_id=qid, sub_id=sub, option_id=oid, option_text=o,
                                      correct_answer=bool(c)))
            qid += 1
    return pd.DataFrame(qrows), pd.DataFrame(orows)


def stated_prior_items(prior: Part, new_context: str, new_ask: str, new: Part, tok: dict, whats: str) -> list:
    """Two stated_prior Items (the part-0 choice stated as the sure option / as the lottery) asking the part-1
    decision. Expected utility implies the same side in the new decision: sure -> option 0, lottery -> option 1.
    `new_context` comes after the statement (see now_context)."""
    a, b = prior.options
    items = []
    for side, chosen in (('stated_sure', a), ('stated_lottery', b)):
        stmt = (f'Earlier, {tok["s1"]} {tok["was"]} offered a choice between two {whats}, "{a}" and "{b}", '
                f'and {tok["sub"]} chose "{chosen}".')
        correct = [side == 'stated_sure', side == 'stated_lottery']
        items.append(Item('stated_prior', side, [Part(f'{stmt} {new_context} {new_ask}', list(new.options),
                                                      correct)]))
    return items


def build_dataset(element: str, out_dir: str, scenarios: list[dict], example_scenarios: list[dict],
                  render: Render, seed: int, difficulty: int):
    """Build the test and few-shot pools and save them into out_dir."""
    from autosteer.steer_dataset import Dataset

    rng = random.Random(seed)
    q, o = build_frames(scenarios, render, rng, difficulty)
    eq, eo = build_frames(example_scenarios, render, rng, difficulty)
    qtypes = {'question_id': int, 'sub_id': int, 'difficulty_level': int}
    otypes = {'question_id': int, 'sub_id': int, 'option_id': int, 'correct_answer': bool}
    ds = Dataset(element)
    ds.questions, ds.options = q.astype(qtypes), o.astype(otypes)
    ds.example_questions, ds.example_options = eq.astype(qtypes), eo.astype(otypes)
    os.makedirs(out_dir, exist_ok=True)
    ds.save(out_dir)
    print(f'{element}: {q.question_id.nunique()} templates, {eq.question_id.nunique()} example templates')
    print(q.drop_duplicates('question_id').domain.value_counts().to_dict())
    return ds
