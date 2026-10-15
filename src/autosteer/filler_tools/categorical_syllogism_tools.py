"""categorical_syllogism (STEER 1.4.a): two categorical premises and four candidate conclusions.

Generator of October 2026 (the released questions were written one by one by GPT-4, with no slots and no computed
answers). Templates are written here: a context sentence naming a population ("Consider the guests at a wedding."),
two premise slots and a question that states the existential assumption (every group mentioned has at least one
member). The filler draws three terms S, M, P from the context's pool and two premises in the classical forms

    A  All X are Y.      E  No X are Y.      I  Some X are Y.      O  Some X are not Y.

one relating M and P, the other S and M, in any of the four figures and any order. The candidate conclusions are the
four forms over (S, P) or, in a quarter of the questions, over (P, S). Which of them must follow is decided by model
checking over the Venn regions of S, M and P: a conclusion follows iff it is true in every set of occupied regions
that makes both premises true and leaves no term empty (`entailed`).

Options: three conclusions and "Cannot be determined." In half of the questions some conclusion follows; the key is
one of the conclusions that follow and the other two listed conclusions do not follow (when two follow, e.g. "All S
are P" and "Some S are P", only one of them is listed). In the other half no conclusion about the two outer terms
follows, in either direction (S-P or P-S), and the key is "Cannot be determined." The terms are attributes that real-world knowledge does not relate (vaccinated / microchipped animals,
wooden / imported items) or made-up words, so the question is decided by the premises alone.
"""
from __future__ import annotations

from itertools import product

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

ELEMENT = 'categorical_syllogism'
CBD = 'Cannot be determined.'
FORMS = 'AEIO'

# domain -> (contexts, term pool). Terms are plural noun phrases used as subject and as predicate.
POOLS = {
    'people': (
        ['Consider the guests at a large wedding.', 'Consider the residents of a small coastal town.',
         'Consider the members of a community centre.'],
        ['musicians', 'teachers', 'vegetarians', 'cyclists', 'chess players', 'gardeners', 'painters', 'swimmers',
         'nurses', 'poets', 'bakers', 'photographers', 'runners', 'volunteers'],
    ),
    'animals_plants': (
        ['Consider the animals at a wildlife rescue centre.', 'Consider the animals living on a large farm.',
         'Consider the plants in a botanical garden.', 'Consider the plants for sale at a garden nursery.'],
        None,  # per context, below
    ),
    'objects': (
        ['Consider the items in an antique shop.', 'Consider the items stored in a museum warehouse.',
         'Consider the items listed at a charity auction.'],
        ['wooden items', 'painted items', 'fragile items', 'imported items', 'discounted items', 'handmade items',
         'insured items', 'signed items', 'catalogued items', 'restored items'],
    ),
    'fictional': (
        ['Consider the creatures of an imaginary planet described in a logic puzzle.',
         'Consider the made-up kinds of things in a logic textbook exercise.'],
        ['wugs', 'blickets', 'daxes', 'feps', 'toves', 'zorbs', 'glorps', 'snarks', 'quibbles', 'mimbles'],
    ),
}
ANIMALS = ['vaccinated animals', 'microchipped animals', 'rescued animals', 'nocturnal animals',
           'long-haired animals', 'injured animals', 'sponsored animals', 'tagged animals', 'nervous animals']
PLANTS = ['flowering plants', 'climbing plants', 'imported plants', 'rare plants', 'potted plants',
          'evergreen plants', 'medicinal plants', 'labelled plants', 'scented plants']

STEMS = [
    'Assume that every group mentioned has at least one member. Solely from the information provided, which of the '
    'following must be true?',
    'Taking both statements as true, and assuming that each group mentioned is non-empty, which of the following '
    'necessarily follows?',
    'If both statements are true and there is at least one of each kind of thing mentioned, which conclusion must '
    'also be true?',
]


def pool_for(domain: str, context: str) -> list[str]:
    contexts, pool = POOLS[domain]
    if pool is not None:
        return pool
    return ANIMALS if 'animals' in context else PLANTS


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for domain, (contexts, _) in POOLS.items():
        for c, context in enumerate(contexts):
            for s, stem in enumerate(STEMS):
                out.append(ct.CodeTemplate(parts=(f'{context} {{premise_1}} {{premise_2}} {stem}',), type='default',
                                           domain=domain, tags='', difficulty_level=1,
                                           spec={'domain': domain, 'context': c}))
    return out


def sentence(form: str, x: str, y: str) -> str:
    return {'A': f'All {x} are {y}.', 'E': f'No {x} are {y}.', 'I': f'Some {x} are {y}.',
            'O': f'Some {x} are not {y}.'}[form]


# Venn regions of three terms (0, 1, 2): bit t of a region is set iff the region lies inside term t.
REGIONS = range(1, 8)


def holds(form: str, x: int, y: int, occupied) -> bool:
    inside = [r for r in occupied if r >> x & 1]
    if form == 'A':
        return all(r >> y & 1 for r in inside)
    if form == 'E':
        return not any(r >> y & 1 for r in inside)
    if form == 'I':
        return any(r >> y & 1 for r in inside)
    return any(not r >> y & 1 for r in inside)


def _models():
    for bits in product((0, 1), repeat=7):
        occ = [r for r, b in zip(REGIONS, bits) if b]
        if all(any(r >> t & 1 for r in occ) for t in range(3)):  # existential import: no empty term
            yield occ


MODELS = list(_models())


def entailed(premises, x: int, y: int) -> set[str]:
    """Forms (of 'AEIO') over (x, y) true in every model of the premises [(form, a, b), ...]."""
    sat = [m for m in MODELS if all(holds(f, a, b, m) for f, a, b in premises)]
    assert sat, premises
    return {f for f in FORMS if all(holds(f, x, y, m) for m in sat)}


def draw(r, want_valid: bool):
    """Premises [(form, a, b)] over S=0, M=1, P=2 and the conclusion orientation (x, y), with some conclusion over
    (x, y) entailed iff want_valid."""
    while True:
        major = (FORMS[int(r.integers(0, 4))],) + ((1, 2) if r.integers(0, 2) else (2, 1))
        minor = (FORMS[int(r.integers(0, 4))],) + ((0, 1) if r.integers(0, 2) else (1, 0))
        xy = (0, 2) if r.random() < 0.75 else (2, 0)
        prem = [major, minor] if r.integers(0, 2) else [minor, major]
        ent = entailed(prem, *xy)
        if bool(ent) != want_valid:
            continue
        # "Cannot be determined" is keyed only if no conclusion about the two outer terms follows in either
        # direction: "Some S are M; No M are P" entails "Some S are not P", which would make "Cannot be determined"
        # wrong even when that conclusion is not listed and the listed ones are over (P, S) (review, Oct 2026)
        if not want_valid and entailed(prem, *xy[::-1]):
            continue
        return prem, xy, ent


@function_registry.register(ELEMENT)
def fill_categorical_syllogism(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    context = POOLS[spec['domain']][0][spec['context']]
    pool = pool_for(spec['domain'], context)
    r = filler_utils.rng
    terms = [pool[i] for i in r.choice(len(pool), size=3, replace=False)]
    want_valid = bool(r.integers(0, 2))
    prem, (x, y), ent = draw(r, want_valid)
    rest = [f for f in FORMS if f not in ent]
    if ent:
        key_form = sorted(ent)[int(r.integers(0, len(ent)))]
        distract = [rest[i] for i in sorted(r.choice(len(rest), size=2, replace=False))]
        forms = [key_form] + distract
    else:
        forms = [rest[i] for i in sorted(r.choice(4, size=3, replace=False))]
    options = [sentence(f, terms[x], terms[y]) for f in forms] + [CBD]
    key = options[0] if ent else CBD
    p1, p2 = (sentence(f, terms[a], terms[b]) for f, a, b in prem)
    return {'premise_1': p1, 'premise_2': p2,
            'options_list': [options], 'correct_answer': [[int(o == key) for o in options]]}


ct.register(ELEMENT, _templates())
