"""transitivity (STEER 2.1.b): stated pairwise preferences chain three or four alternatives; pick the implied one.

Generator of October 2026 (the released questions were written one by one by GPT-4, with no slots and no computed
answers). Templates are written here per domain (job offers in cities, holiday countries, fruit for a salad,
project partners, Saturday activities), perspective and direction. The filler draws three or four alternatives
from the domain's pool, a hidden strict order over them, and a set of stated preferences (each "a over b" with a
before b in the order):

    chain    3 or 4 alternatives, each adjacent pair stated (x1 > x2, x2 > x3[, x3 > x4])
    fork     4 alternatives: x1 > x2, x2 > x3, x2 > x4
    join     4 alternatives: x1 > x3, x2 > x3, x3 > x4

The two options are a pair that the stated preferences order only through transitivity (the pair is never stated
directly, and pairs that the statements leave unordered, such as x3 and x4 of a fork, are never asked). The
statements come in random order, each worded in one of four ways ("prefers a over b", "likes a more than b",
"likes b less than a", "finds b less appealing than a"). The type is the direction asked: most_preferred (which of
the two is preferred) or least_preferred (which is liked less). The key is decided by reachability in the
preference graph.

Perspectives (tags): second_person, third_person;man, third_person;woman, and ai_assistant (the reader is an AI
assistant helping a named client, whose preferences are stated).
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

ELEMENT = 'transitivity'

# domain -> (pool, intros by perspective). Intros use {Name}, {his} (his / her) and {Him} as needed.
DOMAINS = {
    'jobs': (
        ['the job in Lisbon', 'the job in Oslo', 'the job in Toronto', 'the job in Nairobi', 'the job in Seoul',
         'the job in Lima', 'the job in Dublin', 'the job in Melbourne', 'the job in Prague', 'the job in Denver'],
        {'second_person': 'You have received job offers in several cities.',
         'third_person': '{Name} has received job offers in several cities and must choose {his} next job.',
         'ai_assistant': 'You are an AI assistant helping {Name} decide between job offers in several cities.'}),
    'countries': (
        ['Peru', 'Japan', 'Norway', 'Morocco', 'Vietnam', 'Portugal', 'Chile', 'Greece', 'Iceland', 'Kenya'],
        {'second_person': 'You are choosing a country for your next holiday.',
         'third_person': '{Name} is choosing a country for {his} next holiday.',
         'ai_assistant': 'You are an AI assistant planning a holiday abroad for {Name}.'}),
    'fruits': (
        ['mangoes', 'kiwis', 'pears', 'plums', 'cherries', 'apricots', 'figs', 'papayas', 'peaches', 'lychees'],
        {'second_person': 'You are choosing fruit for a fruit salad.',
         'third_person': '{Name} is choosing fruit for {his} fruit salad.',
         'ai_assistant': 'You are an AI assistant ordering groceries for {Name}, who is making a fruit salad.'}),
    'people': (
        # "working with X", not "X": "you like Marie less than Leonard" could mean less than Leonard does
        ['working with ' + x for x in ('Leonard', 'Edward', 'Richard', 'Marie', 'Nadia', 'Felix', 'Irene', 'Victor',
                                       'Yusuf', 'Beatrice')],
        {'second_person': 'Your teacher has asked you to choose a partner for a group project.',
         'third_person': "{Name}'s teacher has asked {Him} to choose a partner for a group project.",
         'ai_assistant': 'You are an AI assistant helping {Name} choose a partner for a group project.'}),
    'activities': (
        ['hiking', 'fishing', 'kayaking', 'baking', 'going to the cinema', 'visiting a museum', 'playing tennis',
         'gardening', 'rock climbing', 'painting'],
        {'second_person': 'You are planning how to spend your Saturday.',
         'third_person': '{Name} is planning how to spend {his} Saturday.',
         'ai_assistant': "You are an AI assistant planning {Name}'s Saturday."}),
}
PERSPECTIVES = ['second_person', 'third_person;man', 'third_person;woman', 'ai_assistant']
MEN = ['Daniel', 'Omar', 'Kenji', 'Lucas', 'Samuel', 'Tomas']
WOMEN = ['Maria', 'Priya', 'Grace', 'Sofia', 'Hannah', 'Leila']
QUESTIONS = {
    ('most_preferred', 'second_person'): 'Of the two options below, which do you prefer?',
    ('least_preferred', 'second_person'): 'Of the two options below, which do you like less?',
    ('most_preferred', 'third'): 'Of the two options below, which does {Name} prefer?',
    ('least_preferred', 'third'): 'Of the two options below, which does {Name} like less?',
}
STRUCTURES = {  # name -> (number of alternatives, stated edges (i before j in the hidden order))
    'chain3': (3, [(0, 1), (1, 2)]),
    'chain4': (4, [(0, 1), (1, 2), (2, 3)]),
    'fork': (4, [(0, 1), (1, 2), (1, 3)]),
    'join': (4, [(0, 2), (1, 2), (2, 3)]),
}


def _client_gender(domain: str) -> str:
    return 'woman' if list(DOMAINS).index(domain) % 2 == 0 else 'man'


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for domain, (_, intros) in DOMAINS.items():
        for persp in PERSPECTIVES:
            intro = intros['second_person' if persp == 'second_person' else
                           'ai_assistant' if persp == 'ai_assistant' else 'third_person']
            gender = persp.split(';')[1] if ';' in persp else _client_gender(domain)
            for direction in ('most_preferred', 'least_preferred'):
                q = QUESTIONS[(direction, 'second_person' if persp == 'second_person' else 'third')]
                text = f'{intro} {{preferences}} {q}'
                if persp != 'second_person':
                    text = text.replace('{his}', 'his' if gender == 'man' else 'her').replace(
                        '{Him}', 'him' if gender == 'man' else 'her')
                out.append(ct.CodeTemplate(parts=(text,), type=direction, domain=domain, tags=persp,
                                           difficulty_level=1,
                                           spec={'domain': domain, 'perspective': persp, 'gender': gender,
                                                 'direction': direction}))
    return out


def closure(n: int, edges) -> set[tuple[int, int]]:
    """All (i, j) with i preferred to j by chaining the stated edges."""
    reach = set(edges)
    changed = True
    while changed:
        changed = False
        for a, b in list(reach):
            for c, d in list(reach):
                if b == c and (a, d) not in reach:
                    reach.add((a, d))
                    changed = True
    return reach


def statement(r, subject: str, third: bool, a: str, b: str) -> str:
    s = 's' if third else ''
    k = int(r.integers(0, 4))
    if k == 0:
        body = f'prefer{s} {a} over {b}'
    elif k == 1:
        body = f'like{s} {a} more than {b}'
    elif k == 2:
        body = f'like{s} {b} less than {a}'
    else:
        body = f'find{s} {b} less appealing than {a}'
    return f'{subject} {body}.'


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


@function_registry.register(ELEMENT)
def fill_transitivity(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    pool = DOMAINS[spec['domain']][0]
    r = filler_utils.rng
    structure = list(STRUCTURES)[int(r.integers(0, len(STRUCTURES)))]
    n, edges = STRUCTURES[structure]
    items = [pool[i] for i in r.choice(len(pool), size=n, replace=False)]  # items[0] is the most preferred
    implied = sorted(closure(n, edges) - set(edges))
    i, j = implied[int(r.integers(0, len(implied)))]
    third = spec['perspective'] != 'second_person'
    names = MEN if spec['gender'] == 'man' else WOMEN
    name = names[int(r.integers(0, len(names)))]
    order = [edges[k] for k in r.permutation(len(edges))]
    sentences = []
    for k, (a, b) in enumerate(order):
        subject = 'You' if not third else (name if k == 0 else ('He' if spec['gender'] == 'man' else 'She'))
        sentences.append(statement(r, subject, third, items[a], items[b]))
    options = [cap(items[i]), cap(items[j])]  # i is preferred to j
    key = options[0] if spec['direction'] == 'most_preferred' else options[1]
    out = {'preferences': ' '.join(sentences),
           'options_list': [options], 'correct_answer': [[int(o == key) for o in options]]}
    if third:
        out['Name'] = name
    return out


ct.register(ELEMENT, _templates())
