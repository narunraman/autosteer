"""conditional_syllogism (STEER 1.4.b): "If P, then Q", a second premise, and which conclusion must be true.

Generator of October 2026 (the released questions were written one by one by GPT-4, with no slots and no computed
answers). The rules are a hand-written bank (`RULES`): for each, the four clauses P, not-P, Q, not-Q in one tense
and with one subject, so every clause can stand as a premise or as an option. (The phrase bank of
equiv_contrapositive was not reused: its recovered clauses mix tenses and subjects, e.g. "she goes to bed early" /
"she did not go to bed early" / "Sarah does not wake up feeling refreshed", which would let the wording give the
key away.)

One template per rule, second premise and question stem. The second premise is one of

    modus_ponens            P      -> key Q
    modus_tollens           not-Q  -> key not-P
    affirming_consequent    Q      -> key "Cannot be determined."
    denying_antecedent      not-P  -> key "Cannot be determined."

and the options are the two clauses of the other variable (Q and not-Q after P or not-P; P and not-P after Q or
not-Q) and "Cannot be determined." So the tempting fallacy (P after Q, not-Q after not-P) is always an option. The
type is the form of the argument. Person rules name a man or a woman (tags third_person;man / third_person;woman;
the name is a slot); event and number rules have no protagonist.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

ELEMENT = 'conditional_syllogism'
CBD = 'Cannot be determined.'
FORMS = {  # form -> (second premise, variable asked about, key literal or None)
    'modus_ponens': ('P', 'Q', 'Q'),
    'modus_tollens': ('nQ', 'P', 'nP'),
    'affirming_consequent': ('Q', 'P', None),
    'denying_antecedent': ('nP', 'Q', None),
}
STEMS = ['Solely from the information provided, which of the following must be true?',
         'Based only on these statements, which conclusion follows?']
MEN = ['Daniel', 'Omar', 'Kenji', 'Lucas', 'Samuel', 'Ravi', 'Tomas', 'Ethan']
WOMEN = ['Maria', 'Priya', 'Grace', 'Sofia', 'Hannah', 'Leila', 'Mei', 'Clara']

# Person rules: (domain, P, not-P, Q, not-Q) as predicates of the named person; the rule is
# "If {name} P, then he/she Q." The gender alternates along the list.
PERSON_RULES = [
    ('travel', 'has a first-class ticket', 'does not have a first-class ticket',
     'can use the airport lounge', 'cannot use the airport lounge'),
    ('travel', 'misses the 8:15 train', 'does not miss the 8:15 train',
     'takes a taxi to the office', 'does not take a taxi to the office'),
    ('travel', 'flies to Lisbon in May', 'does not fly to Lisbon in May',
     'visits the old castle', 'does not visit the old castle'),
    ('work', 'finishes the report by Friday', 'does not finish the report by Friday',
     'gets a day off next week', 'does not get a day off next week'),
    ('work', 'is promoted this year', 'is not promoted this year',
     'moves to the head office', 'does not move to the head office'),
    ('work', 'works from home on Monday', 'does not work from home on Monday',
     'joins the team meeting by video', 'does not join the team meeting by video'),
    ('school', 'scores above 90 on the final exam', 'does not score above 90 on the final exam',
     'receives an A in the course', 'does not receive an A in the course'),
    ('school', 'joins the debate team', 'does not join the debate team',
     'stays late at school on Thursdays', 'does not stay late at school on Thursdays'),
    ('school', 'submits the essay late', 'does not submit the essay late',
     'loses ten points on the essay', 'does not lose ten points on the essay'),
    ('health', 'has a fever', 'does not have a fever', 'stays home from work', 'does not stay home from work'),
    ('health', 'takes the allergy pill', 'does not take the allergy pill',
     'feels sleepy in the afternoon', 'does not feel sleepy in the afternoon'),
    ('health', 'runs in the morning', 'does not run in the morning',
     'skips the gym in the evening', 'does not skip the gym in the evening'),
    ('finance', 'gets the year-end bonus', 'does not get the year-end bonus',
     'buys a new laptop', 'does not buy a new laptop'),
    ('finance', 'sells the old car', 'does not sell the old car', 'pays off the loan', 'does not pay off the loan'),
    ('finance', 'earns more than $60,000 this year', 'does not earn more than $60,000 this year',
     'pays the higher tax rate', 'does not pay the higher tax rate'),
    ('daily_life', 'goes to the market on Saturday', 'does not go to the market on Saturday',
     'cooks fish for dinner', 'does not cook fish for dinner'),
    ('daily_life', 'wakes up before six', 'does not wake up before six',
     'walks the dog before breakfast', 'does not walk the dog before breakfast'),
    ('daily_life', 'has a meeting in the morning', 'does not have a meeting in the morning',
     'wears a suit', 'does not wear a suit'),
    ('sports', 'wins the semi-final', 'does not win the semi-final',
     'plays in the final on Sunday', 'does not play in the final on Sunday'),
    ('sports', 'trains every day this month', 'does not train every day this month',
     'runs the marathon in under four hours', 'does not run the marathon in under four hours'),
]
# Event and number rules: (domain, intro or '', P, not-P, Q, not-Q, Q as worded in the rule) as full clauses.
CLAUSE_RULES = [
    ('daily_life', '', 'it rains on Saturday', 'it does not rain on Saturday',
     'the picnic is moved indoors', 'the picnic is not moved indoors', 'the picnic is moved indoors'),
    ('daily_life', 'A family is repainting its kitchen.', 'the shop has the blue paint in stock',
     'the shop does not have the blue paint in stock', 'the kitchen is painted blue',
     'the kitchen is not painted blue', 'the kitchen is painted blue'),
    ('travel', '', 'the river rises above the warning mark', 'the river does not rise above the warning mark',
     'the ferry stops running', 'the ferry does not stop running', 'the ferry stops running'),
    ('work', '', 'the fire alarm sounds', 'the fire alarm does not sound',
     'the building is evacuated', 'the building is not evacuated', 'the building is evacuated'),
    ('work', '', 'the delivery arrives before noon', 'the delivery does not arrive before noon',
     'the shop opens on time', 'the shop does not open on time', 'the shop opens on time'),
    ('school', '', 'the museum is open late on Friday', 'the museum is not open late on Friday',
     'the class visits the museum on Friday evening', 'the class does not visit the museum on Friday evening',
     'the class visits it on Friday evening'),
    ('numbers', 'A whole number is written on a card.', 'the number is divisible by 6',
     'the number is not divisible by 6', 'the number is even', 'the number is not even', 'it is even'),
    ('numbers', 'A whole number is written on a card.', 'the number is greater than 20',
     'the number is not greater than 20', 'the number is greater than 10', 'the number is not greater than 10',
     'it is greater than 10'),
    ('numbers', 'A whole number is written on a card.', 'the last digit of the number is 0',
     'the last digit of the number is not 0', 'the number is divisible by 5', 'the number is not divisible by 5',
     'the number is divisible by 5'),
    ('numbers', 'A whole number is written on a card.', 'the number is a multiple of 9',
     'the number is not a multiple of 9', 'the number is a multiple of 3', 'the number is not a multiple of 3',
     'it is a multiple of 3'),
    ('numbers', 'A shape is drawn on the board.', 'the shape is a square', 'the shape is not a square',
     'the shape has four sides', 'the shape does not have four sides', 'it has four sides'),
    # Added October 2026 so each (form, numbers) cell has 26 distinct questions (the export wants 20; 5 rules gave
    # 10). As above, each rule is true but its converse is not, so world knowledge never settles a fallacy.
    ('numbers', 'A whole number is written on a card.', 'the number is divisible by 10',
     'the number is not divisible by 10', 'the number is divisible by 2', 'the number is not divisible by 2',
     'it is divisible by 2'),
    ('numbers', 'A whole number is written on a card.', 'the number is a multiple of 8',
     'the number is not a multiple of 8', 'the number is a multiple of 4', 'the number is not a multiple of 4',
     'it is a multiple of 4'),
    ('numbers', 'A whole number is written on a card.', 'the number is divisible by 12',
     'the number is not divisible by 12', 'the number is divisible by 3', 'the number is not divisible by 3',
     'it is divisible by 3'),
    ('numbers', 'A whole number is written on a card.', 'the last digit of the number is 5',
     'the last digit of the number is not 5', 'the number is odd', 'the number is not odd', 'the number is odd'),
    ('numbers', 'A whole number is written on a card.', 'the number is less than 5',
     'the number is not less than 5', 'the number is less than 50', 'the number is not less than 50',
     'it is less than 50'),
    ('numbers', 'A whole number is written on a card.', 'the number is greater than 100',
     'the number is not greater than 100', 'the number has at least three digits',
     'the number does not have at least three digits', 'it has at least three digits'),
    ('numbers', 'A whole number is written on a card.', 'the number is a prime number greater than 2',
     'the number is not a prime number greater than 2', 'the number is odd', 'the number is not odd', 'it is odd'),
    ('numbers', 'A shape is drawn on the board.', 'the shape is a regular hexagon',
     'the shape is not a regular hexagon', 'the shape has six sides', 'the shape does not have six sides',
     'it has six sides'),
]


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def rules() -> list[dict]:
    """Every rule as {domain, tags, intro, rule, P, nP, Q, nQ} with {name} as the person's slot."""
    out = []
    for k, (domain, p, np_, q, nq) in enumerate(PERSON_RULES):
        woman = k % 2 == 0
        pron = 'she' if woman else 'he'
        out.append({'domain': domain, 'tags': 'third_person;woman' if woman else 'third_person;man', 'intro': '',
                    'rule': f'If {{name}} {p}, then {pron} {q}.', 'P': f'{{name}} {p}', 'nP': f'{{name}} {np_}',
                    'Q': f'{{name}} {q}', 'nQ': f'{{name}} {nq}'})
    for domain, intro, p, np_, q, nq, qr in CLAUSE_RULES:
        out.append({'domain': domain, 'tags': '', 'intro': intro, 'rule': f'If {p}, then {qr}.', 'P': p, 'nP': np_,
                    'Q': q, 'nQ': nq})
    return out


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, r in enumerate(rules()):
        for form, (premise, _, _) in FORMS.items():
            for s, stem in enumerate(STEMS):
                text = ' '.join(x for x in (r['intro'], cap(r['rule']), cap(r[premise]) + '.', stem) if x)
                out.append(ct.CodeTemplate(parts=(text,), type=form, domain=r['domain'], tags=r['tags'],
                                           difficulty_level=1, spec={'rule': k, 'form': form}))
    return out


@function_registry.register(ELEMENT)
def fill_conditional_syllogism(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    r = rules()[spec['rule']]
    _, asked, key = FORMS[spec['form']]
    rng = filler_utils.rng
    names = WOMEN if 'woman' in r['tags'] else MEN
    name = names[int(rng.integers(0, len(names)))] if r['tags'] else ''
    lits = [asked, 'n' + asked]
    options = [cap(r[x].format(name=name)) + '.' for x in lits] + [CBD]
    key_text = options[lits.index(key)] if key else CBD
    out = {'options_list': [options], 'correct_answer': [[int(o == key_text) for o in options]]}
    if r['tags']:
        out['name'] = name
    return out


ct.register(ELEMENT, _templates())
