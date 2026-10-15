"""first_order_false_belief and second_order_false_belief (STEER 1.5.a, 1.5.b): unexpected-transfer stories.

Generator of October 2026 (the released questions were written one by one by GPT-4, with no slots and no computed
answers). Ten hand-written scenarios (`SCENARIOS`: an object, the room, its first and second location, what the
owner leaves to do) are told by story templates with slots for the characters' names. Every story is also a list of
events (`events`), and the key is computed by tracking beliefs over them (`track`): who learns where the object is
at each event, and who knows that someone else learned it. The two options are the two locations.

First order (owner A, mover B). Types:
    false_belief  A is away while B moves the object and is not told: A looks in the first location.
    true_belief   A watches the move through the window, stays in the room and sees it, or is sent a message by B
                  about it: A looks in the new location.
  Both types are equally common. Perspectives: third_person (B named) and second_person (the reader is the mover).
  The secret-keeping questions of the released element ("you were told X in confidence; someone asks; what should
  you do?") are not generated: what one should do is a norm, not something the story determines, and recasting it
  as a belief question would make it another unexpected-transfer item.

Second order (redesign of October 2026). Owner B puts the object in a first place and leaves; while B is
away, a mover C (or the reader) moves it to a second place and later to a third, and the observer A, who stays in the
room, watches both moves, so A knows where the object really is. After each move B may learn of it in one of three
ways: not at all ("Nobody tells B"), openly (C phones B in front of A), or secretly (C sends B a message that A does
not know about); after the second move C may also tell A, mistakenly, that B has been told (C sent the message to the
wrong person, and A believes C). The question is where A thinks B will look, which is where A thinks B thinks the
object is: tracked over the events by `track` (belief2[(A, B)]). Types:
    observer_wrong  A's belief about B is false: B secretly learned of a move (key the first or the second place,
                    while B looks in a later one), or A was wrongly told that B learned of the second move (key the
                    third place, while B looks in an earlier one). The key differs from where B will really look.
    observer_right  A's belief about B is true, but B's belief is false: nobody told B anything (key the first place),
                    or B was told of the first move in front of A and not of the second (key the second place). The
                    key differs from where the object is and from what A believes about it.
  In every question the key differs from where the object really is or from where B will really look, and the three
  places are each the key in about a third of the questions, so neither "the place where it was first put" nor "the
  place where it is" answers the questions. Options: the three places. Perspectives: third_person and second_person
  (the reader is the mover C).
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

FIRST, SECOND = 'first_order_false_belief', 'second_order_false_belief'
MEN = ['Daniel', 'Omar', 'Kenji', 'Lucas', 'Samuel', 'Tomas', 'Ravi', 'Felix']
WOMEN = ['Maria', 'Priya', 'Grace', 'Sofia', 'Hannah', 'Leila', 'Clara', 'Nadia']

# domain, room, object, first location, second location (preposition, place), "goes ..." (first order),
# "to ..." (second order). The object is singular ("where will she look for it").
SCENARIOS = [
    ('home', 'the living room', 'marble', ('in', 'the basket'), ('in', 'the toy box'), 'goes outside to play',
     'to play outside'),
    ('office', 'the office', 'stapler', ('in', 'the top drawer'), ('on', 'the filing cabinet'), 'goes to a meeting',
     'to go to a meeting'),
    ('home', 'the hallway', 'car key', ('on', 'the hook by the door'), ('in', 'the fruit bowl'),
     'goes upstairs to shower', 'to take a shower'),
    ('school', 'the classroom', 'notebook', ('in', 'the locker'), ('on', "the teacher's desk"), 'goes to lunch',
     'to have lunch'),
    ('home', 'the garage', 'football', ('in', 'the sports bag'), ('on', 'the top shelf'),
     "goes to a friend's house", 'to visit a friend'),
    ('library', 'the reading room', 'library book', ('on', 'the reading table'), ('on', 'the returns cart'),
     'goes to get a coffee', 'to get a coffee'),
    ('workplace', 'the staff room', 'clipboard', ('on', 'the shelf'), ('in', 'the supply cupboard'),
     'goes to check on a patient', 'to check on a patient'),
    ('travel', 'the beach hut', 'sun hat', ('in', 'the beach bag'), ('in', 'the cooler'), 'goes for a swim',
     'to go for a swim'),
    ('garden', 'the garden', 'trowel', ('in', 'the toolbox'), ('in', 'the flower pot'), 'goes inside for lunch',
     'to have lunch inside'),
    ('restaurant', 'the kitchen', 'recipe card', ('on', 'the counter'), ('in', 'the drawer by the stove'),
     'steps out to take a delivery', 'to take a delivery'),
]
# second order: the third place, per scenario
THIRD = [('on', 'the windowsill'), ('on', 'the bookshelf'), ('on', 'the kitchen table'), ('on', 'the windowsill'),
         ('in', 'the storage box'), ('on', 'the new-arrivals shelf'), ('on', "the nurses' desk"),
         ('on', 'the deck chair'), ('in', 'the wheelbarrow'), ('in', 'the recipe box')]


def move_to(prep: str, place: str) -> str:
    """'into the toy box' / 'onto the filing cabinet': the move says where the object ends up, as the option does
    ("On the filing cabinet"; "to the filing cabinet" left open whether it was put in or on it)."""
    return f"{'into' if prep == 'in' else 'onto'} {place}"


PRON = {'man': ('he', 'him', 'his'), 'woman': ('she', 'her', 'her')}


def track(events, people):
    """events: (location, learners, {knower: learners they know learned it}). Returns (belief: person -> location,
    belief2: (x, y) -> where x thinks y thinks the object is). Before an event, x's model of y is x's own belief
    (x assumes y saw what x saw), which the initial placement makes explicit."""
    belief, belief2 = {}, {}
    for loc, learners, known in events:
        for x in learners:
            belief[x] = loc
        for k, ys in known.items():
            for y in ys:
                belief2[(k, y)] = loc
    return belief, belief2


def first_order_story(variant: str, sc, you: bool):
    """(sentences with {A}/{B} slots and pronouns filled for A, events, asked person). A owns the object."""
    _, room, obj, (p1, l1), (p2, l2), leave_fo, _ = sc
    B = 'you' if you else '{B}'
    moves, sends = ('move', 'send') if you else ('moves', 'sends')
    start = f'{{A}} puts {{his_A}} {obj} {p1} {l1}'
    loc2 = move_to(p2, l2)
    if variant == 'absent':
        text = (f'{start} and {leave_fo}. While {{he_A}} is away, {B} {moves} the {obj} {loc2}. '
                f'Then {{A}} comes back.')
        ev = [(1, {'A', 'B'}, {}), (2, {'B'}, {})]
    elif variant == 'watched':
        text = (f'{start} and steps out of {room}. Through the window, {{he_A}} watches as {B} {moves} the {obj} '
                f'{loc2}. Then {{A}} comes back in.')
        ev = [(1, {'A', 'B'}, {}), (2, {'A', 'B'}, {})]
    elif variant == 'present':
        text = (f'{start}. While {{A}} is still in {room}, {B} {moves} the {obj} {loc2}, and {{A}} sees it '
                f'happen.')
        ev = [(1, {'A', 'B'}, {}), (2, {'A', 'B'}, {})]
    else:  # told
        text = (f'{start} and {leave_fo}. While {{he_A}} is away, {B} {moves} the {obj} {loc2} and {sends} '
                f'{{A}} a message: "I moved your {obj} {loc2}." {{A}} reads the message and then comes back.')
        ev = [(1, {'A', 'B'}, {}), (2, {'B'}, {}), (2, {'A'}, {})]
    return text, ev


def second_order_story(variant: str, k: int, you: bool):
    """B owns the object and is away for both moves; A stays and watches both; C (or you) moves it twice.
    variant = (what B learns of move 1, of move 2): 'none' / 'open' / 'secret' (move 1) and 'none' / 'secret' /
    'false_told' (move 2). Events for `track`: (place, learners, {knower: people they know learned it})."""
    sc = SCENARIOS[k]
    _, room, obj, (p1, l1), (p2, l2), _, leave_so = sc
    p3, l3 = THIRD[k]
    first, second = variant
    C = 'you' if you else '{C}'

    def v(third, second_):
        return second_ if you else third
    text = (f'{{B}} and {{A}} are in {room}. {{B}} puts {{his_B}} {obj} {p1} {l1} and leaves {leave_so}; {{A}} stays '
            f'in {room}. While {{he_B}} is away, {C} {v("comes", "come")} in and {v("moves", "move")} the {obj} '
            f'{move_to(p2, l2)} while {{A}} watches.')
    ev = [(1, {'A', 'B'}, {'A': {'B'}, 'B': {'A'}}), (2, {'A', 'C'}, {})]
    if first == 'none':
        text += ' Nobody tells {B} about this move.'
    elif first == 'open':
        text += (f' Then, in front of {{A}}, {C} {v("phones", "phone")} {{B}} and {v("tells", "tell")} {{him_B}} '
                 f'that the {obj} is now {p2} {l2}.')
        ev.append((2, {'B'}, {'A': {'B'}, 'C': {'B'}}))
    else:
        text += (f' {C[0].upper() + C[1:]} also {v("sends", "send")} {{B}} a message about this move, which {{B}} '
                 f'reads at once; {{A}} does not know about the message.')
        ev.append((2, {'B'}, {'C': {'B'}}))
    text += (f' Later, {C} {v("moves", "move")} the {obj} again, {move_to(p3, l3)}, and {{A}} sees this too.')
    ev.append((3, {'A', 'C'}, {}))
    if second == 'none':
        text += ' Nobody tells {B} about the second move.'
    elif second == 'secret':
        text += (f' {C[0].upper() + C[1:]} then {v("sends", "send")} {{B}} a message about the second move, which '
                 f'{{B}} reads at once; {{A}} does not know about this message.')
        ev.append((3, {'B'}, {'C': {'B'}}))
    else:
        says = v('{C} then says', 'You then say')
        mistaken = v('{C} is mistaken: the message went', 'you are mistaken: your message went')
        text += (f' {says} to {{A}}: "I have just told {{B}} where the {obj} is now." {{A}} believes '
                 f'{v("{C}", "you")}. In fact {mistaken} to the wrong person, and {{B}} never learns about the second '
                 f'move.')
        ev.append((3, set(), {'A': {'B'}}))
    return text, ev


FO_VARIANTS = {'absent': 'false_belief', 'watched': 'true_belief', 'present': 'true_belief', 'told': 'true_belief'}
SO_VARIANTS = {('secret', 'none'): 'observer_wrong', ('none', 'secret'): 'observer_wrong',
               ('open', 'secret'): 'observer_wrong', ('none', 'false_told'): 'observer_wrong',
               ('open', 'false_told'): 'observer_wrong', ('open', 'none'): 'observer_right',
               ('none', 'none'): 'observer_right'}
FO_QUESTIONS = ['Where will {A} look for {his_A} {obj} first?',
                'When {A} wants the {obj}, where will {he_A} look for it first?',
                'Where does {A} think the {obj} is?']
SO_QUESTIONS = ['Where does {A} think {B} will look for the {obj} when {he_B} comes back?',
                'When {B} returns, where does {A} expect {him_B} to look for the {obj} first?']


def _fill_pronouns(text: str, who: str, gender: str) -> str:
    he, him, his = PRON[gender]
    return text.replace(f'{{he_{who}}}', he).replace(f'{{him_{who}}}', him).replace(f'{{his_{who}}}', his)


def _templates(element: str) -> list[ct.CodeTemplate]:
    out = []
    for k, sc in enumerate(SCENARIOS):
        domain, obj = sc[0], sc[2]
        g_main = 'woman' if k % 2 == 0 else 'man'
        g_other = 'man' if k % 2 == 0 else 'woman'
        if element == FIRST:
            # false_belief gets one template per question wording, each true-belief variant one, so both types
            # are equally common
            plan = [('absent', q) for q in range(3)] + [('watched', 0), ('present', 1), ('told', 2)]
            for persp in ('third_person', 'second_person'):
                for variant, qi in plan:
                    body, _ = first_order_story(variant, sc, persp == 'second_person')
                    q = FO_QUESTIONS[qi].replace('{obj}', obj)
                    text = _fill_pronouns(f'{body} {q}', 'A', g_main)
                    tags = f'{persp};{g_main}' if persp == 'third_person' else persp
                    out.append(ct.CodeTemplate(parts=(text,), type=FO_VARIANTS[variant], domain=domain, tags=tags,
                                               difficulty_level=0,
                                               spec={'scenario': k, 'variant': variant, 'you': persp != 'third_person',
                                                     'genders': {'A': g_main, 'B': g_other}}))
        else:
            for variant, kind in SO_VARIANTS.items():
                for persp in ('third_person', 'second_person'):
                    for qi, qtext in enumerate(SO_QUESTIONS):
                        body, _ = second_order_story(variant, k, persp == 'second_person')
                        q = qtext.replace('{obj}', obj)
                        text = _fill_pronouns(_fill_pronouns(f'{body} {q}', 'B', g_main), 'A', g_other)
                        tags = f'{persp};{g_other}' if persp == 'third_person' else persp
                        out.append(ct.CodeTemplate(parts=(text,), type=kind, domain=domain, tags=tags,
                                                   difficulty_level=1,
                                                   spec={'scenario': k, 'variant': variant,
                                                         'you': persp != 'third_person',
                                                         'genders': {'B': g_main, 'A': g_other, 'C': g_main}}))
    return out


def _names(r, genders: dict, roles) -> dict:
    out, used = {}, set()
    for role in roles:
        pool = [n for n in (MEN if genders[role] == 'man' else WOMEN) if n not in used]
        out[role] = pool[int(r.integers(0, len(pool)))]
        used.add(out[role])
    return out


def _options(sc, key_loc: int, third=None):
    places = [sc[3], sc[4]] + ([third] if third else [])
    options = [f'{p.capitalize()} {place}' for p, place in places]
    return options, options[key_loc - 1]


@function_registry.register(FIRST)
def fill_first_order(metadata: dict) -> dict:
    spec = ct.spec_for(FIRST, metadata['question_text'])
    sc = SCENARIOS[spec['scenario']]
    _, events = first_order_story(spec['variant'], sc, spec['you'])
    belief, _ = track(events, 'AB')
    options, key = _options(sc, belief['A'])
    names = _names(filler_utils.rng, spec['genders'], 'A' if spec['you'] else 'AB')
    return {**names, 'options_list': [options], 'correct_answer': [[int(o == key) for o in options]]}


@function_registry.register(SECOND)
def fill_second_order(metadata: dict) -> dict:
    spec = ct.spec_for(SECOND, metadata['question_text'])
    k = spec['scenario']
    _, events = second_order_story(tuple(spec['variant']), k, spec['you'])
    _, belief2 = track(events, 'ABC')
    options, key = _options(SCENARIOS[k], belief2[('A', 'B')], THIRD[k])
    roles = 'BA' if spec['you'] else 'BAC'
    names = _names(filler_utils.rng, spec['genders'], roles)
    return {**names, 'options_list': [options], 'correct_answer': [[int(o == key) for o in options]]}


ct.register(FIRST, _templates(FIRST))
ct.register(SECOND, _templates(SECOND))
