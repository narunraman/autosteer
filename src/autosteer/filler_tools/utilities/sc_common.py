"""Groups, perspectives and ballot formatting shared by the 4.1 social-choice axiom elements (code templates)."""
from __future__ import annotations

from .story_utils import MEN, WOMEN

LETTERS = 'BCDEFGHJKLMNPQRSTUVWXYZ'
NUMBER_WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}
# a-phrase, the-phrase, what is chosen, option noun, option label, domain
GROUPS = [
    ('a book club', 'the club', 'its next book', 'book', 'Book', 'leisure'),
    ('a city council', 'the council', 'a design for the new park', 'design', 'Design', 'politics'),
    ('a band', 'the band', 'the opening song for its tour', 'song', 'Song', 'entertainment'),
    ('a family', 'the family', 'a holiday destination', 'destination', 'Destination', 'family'),
    ('an office team', 'the team', 'a venue for its end-of-year lunch', 'venue', 'Venue', 'workplace'),
    ('a school board', 'the board', 'a new logo for the school', 'logo', 'Logo', 'education'),
]
PERSPECTIVES = ('anon', 'first_person', 'second_person', 'man', 'woman', 'ai_assistant')


def comparing_intro(persp: str, k: int, group: tuple, ing: str, base: str) -> str:
    """'A book club is comparing four voting rules for choosing its next book.' in a perspective.

    `ing` / `base` are the activity in -ing and base form ('comparing four voting rules for choosing' /
    'compare four voting rules for choosing'); the thing chosen follows.
    """
    a_group, thing = group[0], group[2]
    if persp == 'anon':
        s = f'{a_group} is {ing} {thing}.'
        return s[0].upper() + s[1:]
    if persp == 'first_person':
        return f'I chair {a_group} and am {ing} {thing}.'
    if persp == 'second_person':
        return f'You chair {a_group} and are {ing} {thing}.'
    if persp in ('man', 'woman'):
        name = (MEN if persp == 'man' else WOMEN)[k % 7]
        return f'{name} chairs {a_group} and is {ing} {thing}.'
    if persp == 'ai_assistant':
        return f'You are an AI assistant helping {a_group} {base} {thing}.'
    raise ValueError(persp)


def ballot_line(n: int, ranking) -> str:
    return f"{n} member{'s' if n != 1 else ''} voted {' > '.join(ranking)}"


def ballot_lines(prof) -> str:
    return '\n'.join(ballot_line(n, r) for n, r in prof)


def letters_list(cands) -> str:
    return ', '.join(cands[:-1]) + ' and ' + cands[-1]


def borda_points(m: int) -> str:
    if m == 3:
        return '2 points to its first choice, 1 to its second and 0 to its last'
    return f"{', '.join(str(m - 1 - i) for i in range(m - 1))} and 0 points to its choices from first to last"
