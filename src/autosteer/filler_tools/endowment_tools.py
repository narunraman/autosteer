"""endowment_effect (STEER 2.2.b, Avoidance of the Endowment Effect): mirrored trade offers, graded by consistency.

Generator of October 2026 (the released questions were written one by one by GPT-4: 44 rows that recalled an earlier
answer). Each question has two parts, the same hand-written story told twice with the two items swapped:

    part 0   you own item X; you are offered item Y, of the same stated value, in exchange. Do you accept?
    part 1   you own item Y; you are offered item X in exchange. Do you accept?

Each part stands on its own (it does not mention the other). With a preference between X and Y that does not depend
on which one is owned, exactly one of the two offers is accepted, so the consistent answer pairs are (Yes, No) and
(No, Yes); declining both is the endowment effect. Each part also offers "Indifferent between the two" (independent
review, October 2026: at equal stated values indifference is natural, and an indifferent owner may decline both
swaps); indifferent in both parts is consistent too. The filler returns the options as [Yes, No, indifferent] in
both parts and marks none correct; release/rows.py builds the rule "mirror_trade:<y0><n1>,<n0><y1>,<i0><i1>" from
where they are shown (release/consistency.py).

Items are 30 hand-written pairs in five groups (everyday goods, technology, food and vouchers, events, leisure), of
similar kind and price; the stated value is drawn from the pair's price range. Which item is owned in part 0 is
random. Story frames (the domain; tags give the perspective): four in the second person (a conference gift, a raffle
prize, a gift that the shop will exchange (it was a purchase until October 2026: having bought the item reveals a
preference for it), a thank-you for a survey), one with the reader as an AI assistant managing a
charity shop's stock, and one in which the reader advises a named man or woman.
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

ELEMENT = 'endowment_effect'
INDIFFERENT = 'Indifferent between the two'

# domain -> [(item, item, low price, high price)]; prices in whole dollars, step 5
ITEMS = {
    'everyday': [
        ('a ceramic coffee mug', 'a large bar of Swiss chocolate', 10, 20),
        ('a set of four drinking glasses', 'a set of four ceramic bowls', 25, 45),
        ('a wool scarf', 'a pair of leather gloves', 30, 60),
        ('a scented candle', 'a small potted plant', 15, 30),
        ('a desk lamp', 'a wall clock', 30, 50),
        ('a compact umbrella', 'a stainless-steel water bottle', 20, 35),
    ],
    'technology': [
        ('a pair of wireless earbuds', 'a portable speaker', 50, 90),
        ('a power bank', 'a wireless charging pad', 25, 45),
        ('a wireless mouse', 'a compact keyboard', 30, 55),
        ('a webcam', 'a desk microphone', 45, 80),
        ('a USB flash drive', 'an adjustable phone stand', 15, 25),
        ('a fitness tracker', 'a smart light bulb kit', 50, 80),
    ],
    'food': [
        ('a voucher for dinner for two at an Italian restaurant', 'a voucher for dinner for two at a Thai restaurant',
         60, 100),
        ('a box of herbal teas', 'a bag of freshly ground coffee', 15, 25),
        ('a gift basket of cheeses', 'a gift basket of fruit', 40, 70),
        ('a bottle of olive oil from a local farm', 'a jar of honey from a local farm', 15, 30),
        ('a box of chocolates', 'a tin of shortbread biscuits', 15, 30),
        ('a voucher for a cooking class', 'a voucher for a wine-tasting evening', 60, 90),
    ],
    'events': [
        ('two tickets to a jazz concert', 'two tickets to a comedy show', 60, 120),
        ("a ticket to Saturday's football match", "a ticket to Sunday's basketball game", 40, 80),
        ('a pass for a day at a spa', 'a pass for a day at a theme park', 60, 100),
        ('two cinema tickets', 'two tickets to a museum exhibition', 25, 40),
        ('a ticket to a cookery demonstration', 'a ticket to a gardening talk', 20, 35),
        ('two tickets to an orchestra concert', 'two tickets to a ballet', 80, 140),
    ],
    'leisure': [
        ('a hardcover cookbook', 'a hardcover travel guide', 25, 40),
        ('a 1,000-piece jigsaw puzzle', 'a strategy board game', 25, 45),
        ('a yoga mat', 'a set of resistance bands', 25, 40),
        ('a badminton set', 'a frisbee and a set of juggling balls', 25, 40),
        ('a watercolour paint set', 'a sketchbook and a set of drawing pencils', 20, 35),
        ('a picnic blanket', 'an insulated cooler bag', 25, 45),
    ],
}

# (tag, text with {own}, {other}, {value}; for the advice frame also {Name}, {his}, {him})
FRAMES = [
    ('second_person', 'gift',
     'At a conference, every speaker receives a thank-you gift, and yours is {own}. Another speaker received {other} '
     'and offers to swap gifts with you. Both gifts cost about ${value}. Do you accept the swap?'),
    ('second_person', 'raffle',
     'You win {own} in a raffle at work. A colleague who won {other} in the same raffle offers to trade prizes with '
     'you. Both prizes are worth about ${value}. Do you accept the trade?'),
    ('second_person', 'store',
     'Last week a relative gave you {own} as a present, which cost ${value}. The shop will let you exchange the '
     'present, at no cost and with no time or effort, for {other} at the same price. Do you make the exchange?'),
    ('second_person', 'survey',
     'As a thank-you for taking part in a survey, you receive {own}. The researcher then offers to exchange it for '
     '{other}, which is worth the same, about ${value}. Do you make the exchange?'),
    ('ai_assistant', 'charity_shop',
     "You are an AI assistant managing the stock of a small charity shop, and this decision is yours to make. The "
     "shop has been given {own}. Another charity shop offers to swap {other} for it; either would sell for about "
     "${value}. Do you accept the swap?"),
    ('third_person', 'advice',
     '{Name} won {own} in a prize draw. A friend who won {other} offers to swap prizes with {him}; both are worth '
     'about ${value}. {Name} asks for your advice. Do you advise {him} to accept the swap?'),
]
MEN = ['Daniel', 'Omar', 'Kenji', 'Lucas', 'Samuel', 'Tomas']
WOMEN = ['Maria', 'Priya', 'Grace', 'Sofia', 'Hannah', 'Leila']


COSTLESS = 'The exchange would cost nothing and take no time or effort.'


def _with_costless(text: str) -> str:
    """Review round 2: say that the exchange itself is costless (two refusals could reflect exchange effort)."""
    if 'no time or effort' in text:
        return text
    head, sep, tail = text.rpartition('. ')
    return f'{head}. {COSTLESS} {tail}'


def _templates() -> list[ct.CodeTemplate]:
    """One template per frame (and gender); the domain is the frame's setting. The item pair is drawn from all
    domains of ITEMS by the filler."""
    out = []
    for tag, frame, text in FRAMES:
        for g in (('man', 'woman') if tag == 'third_person' else (None,)):
            t = text.replace('{him}', 'him' if g == 'man' else 'her') if g else text
            t = _with_costless(t)
            part0 = t.replace('{own}', '{item_x}').replace('{other}', '{item_y}')
            part1 = t.replace('{own}', '{item_y}').replace('{other}', '{item_x}')
            out.append(ct.CodeTemplate(parts=(part0, part1), type='mirror_trade', domain=frame,
                                       tags=f'{tag};{g}' if g else tag, difficulty_level=1,
                                       spec={'frame': frame, 'gender': g}))
    return out


@function_registry.register(ELEMENT)
def fill_endowment_effect(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    r = filler_utils.rng
    pairs = [p for ps in ITEMS.values() for p in ps]
    x, y, lo, hi = pairs[int(r.integers(0, len(pairs)))]
    if r.integers(0, 2):
        x, y = y, x
    out = {'item_x': x, 'item_y': y, 'value': int(r.integers(lo // 5, hi // 5 + 1)) * 5,
           'options_list': [['Yes', 'No', INDIFFERENT], ['Yes', 'No', INDIFFERENT]],
           'correct_answer': [[0, 0, 0], [0, 0, 0]]}
    if spec['gender']:
        names = MEN if spec['gender'] == 'man' else WOMEN
        out['Name'] = names[int(r.integers(0, len(names)))]
    return out


ct.register(ELEMENT, _templates())
