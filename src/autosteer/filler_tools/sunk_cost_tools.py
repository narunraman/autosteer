"""sunk_cost (STEER 2.2.a, Avoidance of the Sunk Cost Fallacy): 20 hand-written scenarios with named slots.

Decision of 3 October 2026. The old templates (8,382 GPT-written stories with anonymous `{}` slots) did not record
which slot held the sunk amount, the remaining cost or the projected value, so the fill could not be recomputed.
These scenarios are written by hand here. Each states:
  - the sunk amount {sunk} already spent (money or hours), which cannot be recovered;
  - continuing: a further cost {cost_continue} and a future value {value_continue};
  - switching to an alternative: a cost {cost_switch} and a future value {value_switch};
  - stopping: no further cost and no further value.
The key is the option with the highest future net value (value - cost; 0 for stopping), whatever the sunk amount.
The key is continuing, switching or stopping, each a third of the time (review, October 2026: before,
continuing was never the key, so "never continue" scored full marks). When the key is switching or stopping, the
draw is a sunk-cost trap: counting the sunk amount as lost when one does not continue (i.e. adding it to the net
value of continuing) makes continuing look best. When the key is continuing, the sunk amount is drawn independently
of the future values, so it does not predict the answer. The margins are at least three units of the scenario's
scale. Time savings are stated as totals over the coming year, with nothing saved after that (a recurring task such
as weekly reports could otherwise be read as saving the amount every week).

Perspectives (tags): first_person, second_person, third_person;man, third_person;woman, and ai_assistant, where
the reader is an AI assistant in charge of the decision ("You are an AI assistant in charge of ...").
"""
from __future__ import annotations

from . import code_templates as ct
from .fill_registry import FillRegister
from .utilities import filler_utils

function_registry = FillRegister()

# unit: 'money' (amounts in dollars, multiples of `step`) or 'time' (hours, multiples of `step`).
# setup: the story up to the decision, with {Subj} (sentence subject), {poss} (possessive) and {sunk}.
# cont / switch / stop: (imperative, gerund, rest) -> option "Imperative rest", sentence "Gerund rest would ...".
# gain: verb phrase for the value ('bring in' / 'be worth' for money, 'save' for time).
SCENARIOS = [
    dict(domain='technology', unit='money', step=1000, gain='bring in',
         setup="{Subj} already spent {sunk} developing a line of wireless earbuds for {poss} electronics company, and the prototype is half finished. A rival has just launched a cheaper model, so the earbuds will sell for less than planned.",
         ai_role='the product budget of a small electronics company',
         cont=('Finish', 'Finishing', 'the earbud prototype'), switch=('Switch', 'Switching', 'to developing a fitness tracker'),
         stop=('Cancel', 'Cancelling', 'the product line')),
    dict(domain='construction', unit='money', step=10000, gain='bring in',
         setup="{Subj} already spent {sunk} on a half-built hotel on a plot owned by {poss} firm. A new zoning rule limits how many rooms can be rented, which lowers what the hotel can earn.",
         ai_role="a property developer's hotel project",
         cont=('Complete', 'Completing', 'the hotel'), switch=('Convert', 'Converting', 'the building into apartments'),
         stop=('Halt', 'Halting', 'all construction')),
    dict(domain='medical', unit='money', step=1000, gain='be worth',
         setup="{Subj} already spent {sunk} on parts of a new imaging system for {poss} clinic, but the supplier has gone out of business before delivering the rest.",
         ai_role="a clinic's equipment budget",
         cont=('Finish', 'Finishing', 'the imaging system with parts from other vendors'),
         switch=('Buy', 'Buying', 'a complete system from another supplier'),
         stop=('Drop', 'Dropping', 'the imaging upgrade')),
    dict(domain='marketing', unit='money', step=1000, gain='bring in',
         setup="{Subj} already spent {sunk} producing and airing the first half of a television campaign for {poss} sportswear brand. Since then, most of the target audience has moved to streaming and social media.",
         ai_role="a sportswear brand's advertising budget",
         cont=('Run', 'Running', 'the rest of the television campaign'), switch=('Move', 'Moving', 'the campaign online'),
         stop=('End', 'Ending', 'the campaign')),
    dict(domain='technology', unit='money', step=1000, gain='be worth',
         setup="{Subj} already spent {sunk} on a custom inventory system for {poss} warehouse business. Halfway through, an off-the-shelf tool that does most of the same things has come on the market.",
         ai_role="a warehouse company's software projects",
         cont=('Finish', 'Finishing', 'the custom inventory system'), switch=('Adopt', 'Adopting', 'the off-the-shelf tool'),
         stop=('Keep', 'Keeping', 'the old spreadsheets')),
    dict(domain='entertainment', unit='money', step=10000, gain='bring in',
         setup="{Subj} already spent {sunk} shooting a feature film for {poss} production company. The lead actor has left the project, and the distributor has lowered its offer for the finished film.",
         ai_role="a production company's film slate",
         cont=('Finish', 'Finishing', 'the feature film with a new lead'), switch=('Turn', 'Turning', 'the footage into a short series'),
         stop=('Shelve', 'Shelving', 'the film')),
    dict(domain='food_service', unit='money', step=1000, gain='bring in',
         setup="{Subj} already spent {sunk} renovating the dining room of {poss} restaurant. A new office park nearby has made takeaway orders far more common than sit-down meals.",
         ai_role="a restaurant's renovation budget",
         cont=('Complete', 'Completing', 'the dining-room renovation'), switch=('Convert', 'Converting', 'the space into a takeaway kitchen'),
         stop=('Stop', 'Stopping', 'the renovation')),
    dict(domain='agriculture', unit='money', step=1000, gain='bring in',
         setup="{Subj} already spent {sunk} planting a young apple orchard on {poss} farm. A blight has since spread through the region and will cut the harvest for years.",
         ai_role="a family farm's planting decisions",
         cont=('Keep', 'Keeping', 'the apple orchard'), switch=('Replace', 'Replacing', 'the apple trees with a vineyard'),
         stop=('Abandon', 'Abandoning', 'the orchard')),
    dict(domain='personal_finance', unit='money', step=100, gain='be worth',
         setup="{Subj} already spent {sunk} repairing an old family car. Now its transmission has failed as well.",
         ai_role="a household's car and repair budget",
         cont=('Repair', 'Repairing', 'the old car again'), switch=('Buy', 'Buying', 'a used car instead'),
         stop=('Give', 'Giving', 'up the car')),
    dict(domain='events', unit='money', step=100, gain='be worth',
         setup="{Subj} already paid a non-refundable {sunk} for a lakeside venue for {poss} team's annual retreat. A storm warning has since been issued for that weekend, and a city venue is still free.",
         ai_role="the planning of a company's team retreat",
         cont=('Hold', 'Holding', 'the retreat at the lakeside venue'), switch=('Book', 'Booking', 'the city venue'),
         stop=('Cancel', 'Cancelling', 'the retreat')),
    dict(domain='research', unit='money', step=1000, gain='bring in',
         setup="{Subj} already spent {sunk} on an experiment in {poss} laboratory. A published study has since shown that the method gives weaker results than expected.",
         ai_role="a research laboratory's project budget",
         cont=('Continue', 'Continuing', 'the current experiment'), switch=('Start', 'Starting', 'an experiment with the newer method'),
         stop=('End', 'Ending', 'this line of research')),
    dict(domain='manufacturing', unit='money', step=1000, gain='be worth',
         setup="{Subj} already spent {sunk} building a custom packaging machine for {poss} factory. A standard machine that does the same job has just become available from a large supplier.",
         ai_role="a factory's equipment projects",
         cont=('Finish', 'Finishing', 'the custom packaging machine'), switch=('Buy', 'Buying', 'the standard machine'),
         stop=('Continue', 'Continuing', 'to pack by hand')),
    dict(domain='technology', unit='money', step=1000, gain='bring in',
         setup="{Subj} already spent {sunk} developing a puzzle game for phones at {poss} studio. The main app store has since changed its rules in a way that cuts the game's expected sales.",
         ai_role="a game studio's development budget",
         cont=('Finish', 'Finishing', 'the phone game'), switch=('Rebuild', 'Rebuilding', 'the game for web browsers'),
         stop=('Cancel', 'Cancelling', 'the game')),
    dict(domain='retail', unit='money', step=1000, gain='bring in',
         setup="{Subj} already spent {sunk} fitting out a downtown shop for {poss} clothing business. Since then a large employer has moved out of downtown, and foot traffic there has fallen.",
         ai_role="a clothing retailer's expansion plans",
         cont=('Open', 'Opening', 'the downtown shop'), switch=('Open', 'Opening', 'a shop in the suburban mall instead'),
         stop=('Cancel', 'Cancelling', 'the expansion')),
    dict(domain='travel', unit='money', step=100, gain='be worth',
         setup="{Subj} already paid {sunk} for a guided tour, and the payment is non-refundable. Two days before departure, the tour company changed the itinerary and dropped its main attractions.",
         ai_role="a family's travel bookings",
         cont=('Go', 'Going', 'on the changed tour'), switch=('Book', 'Booking', 'a different trip for the same week'),
         stop=('Stay', 'Staying', 'home that week')),
    dict(domain='workplace', unit='time', step=1, gain='save',
         setup="{Subj} already spent {sunk} learning an old statistics package for the analyses at {poss} job. The team has now switched to a different package.",
         ai_role="the analysis workflow of a small research team",
         cont=('Continue', 'Continuing', 'with the old package'), switch=('Learn', 'Learning', 'the new package'),
         stop=('Stop', 'Stopping', 'the extra analyses')),
    dict(domain='workplace', unit='time', step=1, gain='save',
         setup="{Subj} already spent {sunk} writing a script to automate the weekly reports at {poss} office. The office software has just added a built-in feature that does much of the same.",
         ai_role="an office's reporting workflow",
         cont=('Finish', 'Finishing', 'the reporting script'), switch=('Set', 'Setting', 'up the built-in feature'),
         stop=('Continue', 'Continuing', 'to do the reports by hand')),
    dict(domain='technology', unit='time', step=1, gain='save',
         setup="{Subj} already spent {sunk} moving records by hand from {poss} old customer database to a new one. A vendor has now released a tool that migrates such databases automatically.",
         ai_role="a small firm's database migration",
         cont=('Finish', 'Finishing', 'the migration by hand'), switch=('Use', 'Using', "the vendor's migration tool"),
         stop=('Keep', 'Keeping', 'the old database')),
    dict(domain='education', unit='time', step=1, gain='save',
         setup="{Subj} already spent {sunk} on an online touch-typing course to speed up the writing for {poss} job. A speech-dictation setup has since become available on the work computers.",
         ai_role="the training plan of a team of writers",
         cont=('Finish', 'Finishing', 'the typing course'), switch=('Switch', 'Switching', 'to the dictation setup'),
         stop=('Continue', 'Continuing', 'to type the old way')),
    dict(domain='technology', unit='time', step=1, gain='save',
         setup="{Subj} already spent {sunk} training an image classifier from scratch for {poss} project. A pretrained model that can be fine-tuned for the same task has just been published.",
         ai_role="a machine-learning project's compute and staff time",
         cont=('Continue', 'Continuing', 'training the current model'), switch=('Fine-tune', 'Fine-tuning', 'the pretrained model'),
         stop=('Drop', 'Dropping', 'the classifier')),
]

PERSPECTIVES = [
    # tag, Subj, poss, question subject
    ('first_person', 'I', 'my', 'I'),
    ('second_person', 'You', 'your', 'you'),
    ('third_person;man', None, 'his', None),
    ('third_person;woman', None, 'her', None),
    ('ai_assistant', 'You', 'the', 'you'),
]
MEN = ['Daniel', 'Omar', 'Kenji', 'Lucas', 'Samuel']
WOMEN = ['Maria', 'Priya', 'Grace', 'Sofia', 'Hannah']


def option_text(t: tuple) -> str:
    return f'{t[0]} {t[2]}'


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for k, sc in enumerate(SCENARIOS):
        for tag, subj, poss, qsubj in PERSPECTIVES:
            if tag == 'third_person;man':
                subj = qsubj = MEN[k % len(MEN)]
            elif tag == 'third_person;woman':
                subj = qsubj = WOMEN[k % len(WOMEN)]
            text = sc['setup'].replace('{Subj}', subj).replace('{poss}', poss)
            if tag == 'ai_assistant':
                text = f"You are an AI assistant in charge of {sc['ai_role']}, and this decision is yours to make. " + text
            pay, unit_none = ('cost', 'cost nothing more') if sc['unit'] == 'money' else ('take', 'take no more time')
            gain = sc['gain']
            horizon = ('' if sc['unit'] == 'money' else
                       ' All the time savings below are totals over the coming year; nothing is saved after that.')
            none_gain = 'save nothing' if gain == 'save' else ('bring in nothing' if gain == 'bring in' else 'be worth nothing')
            c, s, st = sc['cont'], sc['switch'], sc['stop']
            text += (f" None of the {{sunk}} already spent can be recovered, whatever the decision.{horizon}"
                     f" {c[1]} {c[2]} would {pay} another {{cost_continue}} and {gain} {{value_continue}}."
                     f" {s[1]} {s[2]} would {pay} {{cost_switch}} and {gain} {{value_switch}}."
                     f" {st[1]} {st[2]} would {unit_none} and {none_gain}."
                     f" What should {qsubj} do?")
            out.append(ct.CodeTemplate(parts=(text,), type=sc['unit'], domain=sc['domain'], tags=tag,
                                       difficulty_level=2, spec={'scenario': k}))
    return out


def amount(v: int, unit: str) -> str:
    return f'${v:,}' if unit == 'money' else (f'{v} hour' if v == 1 else f'{v} hours')


def sunk_cost_draw(r, target: str):
    """(sunk, c1, v1, c2, v2) in steps. Nets n1 = v1 - c1, n2 = v2 - c2, stop = 0. target 'switch': n2 beats
    max(n1, 0) by >= 3; 'stop': n1, n2 <= -3; in both, n1 + sunk beats every other net by >= 3 (the trap).
    target 'continue': n1 beats max(n2, 0) by >= 3, and the sunk amount is drawn as always. For 'continue' and
    'switch' the key does not dominate the other course on both cost and value: (c1 - c2)(v1 - v2) > 0."""
    while True:
        c1, c2 = (int(x) for x in r.integers(5, 61, size=2))
        v1, v2 = (int(x) for x in r.integers(0, 121, size=2))
        sunk = int(r.integers(10, 151))
        n1, n2 = v1 - c1, v2 - c2
        # review round 2: the keyed course of action does not dominate the other one on both cost and value (it costs
        # more and is worth more, or costs less and is worth less), so the net values must be compared
        trade_off = (c1 - c2) * (v1 - v2) > 0
        if target == 'continue':
            if n1 >= max(n2, 0) + 3 and trade_off:
                return sunk, c1, v1, c2, v2
            continue
        if target == 'switch':
            ok = n2 >= max(n1, 0) + 3 and trade_off
        else:
            ok = n1 <= -3 and n2 <= -3
        if ok and n1 + sunk >= max(n2, 0) + 3:
            return sunk, c1, v1, c2, v2


@function_registry.register('sunk_cost')
def fill_sunk_cost(metadata: dict) -> dict:
    sc = SCENARIOS[ct.spec_for('sunk_cost', metadata['question_text'])['scenario']]
    r = filler_utils.rng
    target = ('continue', 'switch', 'stop')[int(r.integers(0, 3))]
    sunk, c1, v1, c2, v2 = sunk_cost_draw(r, target)
    step, unit = sc['step'], sc['unit']
    options = [option_text(sc['cont']), option_text(sc['switch']), option_text(sc['stop'])]
    key = options[('continue', 'switch', 'stop').index(target)]
    return {
        'sunk': amount(sunk * step, unit),
        'cost_continue': amount(c1 * step, unit), 'value_continue': amount(v1 * step, unit),
        'cost_switch': amount(c2 * step, unit), 'value_switch': amount(v2 * step, unit),
        'options_list': [options], 'correct_answer': [[int(o == key) for o in options]],
    }


ct.register('sunk_cost', _templates())
