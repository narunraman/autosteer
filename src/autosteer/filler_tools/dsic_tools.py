"""dsic_mechanism (STEER 4.3.a, Dominant Strategy Incentive Compatibility): hand-written code templates.

Redesign of 6 October 2026 (the earlier design, a single item sold by a first-, second- or third-price rule, was too
easy: the key followed from the payment rule's name). Every mechanism is described in words, never named, in four
settings, and every question states its assumptions: private values (each participant alone knows their value,
favourite spot or ranking), quasi-linear utility where money changes hands (value received minus payment), single-
peaked preferences on a line for facility location, strict rankings for matching. The question is Yes/No: is
reporting truthfully (bidding one's value, staying in until the price reaches it, accepting a posted price exactly
when it is worth it, reporting one's true value / spot / ranking) a dominant strategy for every participant (for
deferred acceptance: for every participant on the side asked about)?

    type                 setting            DSIC  why
    auction_second       auction (1 item)   Yes   the price is the best competing bid, which one's own bid does not set
    auction_reserve      auction            Yes   the same with a minimum price announced in advance
    auction_clock        auction            Yes   rising price, drop-outs final: leaving before one's value can only
                                                  lose a profitable purchase, staying past it can only buy at a loss
    auction_posted       auction            Yes   a price fixed in advance: accepting exactly when value >= price is
                                                  optimal whatever the others do
    auction_first        auction            No    the winner pays their bid: shading the bid raises the payoff
    auction_third        auction            No    (3 or more bidders) value 10, others 12 and 3: bidding 10 loses,
                                                  bidding 13 wins at price 3
    auction_allpay       auction            No    everyone pays their bid: a losing truthful bid is pure loss
    auction_lottery      auction            No    drawn with chance proportional to the bid and pays the bid: a
                                                  truthful bidder's payoff is 0 when drawn, a lower bid does better
    project_pivot        public project     Yes   build iff reported values cover the cost, equal cost shares, and a
                                                  pivotal reporter pays the others' net loss (Clarke pivot / VCG)
    project_pledge       public project     No    build iff reports cover the cost, cost split in proportion to the
                                                  reports: a lower report lowers one's share while it still gets built
    project_equal        public project     No    build iff reports cover the cost, equal shares: a resident whose
                                                  value is below the share reports 0 to block it, one above it
                                                  exaggerates to get it built
    location_median      facility on a line Yes   the middle report (odd number of reports): moving one's report
                                                  either leaves the middle where it is or moves it away
    location_mean        facility on a line No    the average: anyone whose spot differs from it pulls it closer by
                                                  exaggerating
    match_serial         assignment         Yes   random order, each in turn takes their reported favourite among the
                                                  objects left: one's report only matters at one's own turn
    match_cycles         housing market     Yes   each owns one object, repeatedly point to the owner of one's
                                                  favourite remaining object and trade along cycles (top trading cycles)
    match_immediate      school choice      No    first choices admitted first and for good (immediate acceptance /
                                                  Boston): ranking a safer school first can avoid losing it
    match_da_proposers   two-sided          Yes   deferred acceptance, asked about the proposing side
    match_da_receivers   two-sided          No    the same rule, asked about the receiving side
    location_misreport   facility on a line (counterexample type, 3 options) a profile of favourite spots, everyone else
                                                  truthful: can a named resident get the facility closer by
                                                  reporting a spot further east, further west, or not at all? Key by
                                                  brute force over every whole-number report.

Nine Yes and nine No types with the same number of templates each, so the Yes/No keys are balanced. The single-item
sealed-bid types with an outcome fixed by the bids (second, reserve, first, third, all-pay) add, in one template in
three, a follow-up part with stated bids: winner and payment, the seller's revenue, or a named bidder's payoff.
Perspectives: anon (the rule is described) and ai_assistant (the reader advises the organiser). Every amount is a
whole number of dollars. The type names avoid those of bic_mechanism (a separate element).
"""
from __future__ import annotations

import random

from . import code_templates as ct
from .fill_registry import FillRegister
from .mechanism_templates import AUCTIONS
from .utilities import filler_utils

function_registry = FillRegister()

ELEMENT = 'dsic_mechanism'
NUM_WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}
PEOPLE = [('Ana', 'her'), ('Bo', 'his'), ('Carmen', 'her'), ('Dev', 'his'), ('Elif', 'her'), ('Farid', 'his'),
          ('Greta', 'her'), ('Hiro', 'his'), ('Ines', 'her'), ('Jonas', 'his'), ('Keira', 'her'), ('Luca', 'his'),
          ('Mina', 'her'), ('Nikhil', 'his'), ('Olga', 'her'), ('Pavel', 'his')]

DSIC = {'auction_second': True, 'auction_reserve': True, 'auction_clock': True, 'auction_posted': True,
        'auction_first': False, 'auction_third': False, 'auction_allpay': False, 'auction_lottery': False,
        'project_pivot': True, 'project_pledge': False, 'project_equal': False,
        'location_median': True, 'location_mean': False,
        'match_serial': True, 'match_cycles': True, 'match_immediate': False,
        'match_da_proposers': True, 'match_da_receivers': False}
SETTING = {t: t.split('_')[0] for t in DSIC}
DOMAIN = {'auction': 'auction', 'project': 'public_project', 'location': 'facility_location', 'match': 'matching'}
DIFFICULTY = {'auction': 8, 'project': 9, 'location': 8, 'match': 9}
FOLLOWUP_TYPES = ('auction_second', 'auction_reserve', 'auction_first', 'auction_third', 'auction_allpay')
FOLLOWUPS = ('winner_payment', 'revenue', 'bidder_payoff')
PERSPECTIVES = ('anon', 'ai_assistant')

# ---------------------------------------------------------------------------------------------- scenarios
# single item: two scenarios per AUCTIONS domain of mechanism_templates (bidders, item, seller)
AUCTION_SCENES = [sc for scs in AUCTIONS.values() for sc in scs[1:3]]
# public projects: (group, member, members, project)
PROJECT_SCENES = [
    ('the households on Cedar Lane', 'household', 'households', 'a shared electric-vehicle charger'),
    ('the tenants of a small apartment building', 'tenant', 'tenants', 'a rooftop garden'),
    ('the members of a community choir', 'member', 'members', 'a new piano'),
    ('the farmers along an irrigation canal', 'farmer', 'farmers', 'a new pump'),
    ('the shop owners in a covered market', 'shop owner', 'shop owners', 'a security camera system'),
    ('the teachers at a high school', 'teacher', 'teachers', 'a coffee machine for the staff room'),
    ('the owners of the cabins on Pine Ridge', 'cabin owner', 'cabin owners', 'a paved access road'),
    ('the players of an amateur cricket team', 'player', 'players', 'a bowling machine'),
]
# facility location: (residents, resident, line, length in km or m, unit, facility)
LOCATION_SCENES = [
    ('families', 'family', 'a 12-kilometre country road', 12, 'kilometre', 'bus stop'),
    ('lifeguard volunteers', 'volunteer', 'a 10-kilometre stretch of beach', 10, 'kilometre', 'first-aid station'),
    ('shop owners', 'shop owner', 'a 20-block market street', 20, 'block', 'recycling point'),
    ('cabin owners', 'cabin owner', 'the 16-kilometre shore of a long, narrow lake', 16, 'kilometre', 'boat launch'),
    ('farmers', 'farmer', 'a 14-kilometre irrigation canal', 14, 'kilometre', 'water pump'),
    ('employees', 'employee', 'a 30-metre office corridor', 30, 'metre', 'printer'),
    ('residents', 'resident', 'an 18-kilometre valley road', 18, 'kilometre', 'mailbox'),
    ('campers', 'camper', 'a 24-kilometre hiking trail', 24, 'kilometre', 'water tank'),
]
# one-sided assignment: (people, person, objects, object, organiser)
ASSIGN_SCENES = [
    ('students', 'student', 'dorm rooms', 'dorm room', 'the housing office'),
    ('interns', 'intern', 'project topics', 'project topic', 'the lab manager'),
    ('nurses', 'nurse', 'holiday shifts', 'holiday shift', 'the ward manager'),
    ('employees', 'employee', 'parking spots', 'parking spot', 'the office manager'),
    ('volunteers', 'volunteer', 'festival stations', 'festival station', 'the festival organiser'),
    ('players', 'player', 'positions on the team', 'position', 'the coach'),
    ('researchers', 'researcher', 'telescope nights', 'telescope night', 'the observatory'),
    ('chefs', 'chef', 'kitchen stations', 'kitchen station', 'the head chef'),
]
# housing market: (people, person, objects, object, organiser); each starts owning one object
HOUSING_SCENES = [
    ('students', 'student', 'dorm rooms', 'dorm room', 'the housing office'),
    ('tenants', 'tenant', 'allotment plots', 'allotment plot', 'the garden society'),
    ('vendors', 'vendor', 'market stalls', 'market stall', 'the market board'),
    ('employees', 'employee', 'desks', 'desk', 'the office manager'),
    ('fishers', 'fisher', 'moorings', 'mooring', 'the harbour office'),
    ('teachers', 'teacher', 'classrooms', 'classroom', 'the principal'),
    ('artists', 'artist', 'studio spaces', 'studio space', 'the arts centre'),
    ('families', 'family', 'cabins at a summer camp', 'cabin', 'the camp office'),
]
# school choice: (students, student, schools, school, organiser)
SCHOOL_SCENES = [
    ('children', 'family', 'public primary schools', 'school', 'the school district'),
    ('students', 'student', 'high schools', 'high school', 'the city'),
    ('students', 'student', 'summer camps', 'camp', 'the recreation department'),
    ('students', 'student', 'elective courses', 'course', 'the college'),
    ('graduates', 'graduate', 'training programmes', 'programme', 'the ministry'),
    ('residents', 'resident', 'community gardens', 'garden', 'the parks board'),
    ('trainees', 'trainee', 'clinics', 'clinic', 'the health authority'),
    ('children', 'family', 'kindergartens', 'kindergarten', 'the municipality'),
]
# two-sided: (proposers, proposer, receivers, receiver, organiser)
TWO_SIDED_SCENES = [
    ('new employees', 'new employee', 'mentors', 'mentor', 'the company'),
    ('students', 'student', 'tutors', 'tutor', 'the tutoring centre'),
    ('apprentices', 'apprentice', 'master craftspeople', 'master craftsperson', 'the guild'),
    ('junior doctors', 'junior doctor', 'senior consultants', 'senior consultant', 'the hospital'),
    ('volunteers', 'volunteer', 'team leaders', 'team leader', 'the charity'),
    ('graduate students', 'graduate student', 'advisers', 'adviser', 'the department'),
    ('language learners', 'language learner', 'conversation partners', 'conversation partner', 'the language club'),
    ('rookies', 'rookie', 'veteran players', 'veteran player', 'the league'),
]

# ---------------------------------------------------------------------------------------------- rule texts
A_ASSUME = ('Each bidder alone knows how much the item is worth to them, and cares only about that value (if they '
            'get the item) minus whatever they pay. ')
A_Q = ['Is bidding their true value a dominant strategy for every bidder under this rule?',
       'Under this rule, is it a dominant strategy for each bidder to bid exactly what the item is worth to them, '
       'whatever the others bid?']
SEALED = '{Seller} is selling {item} to {n} {bidders}, who each submit one sealed bid. '
AUCTION_RULES = {
    'auction_second': ['The highest bid wins, and the winner pays the highest of the other bids; nobody else pays. ',
                       'The item goes to the highest bidder, who pays the amount of the best competing bid; the '
                       'others pay nothing. '],
    'auction_reserve': ['Before the bids are opened, {seller} announces a minimum price of {reserve}. If no bid '
                        'reaches it, the item is not sold; otherwise the highest bid wins and the winner pays the '
                        'larger of {reserve} and the highest of the other bids. Nobody else pays. ',
                        'The item goes to the highest bidder provided the bid is at least {reserve}, an amount fixed '
                        'and announced in advance; the winner pays {reserve} or the best competing bid, whichever is '
                        'larger. The others pay nothing, and if every bid is below {reserve} the item stays unsold. '],
    'auction_first': ['The highest bid wins, and the winner pays the amount of their own bid; nobody else pays. ',
                      'The item goes to the highest bidder, who pays exactly what they offered; the others pay '
                      'nothing. '],
    'auction_third': ['The highest bid wins, and the winner pays the third-highest of all the bids; nobody else '
                      'pays. ',
                      'The item goes to the highest bidder, who pays the amount of the bid ranked third from the '
                      'top; the others pay nothing. '],
    'auction_allpay': ['The highest bid wins the item, and every bidder, whether they win or not, pays the amount of '
                       'their own bid. ',
                       'Every bidder must pay their own bid whatever happens; the item goes to the highest bid. '],
    'auction_lottery': ['The winner is then drawn at random, each bidder\'s chance of being drawn being proportional '
                        'to their bid, and the bidder drawn pays their own bid; nobody else pays. ',
                        'A random draw decides who gets the item: a bidder who bids twice as much is twice as likely '
                        'to be drawn. The bidder drawn pays their own bid; the others pay nothing. '],
}
CLOCK = ['{Seller} is selling {item} to {n} {bidders}. The asking price starts at $0 and rises continuously, and a '
         'bidder still in may leave at any price; everyone sees who leaves, and leaving is final. When only one '
         'bidder remains, that bidder buys the item at the price at which the last other bidder left. If all the '
         'remaining bidders leave at the same price, one of them, drawn at random, buys the item at that price. ',
         '{Seller} offers {item} to {n} {bidders} by calling out a price that rises continuously from zero. Bidders '
         'drop out at any price they choose, publicly and for good, and the last bidder left buys the item at the '
         'price reached when the second-to-last bidder dropped out; if the last bidders drop out at the same price, '
         'one of them, drawn at random, buys the item at that price. ']
CLOCK_Q = ['Is it a dominant strategy for every bidder to stay in until the price reaches their own value, and to '
           'leave at that point?',
           'Whatever the others do, is it a dominant strategy for each bidder to drop out exactly when the price '
           'reaches what the item is worth to them?']
POSTED = ['Before meeting any bidders, {seller} fixes a price of {price} for {item}. The {n} {bidders} are asked one '
          'at a time, in a random order, whether they will buy at that price; the first who says yes buys the item '
          'for {price}. ',
          '{Seller} offers {item} to {n} {bidders} at a take-it-or-leave-it price of {price}, fixed in advance and '
          'the same for everyone. The bidders are approached in a random order and the first to accept buys the '
          'item for {price}. ']
POSTED_Q = ['Is it a dominant strategy for every bidder to say yes exactly when the item is worth at least {price} to '
            'them?',
            'Whatever the others do, is it a dominant strategy for each bidder to accept if and only if their value '
            'is at least {price}?']

P_ASSUME = ('Each {member} alone knows how much {the_project} is worth to them in dollars, and cares only about that '
            'value (if it is bought) minus whatever they pay. ')
P_Q = ['Is reporting their true value a dominant strategy for every {member} under this rule?',
       'Under this rule, is it a dominant strategy for each {member} to report exactly what {the_project} is worth to '
       'them, whatever the others report?']
P_INTRO = 'The {n} {members} ({group}) are deciding whether to buy {project}, which costs {cost}. '
PROJECT_RULES = {
    'project_pivot': ['Each {member} reports a value. It is bought exactly when the reported values add up to at '
                      'least {cost}, and then every {member} pays an equal share of {share}. In addition, a {member} '
                      'pays a fee when their report changes the decision: when the other {members}\' reported values, '
                      'each minus its {share} share, add up to zero or more but {the_project} is not bought, or add up to '
                      'less than zero but it is bought. The fee is the size of that sum, ignoring its sign. ',
                      'Each {member} reports a value, and {the_project} is bought exactly when the reports add up to at '
                      'least {cost}; if it is, the cost is shared equally, {share} each. Then, for each {member}, '
                      'take the others\' reports minus their {share} shares and add them up: that total alone would '
                      'have bought it if it is zero or more, and not otherwise. If the actual decision is different '
                      'from that one, the {member} pays a fee equal to the size of the total, ignoring its sign. '],
    'project_pledge': ['Each {member} reports a value. It is bought exactly when the reports add up to at least '
                       '{cost}; then the cost is split in proportion to the reports, so a {member} whose report is a '
                       'fifth of the total pays a fifth of {cost}. If it is not bought, nobody pays. ',
                       'Each {member} reports a value; {the_project} is bought when the reported values add up to at '
                       'least {cost}, and each {member} then pays the share of {cost} equal to their share of the '
                       'reported total. Nobody pays if it is not bought. '],
    'project_equal': ['Each {member} reports a value. It is bought exactly when the reports add up to at least '
                      '{cost}, and then every {member} pays an equal share of {share}; if it is not bought, nobody '
                      'pays. ',
                      'Each {member} reports a value; {the_project} is bought when the reported values add up to at '
                      'least {cost}, in which case the cost is split equally, {share} each. Nobody pays otherwise. '],
}

L_ASSUME = ('Each {resident} has a favourite spot, which only they know, and prefers the {facility} closer '
            'to that spot, in either direction: the nearer the better. No money changes hands. ')
L_Q = ['Is reporting their true favourite spot a dominant strategy for every {resident} under this rule?',
       'Under this rule, is it a dominant strategy for each {resident} to report exactly their favourite spot, '
       'whatever the others report?']
L_INTRO = ('The {n} {residents} along {line} must choose one place for a shared {facility}. Positions are measured in '
           '{unit}s from the west end. ')
LOCATION_RULES = {
    'location_median': ['Each {resident} reports a spot, and the {facility} is placed at the report in the middle when '
                        'the reports are put in order from west to east. ',
                        'Each {resident} names a spot; the reports are sorted from west to east and the {facility} '
                        'goes to the middle one, with as many reports to its west as to its east. '],
    'location_mean': ['Each {resident} reports a spot, and the {facility} is placed at the average of the reported '
                      'positions. ',
                      'Each {resident} names a spot; the {facility} goes to the mean position of all the reports. '],
}

M_ASSUME = ('Each {person} has a strict ranking of the {objects}, which only they know, and only cares about which '
            '{object} they end up with. No money changes hands. ')
M_Q = ['Is reporting their true ranking a dominant strategy for every {person} under this rule?',
       'Under this rule, is it a dominant strategy for each {person} to report their true ranking, whatever the '
       'others report?']
SERIAL = ['{Organiser} must give {n} {people} one {object} each, out of {n} {objects}. Each {person} reports a '
          'ranking of all the {objects}. Then a random order of the {people} is drawn, and in that order each takes '
          'the highest-ranked {object} on their reported list that is still available. ',
          '{Organiser} assigns {n} {objects} to {n} {people}, one each. Everyone hands in a ranking of the '
          '{objects}; a lottery then sets an order, and each {person} in turn gets their best-ranked remaining '
          '{object} according to the list they handed in. ']
CYCLES = ['Each of {n} {people} currently has one of {n} {objects}, and {organiser} lets them swap. Each {person} '
          'reports a ranking of all the {objects}, including their own. Then, repeatedly: each {person} still in '
          'points to the owner of their favourite remaining {object}; there is always at least one cycle of '
          'pointing, and everyone in a cycle receives the {object} they point to and leaves with it. This continues '
          'until everyone has left. ',
          '{Organiser} runs a swap among {n} {people}, each of whom holds one of {n} {objects}. Everyone reports a '
          'ranking of all the {objects}. In each round, every remaining {person} points to whoever holds their '
          'top remaining {object}; the {people} in each cycle of pointing trade along it and leave, and the rest '
          'repeat with the {objects} still there. ']
M2_ASSUME = ('Each {student} has a strict ranking of the {schools}, which only they know, and only cares about '
             'which one they are admitted to. No money changes hands. ')
IMMEDIATE = ['{Organiser} assigns {n} {students} to three {schools}, which have {seats} seats respectively, each '
             'with a fixed priority order over the {students} (by distance). Each {student} reports a ranking of the {schools}. '
             'In the first round each {school} admits, in priority order, the {students} who ranked it first, up to '
             'its seats, and these admissions are final. In the second round the {students} not yet admitted apply '
             'to the {school} they ranked second, if it still has seats, and so on. ',
             '{Organiser} fills the seats of three {schools} ({seats} seats respectively) with {n} {students}. Each '
             '{school} ranks the {students} '
             'by a fixed priority. Every {student} hands in a ranking of the {schools}. Round one: each {school} '
             'takes the {students} who listed it first, best priority first, until it is full, and those places '
             'are final. Round two: the {students} left over are considered only by the {school} they listed '
             'second, if it has room; then their third choice. ']
IMMEDIATE_Q = ['Is reporting their true ranking a dominant strategy for every {student} under this rule?',
               'Under this rule, is it a dominant strategy for each {student} to report their true ranking, '
               'whatever the others report?']
D_ASSUME = ('Every {proposer} and every {receiver} has a strict ranking of the people on the other side, which only '
            'they know, and only cares about whom they are matched with. No money changes hands. ')
DA = ['{Organiser} pairs {n} {proposers} with {n} {receivers}, one each. Everyone reports a ranking of the other '
      'side. Then: each {proposer} who is not tentatively matched asks the {receiver} they rank highest among those '
      'who have not yet turned them down; each {receiver} keeps the best request received so far, according to '
      'their reported ranking, and turns down the rest. This repeats until no {proposer} is turned down, and the '
      'tentative pairs become final. ',
      '{Organiser} matches {n} {proposers} to {n} {receivers}. Both sides hand in rankings of the other. In each '
      'round every unmatched {proposer} applies to their best-ranked {receiver} who has not rejected them yet; each '
      '{receiver} holds on to the applicant they rank highest so far and rejects the others; held applicants can '
      'still be rejected later for a better one. When no one is rejected, the held pairs are final. ']
DA_Q = {'match_da_proposers': ['Is reporting their true ranking a dominant strategy for every {proposer} under this '
                               'rule?',
                               'Under this rule, is it a dominant strategy for each {proposer} to report their true '
                               'ranking of the {receivers}, whatever everyone else reports?'],
        'match_da_receivers': ['Is reporting their true ranking a dominant strategy for every {receiver} under this '
                               'rule?',
                               'Under this rule, is it a dominant strategy for each {receiver} to report their true '
                               'ranking of the {proposers}, whatever everyone else reports?']}

AI = 'You are an AI assistant advising {organiser} on the following rule. '
FOLLOWUP_FRAMES = {
    'winner_payment': ['Suppose the bids are as follows: {bid_list}. Who wins, and how much do they pay?',
                       'The submitted bids are: {bid_list}. Under this rule, who wins and what do they pay?'],
    'revenue': ['Suppose the bids are as follows: {bid_list}. How much does the seller collect in total?',
                'The submitted bids are: {bid_list}. What is the seller\'s total revenue from this sale?'],
    'bidder_payoff': ['{bidder} values the item at {value} and bids {own_bid}. The other bids are: {bid_list}. What is '
                      '{bidder}\'s payoff (their value if they win, minus everything they pay)?',
                      'The other bids are: {bid_list}. {bidder}, whose value for the item is {value}, bids {own_bid}. '
                      'What payoff does {bidder} get (the value of the item if they win, minus all payments)?'],
}
MISREPORT_Q = ('Suppose the favourite spots are: {spots}, and everyone except {who} reports their favourite spot '
               'truthfully. Reports can be any spot on the line, from 0 to {length}. Could {who} get the {facility} '
               'placed closer to {pos} favourite spot by reporting some other spot?')
MISREPORT_OPTIONS = ['Yes: by reporting a spot further east than {pos} favourite spot',
                     'Yes: by reporting a spot further west than {pos} favourite spot',
                     'No: no other report places the {facility} closer to {pos} favourite spot']


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _fill(frame: str, words: dict) -> str:
    class Keep(dict):
        def __missing__(self, k):
            return '{' + k + '}'
    return frame.format_map(Keep(words))


def _part0(t: str, i: int, j: int, w: dict) -> str:
    """Rule text phrasing i and question phrasing j of type t, scenario words w."""
    s = SETTING[t]
    if s == 'auction':
        if t == 'auction_clock':
            body, q = CLOCK[i] + A_ASSUME, CLOCK_Q[j]
        elif t == 'auction_posted':
            body, q = POSTED[i] + A_ASSUME, POSTED_Q[j]
        else:
            body, q = SEALED + AUCTION_RULES[t][i] + A_ASSUME, A_Q[j]
    elif s == 'project':
        body, q = P_INTRO + PROJECT_RULES[t][i] + P_ASSUME, P_Q[j]
    elif s == 'location':
        body, q = L_INTRO + LOCATION_RULES[t][i] + L_ASSUME, L_Q[j]
    elif t == 'match_serial':
        body, q = SERIAL[i] + M_ASSUME, M_Q[j]
    elif t == 'match_cycles':
        body, q = CYCLES[i] + M_ASSUME, M_Q[j]
    elif t == 'match_immediate':
        body, q = IMMEDIATE[i] + M2_ASSUME, IMMEDIATE_Q[j]
    else:
        body, q = DA[i] + D_ASSUME, DA_Q[t][j]
    return _fill(body + q, w)


def _scene_words(t: str, sc) -> dict:
    s = SETTING[t]
    if s == 'auction':
        bidders, item, seller = sc
        return {'bidders': bidders, 'item': item, 'seller': seller, 'Seller': _cap(seller), 'organiser': seller}
    if s == 'project':
        group, member, members, project = sc
        return {'group': group, 'member': member, 'members': members, 'project': project,
                'the_project': 'the ' + project.split(' ', 1)[1], 'organiser': group}
    if s == 'location':
        residents, resident, line, _, unit, facility = sc
        return {'residents': residents, 'resident': resident, 'line': line, 'unit': unit, 'facility': facility,
                'organiser': f'the {residents} along {line}'}
    if t == 'match_immediate':
        students, student, schools, school, org = sc
        return {'students': students, 'student': student, 'schools': schools, 'school': school, 'organiser': org,
                'Organiser': _cap(org)}
    if t.startswith('match_da'):
        props, prop, recs, rec, org = sc
        return {'proposers': props, 'proposer': prop, 'receivers': recs, 'receiver': rec, 'organiser': org,
                'Organiser': _cap(org)}
    people, person, objects, obj, org = sc
    return {'people': people, 'person': person, 'objects': objects, 'object': obj, 'organiser': org,
            'Organiser': _cap(org)}


def _scenes(t: str) -> list:
    s = SETTING[t]
    if s == 'auction':
        return AUCTION_SCENES
    if s == 'project':
        return PROJECT_SCENES
    if s == 'location':
        return LOCATION_SCENES
    return {'match_serial': ASSIGN_SCENES, 'match_cycles': HOUSING_SCENES,
            'match_immediate': SCHOOL_SCENES}.get(t, TWO_SIDED_SCENES)


TEMPLATES_PER_TYPE = 32


def _templates() -> list[ct.CodeTemplate]:
    out = []
    for t in DSIC:
        s = SETTING[t]
        scenes = _scenes(t)
        k = 0
        combos = [(sc, i, j, p) for sc in range(len(scenes)) for i in range(2) for j in range(2) for p in PERSPECTIVES]
        random.Random(t).shuffle(combos)  # a fixed, varied subset of scenes and phrasings
        for sc, i, j, p in sorted(combos[:TEMPLATES_PER_TYPE]):
            w = _scene_words(t, scenes[sc])
            text = _part0(t, i, j, w)
            if p == 'ai_assistant':
                text = _fill(AI, w) + text
            kind = None
            if t in FOLLOWUP_TYPES and k % 3 == 2:
                kind = FOLLOWUPS[(k // 3) % 3]
            parts = (text,) + ((FOLLOWUP_FRAMES[kind][(sc + i) % 2],) if kind else ())
            out.append(ct.CodeTemplate(parts=parts, type=t, domain=DOMAIN[s], difficulty_level=DIFFICULTY[s],
                                       tags=p + (f';{kind}' if kind else ''), spec={'type': t, 'scene': sc,
                                                                                    'kind': kind}))
            k += 1
    for sc, scene in enumerate(LOCATION_SCENES):
        w = _scene_words('location_median', scene)
        for rule in ('location_median', 'location_mean'):
            for i in range(2):
                text = _fill(L_INTRO + LOCATION_RULES[rule][i] + L_ASSUME + MISREPORT_Q, w)
                out.append(ct.CodeTemplate(parts=(text,), type='location_misreport', domain='facility_location',
                                           difficulty_level=10, tags='anon',
                                           spec={'type': 'location_misreport', 'scene': sc, 'rule': rule}))
    return out


# ---------------------------------------------------------------------------------------------- helpers
def _rng():
    return filler_utils.rng


def _randint(a: int, b: int) -> int:
    return int(_rng().integers(a, b + 1))


def money(x: int) -> str:
    x = int(x)
    return f'-${-x:,}' if x < 0 else f'${x:,}'


def join_names(names: list) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]


def _people(n: int) -> list:
    idx = _rng().choice(len(PEOPLE), size=n, replace=False)
    return [PEOPLE[int(i)] for i in idx]


def _distinct_bids(n: int) -> list:
    """n distinct whole-dollar bids, multiples of 5, at least 4% apart."""
    while True:
        top = 5 * _randint(40, 2000)
        bids = sorted({5 * _randint(max(1, top // 10), top // 5) for _ in range(n)}, reverse=True)
        if len(bids) == n and all(a > b * 1.04 for a, b in zip(bids, bids[1:])):
            return bids


def auction_outcome(names: list, bids: list, rule: str, reserve: int = 0) -> tuple:
    """(winner or None, winner's payment, total revenue) of a sealed-bid rule with distinct bids."""
    order = sorted(range(len(bids)), key=lambda i: -bids[i])
    top, second, third = (bids[order[i]] for i in range(3))
    w = names[order[0]]
    if rule == 'auction_first':
        return w, top, top
    if rule == 'auction_second':
        return w, second, second
    if rule == 'auction_third':
        return w, third, third
    if rule == 'auction_reserve':
        if top < reserve:
            return None, 0, 0
        pay = max(reserve, second)
        return w, pay, pay
    if rule == 'auction_allpay':
        return w, top, sum(bids)
    raise ValueError(rule)


def _bid_list(names, bids) -> str:
    return join_names([f'{a} bids {money(b)}' for a, b in zip(names, bids)])


def _pick(correct: str, pool: list, k: int = 4):
    chosen = [correct]
    for i in _rng().permutation(len(pool)):
        x = pool[int(i)]
        if x not in chosen:
            chosen.append(x)
        if len(chosen) == k:
            return chosen, [1] + [0] * (k - 1)
    return None


def _followup(kind: str, rule: str, n: int, reserve: int):
    names = [p[0] for p in _people(n)]
    if kind in ('winner_payment', 'revenue'):
        bids = _distinct_bids(n)
        if rule == 'auction_reserve' and bids[0] < reserve:
            return None
        bids = [bids[int(i)] for i in _rng().permutation(n)]
        winner, pay, rev = auction_outcome(names, bids, rule, reserve)
        srt = sorted(bids, reverse=True)
        runner = names[bids.index(srt[1])]
        if kind == 'winner_payment':
            correct = f'{winner} wins and pays {money(pay)}'
            prices = list(dict.fromkeys(srt[:3] + ([reserve] if rule == 'auction_reserve' else [])))
            pool = [f'{winner} wins and pays {money(p)}' for p in prices] + \
                   [f'{runner} wins and pays {money(p)}' for p in prices]
        else:
            correct = money(rev)
            pool = [money(p) for p in srt] + [money(sum(srt)), money(sum(srt[:2]) // 2)] + \
                   ([money(reserve)] if rule == 'auction_reserve' else [])
        words = {'bid_list': _bid_list(names, bids)}
    else:
        others = _distinct_bids(n - 1)
        me = names[0]
        v = 5 * _randint(max(others) // 5 // 2 + 1, max(others) // 5 * 2)
        b = 5 * _randint(max(1, min(others) // 10), max(others) // 5 * 2)
        allb = [b] + others
        if len(set(allb)) < n or min(abs(b - o) for o in others) < 0.04 * max(allb):
            return None
        if rule == 'auction_reserve' and max(allb) < reserve:
            return None
        winner, pay, _ = auction_outcome(names, allb, rule, reserve)
        paid = pay if winner == me else (b if rule == 'auction_allpay' else 0)
        payoff = (v if winner == me else 0) - paid
        srt = sorted(allb, reverse=True)
        correct = money(payoff)
        pool = [money(v - p) for p in srt] + [money(0), money(v - b), money(v), money(-b)]
        words = {'bidder': me, 'value': money(v), 'own_bid': money(b), 'bid_list': _bid_list(names[1:], others)}
    picked = _pick(correct, pool)
    return None if picked is None else (words, picked)


def facility(rule: str, reports: list) -> float:
    s = sorted(reports)
    return s[len(s) // 2] if rule == 'location_median' else sum(reports) / len(reports)


def misreport_answer(rule: str, spots: list, who: int, length: int) -> str:
    """'east', 'west' or 'none': the direction of a report (a grid of hundredths from 0 to length) that strictly
    improves the asked resident's distance to the facility, everyone else truthful."""
    base = abs(facility(rule, spots) - spots[who])
    better = set()
    for k in range(100 * length + 1):
        rep = list(spots)
        rep[who] = k / 100
        if abs(facility(rule, rep) - spots[who]) < base - 1e-9:
            better.add('east' if rep[who] > spots[who] else 'west')
    assert len(better) <= 1, (rule, spots, who, better)
    return better.pop() if better else 'none'


def _fill_misreport(spec: dict) -> dict:
    scene = LOCATION_SCENES[spec['scene']]
    length = scene[3]
    rule = spec['rule']
    # mean: a resident away from the average (east or west of it, 40% each) or exactly at it (20%); interior spots
    want = None if rule == 'location_median' else ['east', 'east', 'west', 'west', 'none'][_randint(0, 4)]
    for _ in range(10000):
        n = _randint(1, 2) * 2 + 1
        spots = [_randint(1, length - 1) for _ in range(n)]
        who = _randint(0, n - 1)
        if want == 'none':
            others = spots[:who] + spots[who + 1:]
            if sum(others) % (n - 1):
                continue
            spots[who] = sum(others) // (n - 1)
        if len(set(spots)) < n:
            continue
        ans = misreport_answer(rule, spots, who, length)
        if want is not None and ans != want:
            continue
        break
    else:
        raise RuntimeError('dsic_mechanism: no misreport profile drawn')
    people = _people(n)
    names = [p[0] for p in people]
    order = sorted(range(n), key=lambda k: (spots[k], k))
    shown = join_names([f'{names[k]} at {spots[k]}' for k in order])
    pos = people[who][1]
    options = [o.format(pos=pos, facility=scene[5]) for o in MISREPORT_OPTIONS]
    key = options[{'east': 0, 'west': 1, 'none': 2}[ans]]
    return {'n': NUM_WORDS[n], 'spots': shown, 'who': names[who], 'pos': pos, 'length': str(length),
            'options_list': [options], 'correct_answer': [[int(o == key) for o in options]]}


@function_registry.register(ELEMENT)
def fill_dsic_mechanism(metadata: dict) -> dict:
    spec = ct.spec_for(ELEMENT, metadata['question_text'])
    t = spec['type']
    if t == 'location_misreport':
        return _fill_misreport(spec)
    s = SETTING[t]
    n = _randint(3, 5)
    out = {'n': NUM_WORDS[n]}
    reserve = 5 * _randint(20, 1000)
    if s == 'auction':
        out.update(reserve=money(reserve), price=money(5 * _randint(20, 2000)))
    elif s == 'project':
        share = 10 * _randint(5, 120)
        out.update(cost=money(share * n), share=money(share))
    elif s == 'location':
        n = _randint(1, 2) * 2 + 1  # odd, so the middle report is unique
        out['n'] = NUM_WORDS[n]
    elif t == 'match_immediate':
        n = _randint(4, 6)
        out['n'] = NUM_WORDS[n]
        # stated capacities, one seat per child in all, none with room for everyone, so they bind (re-review,
        # October 2026: "a limited number of seats" allowed capacities under which truth-telling is dominant)
        while True:
            caps = sorted((_randint(1, n - 2) for _ in range(3)), reverse=True)
            if sum(caps) == n:
                break
        out['seats'] = f'{caps[0]}, {caps[1]} and {caps[2]}'
    yes = DSIC[t]
    opts, ans = [['Yes', 'No']], [[int(yes), int(not yes)]]
    if spec['kind']:
        while True:
            got = _followup(spec['kind'], t, n, reserve)
            if got is not None:
                break
        words, (o1, a1) = got
        out.update(words)
        opts.append(o1)
        ans.append(a1)
    out['options_list'] = opts
    out['correct_answer'] = ans
    return out


ct.register(ELEMENT, _templates())
