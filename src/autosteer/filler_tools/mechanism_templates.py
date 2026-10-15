"""Build the question templates for ir_mechanism, budget_balance and auctions_risk (dsic_mechanism: dsic_tools.py).

    uv run python -m autosteer.filler_tools.mechanism_templates [element ...]   # writes data/templates/<element>/ (Parquet)

Templates are story frames x hand-written scenarios; numbers, names, rules and options come from the
fillers in mechanism_tools.py. The first scenario of every domain goes to the few-shot example pool
(with a worked explanation of the method); the rest form the test pool. Not a *_tools.py module, so the
filler registry does not import it.
"""
import os
import re

import pandas as pd

from autosteer.paths import templates_dir
from autosteer.steer_dataset import Dataset
from .mechanism_tools import GRADES

TAGS = 'explicit'

# ----------------------------------------------------------------------------- public-project scenarios
# (group, member, where, project, short, alternative-if-opting-out)
PROJECTS = {
    'community': [
        ('residents', 'resident', 'of Maple Grove', 'a new playground', 'the playground',
         'keeps taking their children to the park across town'),
        ('residents', 'resident', 'of Cedar Lane', 'a set of solar street lights for their block', 'the lights',
         'keeps relying on their own porch lights'),
        ('neighbors', 'neighbor', 'on Birch Street', 'a shared tool library', 'the tool library',
         'keeps renting tools from the hardware store'),
        ('residents', 'resident', 'of Harbor View', 'a community garden', 'the garden',
         'keeps buying vegetables at the market'),
        ('members', 'member', 'of the Riverside neighborhood association', 'a members-only walking path',
         'the path', 'keeps walking on the old trail'),
        ('residents', 'resident', 'of Pine Hollow', 'a shared snow-plowing contract', 'the plowing service',
         'keeps shoveling their own driveway'),
        ('neighbors', 'neighbor', 'in the Elm Court cul-de-sac', 'a gated bike shelter', 'the shelter',
         'keeps storing their bike in the garage'),
        ('residents', 'resident', 'of Willow Creek', 'a shuttle van to the train station', 'the shuttle',
         'keeps driving to the station'),
    ],
    'workplace': [
        ('employees', 'employee', 'at Brightline Logistics', 'an espresso machine for the break room',
         'the espresso machine', 'keeps buying coffee at the cafe downstairs'),
        ('staff members', 'staff member', 'at the Northgate design studio', 'a shared large-format printer',
         'the printer', 'keeps sending jobs to a print shop'),
        ('engineers', 'engineer', 'at Quarry Software', 'a team license for a code-review tool', 'the tool',
         'keeps using the free version'),
        ('nurses', 'nurse', 'on the night shift at Lakeside Clinic', 'a new lounge refrigerator',
         'the refrigerator', 'keeps bringing a cooler from home'),
        ('warehouse workers', 'warehouse worker', 'at Summit Supply', 'anti-fatigue mats for their stations',
         'the mats', 'keeps wearing cushioned insoles'),
        ('consultants', 'consultant', 'at Redwood Advisory', 'a shared booth at a trade fair', 'the booth',
         'keeps meeting clients at their own office'),
        ('drivers', 'driver', 'at Metro Cabs', 'a covered parking shelter', 'the shelter',
         'keeps parking on the street'),
        ('analysts', 'analyst', 'at Bayview Capital', 'an office gym corner', 'the gym corner',
         'keeps their own gym membership'),
    ],
    'school': [
        ('parents', 'parent', 'at Oakridge Elementary', 'an after-school science club', 'the club',
         'keeps sending their child to a weekend nature program'),
        ('students', 'student', 'in the Hillcrest robotics club', 'a competition robot kit', 'the kit',
         'keeps practicing with the old kit'),
        ('teachers', 'teacher', 'at Westbrook High', 'a shared classroom projector', 'the projector',
         "keeps borrowing the library's projector"),
        ('graduate students', 'graduate student', 'in the chemistry department', 'a lounge coffee machine',
         'the coffee machine', 'keeps buying coffee on campus'),
        ('parents', 'parent', 'at Sunnyside Preschool', 'a shaded play area', 'the play area',
         'keeps taking their child to the public park nearby'),
        ('students', 'student', 'in a university residence hall', 'a renovated study room', 'the study room',
         'keeps studying in the main library'),
        ('band members', 'band member', 'at Lincoln Middle School', 'a new drum set', 'the drums',
         'keeps renting drums for practice'),
        ('parents', 'parent', 'at Greenfield Academy', 'a school bus route extension', 'the bus route',
         'keeps driving their child to school'),
    ],
    'housing': [
        ('tenants', 'tenant', 'in the Ashford apartment building', 'a bike storage room', 'the bike room',
         'keeps their bike in their apartment'),
        ('condo owners', 'condo owner', 'at Seaside Towers', 'a rooftop terrace', 'the terrace',
         'keeps using the public beach'),
        ('housemates', 'housemate', 'in a shared house on Grove Road', 'a faster internet plan',
         'the new plan', 'keeps using their mobile data'),
        ('tenants', 'tenant', 'at Parkside Lofts', 'a laundry room upgrade', 'the new machines',
         'keeps going to the laundromat'),
        ('owners', 'owner', 'in the Hilltop townhouse complex', 'a shared electric-vehicle charger',
         'the charger', 'keeps charging at a public station'),
        ('residents', 'resident', 'of the Oak Terrace co-op', 'a parcel locker', 'the locker',
         'keeps picking parcels up at the post office'),
        ('tenants', 'tenant', 'in the Marlow building', 'a lobby water-filter station', 'the station',
         'keeps buying bottled water'),
        ('housemates', 'housemate', 'in a student house on Queen Street', 'a shared streaming bundle',
         'the bundle', 'keeps their own single-service subscription'),
    ],
    'online_platform': [
        ('moderators', 'moderator', 'of a hobby forum', 'a premium spam-filter service', 'the spam filter',
         'keeps deleting spam by hand'),
        ('streamers', 'streamer', 'in a gaming collective', 'a shared green-screen studio', 'the studio',
         'keeps streaming from home'),
        ('sellers', 'seller', 'in an online craft marketplace group', 'a joint advertising campaign',
         'the campaign', 'keeps running their own ads'),
        ('contributors', 'contributor', 'to an open-source project', 'a dedicated build server',
         'the build server', 'keeps building on their own laptop'),
        ('podcasters', 'podcaster', 'in an online network', 'a shared audio-editing license', 'the license',
         'keeps using free editing software'),
        ('writers', 'writer', 'on a newsletter platform', 'a shared design template pack', 'the template pack',
         'keeps using the default templates'),
        ('players', 'player', 'in an online gaming guild', 'a private game server', 'the server',
         'keeps playing on public servers'),
        ('members', 'member', 'of a photo-sharing club', 'a shared cloud-storage plan', 'the storage plan',
         'keeps storing photos on their own drive'),
    ],
    'health': [
        ('doctors', 'doctor', 'in the Valley health network', 'a shared diagnostic scanner', 'the scanner',
         'keeps referring patients to the city hospital'),
        ('doctors', 'doctor', 'at a group practice', 'a new online booking system', 'the booking system',
         'keeps booking appointments by phone'),
        ('pharmacists', 'pharmacist', 'in the Eastside pharmacy cooperative', 'a shared delivery van',
         'the van', 'keeps using a courier service'),
        ('physiotherapists', 'physiotherapist', 'at a rehabilitation center', 'a hydrotherapy pool',
         'the pool', 'keeps sending clients to the public pool'),
        ('patients', 'patient', 'in a diabetes support group', 'a shared nutrition coach', 'the coach',
         'keeps following free online guides'),
        ('dentists', 'dentist', 'in a shared office', 'a panoramic imaging machine', 'the imaging machine',
         'keeps referring patients to an imaging lab'),
        ('nurses', 'nurse', 'at Pinecrest Hospital', 'an on-site childcare room', 'the childcare room',
         'keeps using a private daycare'),
        ('residents', 'resident', 'of the Maplewood care home', 'a weekly visiting-musician program',
         'the music program', 'keeps listening to the radio'),
    ],
    'recreation': [
        ('members', 'member', 'of the Lakeside tennis club', 'resurfaced courts', 'the new courts',
         'keeps playing at the public courts'),
        ('players', 'player', 'on an amateur soccer team', 'floodlights for evening practice',
         'the floodlights', 'keeps practicing on weekend mornings'),
        ('members', 'member', 'of a hiking club', 'a shared camping-gear kit', 'the gear kit',
         'keeps renting gear'),
        ('members', 'member', 'of a rowing club', 'a new racing boat', 'the boat', 'keeps using the old boat'),
        ('members', 'member', 'of a chess club', 'a set of digital tournament clocks', 'the clocks',
         'keeps playing casual games without clocks'),
        ('climbers', 'climber', 'at a bouldering co-op', 'new climbing holds', 'the new holds',
         'keeps climbing on the old walls'),
        ('cyclists', 'cyclist', 'in a weekend riding group', 'a support van for long rides', 'the van',
         'keeps riding without support'),
        ('singers', 'singer', 'in a community choir', 'a digital piano for rehearsals', 'the piano',
         'keeps rehearsing with the old keyboard'),
    ],
    'environment': [
        ('farmers', 'farmer', 'along Mill River', 'a shared irrigation pump', 'the pump', 'keeps hauling water'),
        ('homeowners', 'homeowner', 'in Coral Bay', 'a shared rainwater tank', 'the tank',
         'keeps using town water for the garden'),
        ('homeowners', 'homeowner', 'in Fernwood', 'a food-scrap composting service', 'the service',
         'keeps putting food scraps in the trash'),
        ('shop owners', 'shop owner', 'on Market Street', 'a shared recycling compactor', 'the compactor',
         'keeps paying for regular trash pickup'),
        ('homeowners', 'homeowner', 'in the Ridge Road neighborhood', 'a community solar array',
         'the solar array', 'keeps buying all their power from the utility'),
        ('growers', 'grower', 'in the Sunvale orchard cooperative', 'a frost-protection wind machine',
         'the wind machine', 'keeps lighting smudge pots on cold nights'),
        ('fishers', 'fisher', 'at Port Elms', 'a shared ice machine', 'the ice machine',
         'keeps buying ice in town'),
        ('homeowners', 'homeowner', 'in Aspen Heights', 'a shared electric cargo bike', 'the cargo bike',
         'keeps using their car for errands'),
    ],
}

# ----------------------------------------------------------------------------- auction scenarios
# (bidders, item, seller)
AUCTIONS = {
    'art': [
        ('collectors', 'a landscape painting by a regional artist', 'the gallery'),
        ('museums', 'a rare woodblock print', 'the estate'),
        ('collectors', 'a bronze sculpture', 'the auction house'),
        ('dealers', 'a signed photograph', 'the photographer'),
        ('collectors', 'an early sketchbook of a famous illustrator', 'the family trust'),
        ('galleries', 'a large mural panel', 'the city arts council'),
        ('collectors', 'a hand-woven tapestry', 'the weaving guild'),
        ('dealers', 'a vintage film poster', 'the archive'),
    ],
    'real_estate': [
        ('developers', 'a vacant downtown lot', 'the city'),
        ('buyers', 'a lakeside cabin', 'the owner'),
        ('investors', 'a small office building', 'the bank'),
        ('families', 'a restored farmhouse', 'the estate agent'),
        ('developers', 'a former factory site', 'the county'),
        ('buyers', 'a seaside apartment', 'the seller'),
        ('investors', 'a parking garage', 'the transit authority'),
        ('restaurateurs', 'a corner storefront lease', 'the landlord'),
    ],
    'online_marketplace': [
        ('shoppers', 'a limited-edition sneaker pair', 'the online store'),
        ('gamers', 'a rare in-game item', 'the marketplace'),
        ('buyers', 'a refurbished laptop', 'the reseller'),
        ('fans', 'a signed concert poster', 'the fan-club site'),
        ('buyers', 'a vintage camera', 'the online seller'),
        ('advertisers', 'a banner slot on a busy website', 'the website'),
        ('buyers', 'a domain name', 'the registry'),
        ('shoppers', 'a designer handbag', 'the resale platform'),
    ],
    'government': [
        ('telecom firms', 'a radio-spectrum license', 'the regulator'),
        ('taxi operators', 'a taxi medallion', 'the city'),
        ('food vendors', 'a stall permit in the central square', 'the council'),
        ('energy firms', 'an offshore wind lease', 'the government'),
        ('bus operators', 'an airport shuttle concession', 'the airport authority'),
        ('mining firms', 'an exploration permit', 'the ministry'),
        ('fishing firms', 'a fishing quota', 'the fisheries agency'),
        ('broadcasters', 'a regional broadcast license', 'the regulator'),
    ],
    'collectibles': [
        ('collectors', 'a rare coin', 'the coin dealer'),
        ('fans', 'a game-worn jersey', 'the sports club'),
        ('collectors', 'a first-edition novel', 'the bookshop'),
        ('collectors', 'a vintage wristwatch', 'the jeweler'),
        ('hobbyists', 'a rare stamp', 'the stamp society'),
        ('collectors', 'a classic comic book', 'the comic store'),
        ('fans', 'a guitar signed by a rock band', 'the music charity'),
        ('collectors', 'an antique map', 'the antiques shop'),
    ],
    'agriculture': [
        ('ranchers', 'a prize-winning bull', 'the livestock market'),
        ('farmers', 'a used combine harvester', 'the equipment dealer'),
        ('vineyards', 'a lot of premium grapes', 'the grower'),
        ('bakers', 'a truckload of heritage wheat', 'the farm'),
        ('ranchers', 'a grazing lease', 'the land trust'),
        ('farmers', 'a water-rights allocation', 'the irrigation district'),
        ('breeders', 'a champion horse', 'the stable'),
        ('grocers', 'a crate of the first spring cherries', 'the orchard'),
    ],
    'charity': [
        ('donors', 'a dinner with a celebrity chef', 'the charity gala'),
        ('guests', 'a weekend at a mountain lodge', 'the school fundraiser'),
        ('donors', 'a signed football', 'the hospital foundation'),
        ('guests', 'a hand-made quilt', 'the community fundraiser'),
        ('donors', 'a private museum tour', 'the museum foundation'),
        ('guests', 'a private cooking class', 'the animal shelter'),
        ('donors', 'a front-row concert package', 'the arts charity'),
        ('guests', 'a hot-air balloon ride', 'the library fundraiser'),
    ],
    'vehicles': [
        ('dealers', 'a classic convertible', 'the auction house'),
        ('buyers', 'a retired city bus', 'the transit agency'),
        ('collectors', 'a vintage motorcycle', 'the estate'),
        ('dealers', 'a fleet pickup truck', 'the utility company'),
        ('buyers', 'a used sailboat', 'the marina'),
        ('collectors', 'a restored race car', 'the racing team'),
        ('buyers', 'a surplus police cruiser', 'the county'),
        ('dealers', 'an electric delivery van', 'the leasing company'),
    ],
}

_NUMBER_WORDS = re.compile(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b', re.I)

# ----------------------------------------------------------------------------- frames
# Placeholders filled here: {group} {member} {where} {project} {Project} {short} {alt}; the rest by the filler.
IR_FRAMES = [
    '{n_cap} {group} {where} are deciding on {project} costing {cost}. Values: {value_list}. Rule: {rule_text} '
    '{optout} Which {member} would be better off not participating?',
    'A group of {n} {group} {where} is considering {project}, which would cost {cost}. Their values for it are '
    '{value_list}. The group has agreed on this rule: {rule_text} {optout} Which {member} would be better off '
    'not participating?',
    '{Project} is proposed for {n} {group} {where}, at a total cost of {cost}. Each {member}\'s value for it: '
    '{value_list}. Payments are set as follows. {rule_text} {optout} Taking the rule as given, which {member} '
    'would be better off not participating?',
    'Consider {n} {group} {where} who could fund {project} (cost: {cost}). They value it at {value_list}. '
    'Rule: {rule_text} {optout} Which {member}, if any, would be better off not participating?',
    '{n_cap} {group} {where} have been asked to share the cost of {project}, priced at {cost}. Their values: '
    '{value_list}. Rule: {rule_text} {optout} Which {member} would be better off staying out?',
]
IR_OPTOUT = [
    'Each {member} can choose not to participate: one who opts out pays nothing and gets nothing from {short}, '
    'and the others\' payments do not change.',
    'Participation is voluntary: {a_member} who stays out pays nothing, cannot use {short}, and does not change '
    'what anyone else pays.',
    'Any {member} may decline to take part, paying nothing and getting nothing from {short}; the others\' '
    'payments stay the same.',
]
IR_OPTOUT_OUTSIDE = [
    'Each {member} can choose not to participate: one who opts out pays nothing, cannot use {short}, and instead '
    '{alt}, which is worth {outside_option} to each {member}. The others\' payments do not change.',
    'Participation is voluntary: {a_member} who stays out pays nothing and {alt} instead, an alternative worth '
    '{outside_option} to each {member}; nobody else\'s payment changes.',
    'Any {member} may decline to take part and pay nothing; they then {alt_bare}, which is worth '
    '{outside_option} to each {member}. The others\' payments stay the same.',
]

BB_FRAMES = [
    '{n_cap} {group} {where} are funding {project} that costs {cost}. {mechanism_text} {definition_text} '
    'How does the money the mechanism collects compare with the cost of {short}?',
    'To pay for {project} ({cost} in total), the {n} {group} {where} agreed on the following scheme. '
    '{mechanism_text} {definition_text} Is the scheme budget balanced: how does what it collects compare with '
    'the cost?',
    '{Project} for {n} {group} {where} will cost {cost}. {mechanism_text} {definition_text} Compare the total the '
    'mechanism collects with the cost. Is it budget balanced?',
    'A committee of {n} {group} {where} must raise {cost} for {project}. {mechanism_text} {definition_text} '
    'How does what this mechanism collects compare with what {short} costs?',
    'The {n} {group} {where} are considering {project}, costing {cost}. {mechanism_text} {definition_text} '
    'Which statement about the mechanism\'s budget is correct?',
]

AR_FRAMES = {
    'risk_ranking': [
        '{n_cap} {bidders} are bidding for {item} sold by {seller}. Each bidder\'s value is private and '
        'independently drawn from a uniform distribution between $0 and {V}. {attitude_text} {Seller} can use '
        'either {format_a} or {format_b}. Which format gives {seller} the higher expected revenue?',
        '{Seller} is selling {item} to {n} {bidders}, whose private values are independent draws from a uniform '
        'distribution between $0 and {V}. {attitude_text} It is choosing between {format_a} and {format_b}. '
        'Which should it choose to maximize its expected revenue?',
        'There are {n} {bidders} interested in {item}. Their values are private, independent and uniformly '
        'distributed between $0 and {V}. {attitude_text} {Seller} is deciding whether to sell it with {format_a} '
        'or with {format_b}. Which yields the higher expected revenue?',
        'Consider {seller} auctioning {item} to {n} {bidders} with independent private values, each uniform '
        'between $0 and {V}. {attitude_text} Comparing {format_a} with {format_b}, which format maximizes '
        '{seller}\'s expected revenue?',
        '{n_cap} {bidders} want {item}, which {seller} is putting up for auction. Values are private and drawn '
        'independently from a uniform distribution between $0 and {V}. {attitude_text} If {seller} must pick '
        '{format_a} or {format_b}, which brings in more revenue on average?',
    ],
    'expected_revenue': [
        '{n_cap} risk-neutral {bidders} are bidding for {item} sold by {seller}. Their values are private and '
        'independently drawn from a uniform distribution between $0 and {V}. {Seller} runs {format} with no '
        'reserve price, and the bidders follow the symmetric equilibrium. What is {seller}\'s expected revenue?',
        '{Seller} is selling {item} to {n} risk-neutral {bidders}, whose private values are independent draws '
        'from a uniform distribution between $0 and {V}. It uses {format} with no reserve price. Assuming '
        'equilibrium bidding, how much revenue does {seller} expect?',
        'There are {n} risk-neutral {bidders} interested in {item}. Each has a private value drawn independently '
        'and uniformly between $0 and {V}. {Seller} sells it with {format} and no reserve price. What is the '
        'expected selling price in equilibrium?',
        'Consider {seller} auctioning {item} to {n} risk-neutral {bidders} with independent private values, each '
        'uniform between $0 and {V}. The format is {format}, with no reserve price. In the symmetric '
        'equilibrium, what is the expected revenue?',
        '{n_cap} risk-neutral {bidders} want {item}. Values are private and drawn independently from a uniform '
        'distribution between $0 and {V}. If {seller} runs {format} with no reserve price, what revenue should '
        'it expect in equilibrium?',
    ],
    'equilibrium_bid': [
        '{n_cap} risk-neutral {bidders} are bidding for {item} sold by {seller}. Their values are private and '
        'independently drawn from a uniform distribution between $0 and {V}. {Seller} runs a first-price '
        'sealed-bid auction with no reserve price. In the symmetric Bayes-Nash equilibrium, how much does a '
        'bidder whose value is {v} bid?',
        '{Seller} is selling {item} to {n} risk-neutral {bidders} in a first-price sealed-bid auction with no '
        'reserve price. Values are private, independent and uniform between $0 and {V}. What does a bidder '
        'whose value is {v} bid in the symmetric equilibrium?',
        'In a first-price sealed-bid auction for {item}, {n} risk-neutral {bidders} have private values drawn '
        'independently and uniformly between $0 and {V}. There is no reserve price. In the symmetric Bayes-Nash '
        'equilibrium, what is the bid of a bidder whose value is {v}?',
        'Consider {seller} auctioning {item} to {n} risk-neutral {bidders}. Each bidder\'s private value is '
        'uniform between $0 and {V}, independently of the others. The winner pays their own bid (first-price, '
        'sealed bids, no reserve). How much should a bidder whose value is {v} bid in the symmetric equilibrium?',
        '{n_cap} risk-neutral {bidders} compete for {item} in a first-price sealed-bid auction run by {seller}, '
        'with no reserve price. Values are private and independently uniform between $0 and {V}. What is the '
        'symmetric equilibrium bid of a bidder whose value is {v}?',
    ],
}

# dsic_mechanism: its templates are now code templates in dsic_tools.py (October 2026 cleanup); the data
# directory's dsic_mechanism templates, written by an earlier version of this script, are not used.

EXPLANATIONS = {
    ('ir_mechanism', 'equal_split'): 'Each person pays the cost divided by the number of people. A person is '
    'better off not participating exactly when their value is below their payment (opting out gives zero).',
    ('ir_mechanism', 'proportional'): 'Compute each payment from the rule. With shares proportional to value, '
    'everyone keeps the same fraction of their value, so nobody loses as long as the values cover the cost. '
    'With shares proportional to usage points, a person pays (their points / total points) x cost; they are '
    'better off out if their value is below that payment.',
    ('ir_mechanism', 'outside_option'): 'Each person pays the cost divided by the number of people. A person is '
    'better off not participating when value minus payment is below the outside option they get by opting out.',
    ('ir_mechanism', 'vcg'): 'Subtract the equal cost share from each value and add the results to decide '
    'whether the project is built. A person is pivotal if the sum over the others has the opposite sign; they '
    'pay that sum\'s absolute value as a Clarke tax. Net utility is value minus cost share minus tax; a person '
    'with negative net utility is better off not participating (a pivotal person keeps the total surplus, '
    'which is never negative).',
    ('budget_balance', 'direct_transfers'): 'Add up all payments, subtract all rebates paid out, and compare the '
    'result with the cost: equal means exactly balanced, more is a surplus, less is a deficit.',
    ('budget_balance', 'fees'): 'The equal shares add up to exactly the cost, so the balance is the total of the '
    'extra fees collected minus the total of the credits paid out.',
    ('budget_balance', 'clarke'): 'The equal cost shares cover the cost exactly. Work out who is pivotal: a '
    'person is pivotal if the sum of everyone else\'s value minus cost share has the opposite sign to the '
    'decision. The Clarke taxes of pivotal people are extra money collected, so the mechanism runs a surplus '
    'equal to the total Clarke tax (exactly balanced if nobody is pivotal).',
    ('auctions_risk', 'risk_ranking'): 'With risk-neutral bidders, revenue equivalence makes all standard '
    'auctions earn the same. Risk-averse bidders bid more aggressively in first-price and Dutch auctions to '
    'reduce the risk of losing, so these earn more; risk-seeking bidders shade more there, so second-price and '
    'English auctions earn more. (Dutch = first-price and English = second-price strategically.)',
    ('auctions_risk', 'expected_revenue'): 'By revenue equivalence every standard auction earns the expected '
    'second-highest value. For n values uniform on [0, V] that is V(n-1)/(n+1).',
    ('auctions_risk', 'equilibrium_bid'): 'With n risk-neutral bidders and values uniform on [0, V], the '
    'symmetric first-price equilibrium bid is b(v) = v(n-1)/n, whatever V is.',
}


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _article(noun: str) -> str:
    return ('an ' if noun[0].lower() in 'aeiou' else 'a ') + noun


def _escape(s: str) -> str:
    return s.replace('{', '{{').replace('}', '}}')


def _fill(frame: str, words: dict) -> str:
    """Substitute scenario words, leaving filler placeholders intact."""
    class Keep(dict):
        def __missing__(self, k):
            return '{' + k + '}'
    return frame.format_map(Keep({k: _escape(v) for k, v in words.items()}))


def _ir_texts(sc):
    group, member, where, project, short, alt = sc
    words = {'group': group, 'member': member, 'a_member': _article(member), 'where': where,
             'project': project, 'Project': _cap(project), 'short': short}
    out = {}
    for t in GRADES['ir_mechanism']:
        texts = []
        for i, fr in enumerate(IR_FRAMES):
            pool = IR_OPTOUT_OUTSIDE if t == 'outside_option' else IR_OPTOUT
            opt = pool[i % len(pool)]
            alt_bare = alt.replace('keeps ', 'keep ', 1)
            opt = _fill(opt, {**words, 'alt': alt, 'alt_bare': alt_bare})
            texts.append(_fill(fr.replace('{optout}', opt), words))
        out[t] = texts
    return out


def _bb_texts(sc):
    group, member, where, project, short, _ = sc
    words = {'group': group, 'member': member, 'a_member': _article(member), 'where': where,
             'project': project, 'Project': _cap(project), 'short': short}
    return {t: [_fill(fr, words) for fr in BB_FRAMES] for t in GRADES['budget_balance']}


def _ar_texts(sc):
    bidders, item, seller = sc
    words = {'bidders': bidders, 'item': item, 'seller': seller, 'Seller': _cap(seller)}
    return {t: [_fill(fr, words) for fr in AR_FRAMES[t]] for t in GRADES['auctions_risk']}


def _check(sc) -> None:
    for s in sc:
        assert not _NUMBER_WORDS.search(s) and '$' not in s and not re.search(r'\d', s), s


def build(element: str) -> Dataset:
    scen, texts_of = {
        'ir_mechanism': (PROJECTS, _ir_texts),
        'budget_balance': (PROJECTS, _bb_texts),
        'auctions_risk': (AUCTIONS, _ar_texts),
    }[element]
    rows = {'questions': [], 'example_questions': []}
    for domain, scs in scen.items():
        for k, sc in enumerate(scs):
            _check(sc)
            pool = 'example_questions' if k == 0 else 'questions'
            for t, texts in texts_of(sc).items():
                if pool == 'example_questions':
                    texts = texts[k % len(texts):][:1]  # one worked example per type and domain
                for text in texts:
                    rows[pool].append({'question_text': text, 'explanation': EXPLANATIONS[(element, t)],
                                       'domain': domain, 'type': t,
                                       'difficulty_level': GRADES[element][t], 'tags': TAGS})
    ds = Dataset(element)
    for pool, opt_attr in (('questions', 'options'), ('example_questions', 'example_options')):
        q = pd.DataFrame(rows[pool])
        q.insert(0, 'sub_id', 0)
        q.insert(0, 'question_id', range(len(q)))
        if pool == 'questions':
            q = q.drop(columns='explanation')
        q = q.astype({'question_id': int, 'sub_id': int, 'difficulty_level': int})
        # Mode A fillers supply the options; one placeholder option row keeps the pool shape.
        o = pd.DataFrame({'question_id': q.question_id, 'sub_id': 0, 'option_id': 0, 'option_text': '',
                          'correct_answer': False})
        setattr(ds, pool, q.reset_index(drop=True))
        setattr(ds, opt_attr, o.reset_index(drop=True))
    return ds


ELEMENTS = ('ir_mechanism', 'budget_balance', 'auctions_risk')


def main(elements=None) -> None:
    import sys
    for e in elements or sys.argv[1:] or ELEMENTS:
        ds = build(e)
        path = os.path.join(templates_dir(), e)
        ds.save(path)
        print(f'{e}: {len(ds.questions)} test templates, {len(ds.example_questions)} example templates -> {path}')


if __name__ == '__main__':
    main()
