"""The port reproduces the source generator: seeded samples equal those of the source commit.

tests/reference/source_digests.json holds, for every element the source could sample, a digest of 25 seeded draws per
pool (test templates and few-shot examples) from STEER-generation at the commit it names, with the same template
data. Here the same draws are made with autosteer's sampler in this environment.
A difference means the port, the template data or the environment changed what is generated.

Elements changed on purpose after the port are listed in CHANGED_SINCE_PORT (their generator no longer matches the
source), and elements that the source could not sample in ADDED_SINCE_PORT. 
"""
import json
import os

import pytest

from _digest_dump import digests
from autosteer.sampler import ElementSampler, SamplerConfig

REF = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference', 'source_digests.json')))
ELEMENTS = sorted({k.rsplit('/', 1)[0] for k in REF['digests']})

# element -> what changed
CHANGED_SINCE_PORT = {
    'aggregation_of_labor_supply': 'review: P and group conventions stated, strategy stories dropped, mistake distractors',
    'deriving_labor_supply': 'review: hourly wage and daily income stated, inconsistent stories dropped, mistake distractors',
    'labor_supply_elasticity': 'review: rebuilt as code templates (percent changes, linear approximation stated)',
    'price_of_risk': 'review: rebuilt as code templates (utility, borrowing rule and price of risk stated)',
    'arbitrage': 'prices drawn at 2 decimals; no arbitrage in either direction in the "No" case; distinct profits; review: currency and two-way transport cost stated',
    'bayes_nash': 'review: games with a weak or mixed second equilibrium redrawn (bayes_unique)',
    'backward_induction': 'review: 11 templates dropped (mislabelled treatment stories, equity percentages); no claim that terms improve',
    'borda_count': 'review: the truncated-ballot scoring stated in every question',
    'budget_balance': 'review: asks how the money collected compares with the cost; Clarke pivot defined with the cost share left out',
    'ir_mechanism': "review: Clarke pivot defined with the person's value and cost share both left out",
    'dominated_strategies': 'review: no second action weakly dominated or dominated by a mixture',
    'iterated_removal': 'review: no second action weakly dominated or dominated by a mixture; part 2 says the first action is removed',
    'consumer_surplus': 'draws with a surplus below 0.05 are redrawn; review: inverse demand and divisibility stated',
    'aggregation_of_consumer_demand': 'review: mistake-based options at a random rank; per-member convention stated',
    'aggregation_of_offer_curve': 'review: mistake-based options (2 x 2 grid for curves, random rank for totals); per-member convention',
    'deriving_demand': 'review: options c/p from the 2 x 2 share grid; x and y named, no money-spent goods, p_x/p_y defined',
    'marginal_utility': 'review: distractors are differentiation mistakes on the stem exponents; answer format stated',
    'mrs_utility': 'review: options are the 2 x 2 grid of exponent / variable ratios; ordered-pair format stated',
    'law_of_demand': 'review: ceteris paribus stated; demand-shift / non-demand / direction-giving templates dropped; final question asks for the new value',
    'diminishing_marginal_utility': 'review: final question asks for the marginal utility (derivative) of the varied good; x/y named',
    'certainty_effect': 'review: stated_prior questions say the preferences have not changed',
    'reflection_effect': 'review: stated_prior assumption sentence; loss options say "having to pay"',
    'gamblers_fallacy': 'review: 35 templates dropped (Yes defensible, dependent card draws, no comparison)',
    'producer_surplus': 'draws with a surplus below 0.05 are redrawn',
    'efficient_surplus': 'draws whose intercepts differ by less than 0.5 are redrawn; review: inverse curves stated; given price marked as rounded',
    'pure_nash': 'rebuilt: code-written matrix, pricing and salary templates from generated payoff tables',
    'dominant_strategies': 'rebuilt: code-written pricing and negotiation templates, objective stated',
    'completeness': 'rebuilt (October 2026): code templates, every pair of 4-5 sure alternatives, consistency-graded (pairwise_complete)',
    'completeness_risk': 'rebuilt (October 2026): code templates, every pair of 4-5 lotteries, consistency-graded (pairwise_complete)',
    'intertemporal_consumption_smoothing': 'review: timing and CRRA objective stated; distractors are Euler-equation mistakes',
    'state_based_consumption': 'review: ex-ante dollar allocation stated; state B dearer in the linear type',
    'welfare_theorem_1': 'review: competitive-equilibrium premise stated; externality and goods-less stories dropped',
    'welfare_theorem_2': 'review: Pareto-efficiency and convexity premise stated; one final question per type',
    'tfp_shocks': 'capital-stock types dropped; competitive long-run assumption stated in every question',
    'dsic_mechanism': 'redesign: code templates (dsic_tools.py), 18 mechanisms in four settings described in words, plus a counterexample type',
    'equiv_contrapositive': 'code templates from the normalised context bank (contrapositive_bank.json); options from the same clause strings',
    'feasibility': 'rebuilt (October 2026): code templates; folk type (no discount factor) and discount type (exact grim-trigger SPE plans)',
    'trigger': '7 templates stating a payoff as a loss dropped (pd_role_fixes.json); the payoff table printed after the question; review: 10 more dropped (roles swapped, unconditional cooperation named), rounding stated',
    'deadweight_loss': 'quality audit: deadweight loss at least 0.5; no distractor printed as 0.0 or a few cents; review: marginal cost, single price and competitive benchmark stated',
    'deriving_hicksian_demand': 'quality audit: Cobb-Douglas exponents >= 0.1, degree >= 0.5 (no 3.8e+19 keys); no " , where"; review: functional options a 2 x 2 grid of constant / exponent mistakes, rounding stated, earnings and money-spent templates dropped',
    'find_eq_price': 'quality audit: no distractor printed as 0.0 or a few cents (filler_utils.gen_answer_and_options); review: price options between the intercepts of the curves',
    'profit_maximization': 'quality audit: no distractor printed as 0.0 or a few cents (filler_utils.gen_answer_and_options)',
    'labor_supply_distortions': 'review: rebuilt as code templates (tax wedge model stated, percent answers)',
    'average_cost': 'quality audit: inputs and coefficients at least 0.05, no Q^0.0, wage alone where a template says "per hour"',
    'marginal_cost': 'quality audit: inputs and coefficients at least 0.05, no Q^0.0, wage alone where a template says "per hour"',
    'output_elasticity': 'quality audit: capital at least 0.5 (K = 0.0 made two options correct), exponents at least 0.1',
    'returns_to_scale': 'quality audit: production-function coefficients at least 0.1 / 0.5 (sample_prod_function)',
    'marginal_production': 'quality audit: Cobb-Douglas exponents at least 0.1 (sample_prod_function)',
    'comp_stat_cobb': 'quality audit: Cobb-Douglas exponents at least 0.1 (sample_prod_function); no longer used',
    'supply_elasticity': 'quality audit: supply slope and intercept at least 0.5 (no "Q = 7.2P + 0.0"); no 0.0 distractor',
    'dynamic_profit_max': 'quality audit: capital and prices at least 0.5, discount factor 0.5-0.99, probabilities at least 0.05, no key or option 0.0',
    'price_elasticity': 'quality audit: convention stated; whole quantities; mistake-based distractors away from the midpoint value; review: distributed / produced quantities dropped',
    'diminishing_marginal_product': 'quality audit: templates write K for capital (they said C); exponents at least 0.1',
    'capital_market_distortions': 'review: rebuilt as code templates (after-tax returns reinvested, K0 (1 + g(1 - t))^n)',
    'plurality_voting': 'quality audit: ballot lines and the question each on their own line',
    'interpret_games': 'quality audit: options formatted as in the table ("7", not "7.0"); "row player. What is"',
    'aggregation_of_labor_demand': 'Review: demands nonnegative at the stated prices, price range stated; mistake distractors',
    'aggregation_of_capital_demand': 'Review: demands nonnegative at the stated prices, price range stated; mistake distractors',
    'cost_minimization_problem': 'Review: distractors from mistaken bundles (key was nearly always the largest)',
    # review (October 2026), logic / arithmetic / probability elements
    'add_sub': 'review: story templates whose negative amounts conflict with the question asked dropped',
    'mult_div': 'review: rounding stated; options within a factor 3 of the answer in size',
    'bayes_rule': 'review: rounding stated, every option with 2 decimals, at most one copied-input distractor',
    'complement_rule': 'review: wordings whose asked event is not the complement reworded',
    'compute_expectations': 'review: fair coin / die stated; options within the range of the outcome; 2 decimals kept',
    'compute_expected_utility': 'review: options between the smallest and largest utility, 2 decimals',
    'enforceability': 'review: money in options keeps 2 decimals (format_value no longer prints "$69551.7")',
    'expected_value_counts_aug2024': 'review: options keep 2 decimals (format_value)',
    # October 2026: repeated templates deleted from the data and the balance they gave kept by the sampler
    # (SAMPLING_BALANCE); a template text with several fill settings is drawn as one template
    'condorcet_criterion': 'a template text with several fill settings (numbers of candidates) is drawn as one template',
    'independence_risk': 'review: each template also emitted with the two lotteries in the other order; repeated templates deleted, domains balanced by the sampler',
    'subgame_nash': 'options written per round in words ("Round 1: you play Action S, your opponent plays Action K; ...") instead of tuples; repeated templates deleted, levels balanced by the sampler',
}
# element -> why it is sampleable here but not in the source
ADDED_SINCE_PORT = {
    'input_price_leontief': 'hidden code-template source of the leontief type of input_price_response (rebuilt October 2026)',
    'input_price_cobb': 'hidden code-template source of the cobb_douglas type of input_price_response (rebuilt in the quality audit)',
    'loss_aversion': 'new generator: positive-EV mixed gambles against nothing or a small sure amount, keyed by expected value (code templates)',
    'optimize_continuous': 'new generator: maximiser or minimiser of a quadratic or cubic on an interval, box-volume and fencing problems (code templates)',
    'auctions_affiliated': 'new generator: revenue rankings of described formats with the environment read from the story, and the appraisal-publishing policy (code templates)',
    'bic_mechanism': 'new generator: is truthful reporting a Bayes-Nash equilibrium of a described mechanism, and the first-price equilibrium bid (code templates)',
    'fair_division': 'new generator: divide and choose, last diminisher, adjusted winner and envy-free / proportional checks; keys by simulation (code templates)',
    'sc_nondictatorship': 'new generator: dictatorships in disguise, outcome tables and subtle procedures; keys by simulation (code templates)',
    'sc_transitivity_rules': 'new generator: four unnamed rules turning a profile into pairwise verdicts; keys by cycle detection (code templates)',
    'sc_monotonicity_rules': 'new generator: four unnamed rules that rank the options, on a profile and the profile with one option raised; keys by recomputing every ranking (code templates)',
    'input_price_response': 'composite element: the templates and fillers of comp_stat_cobb and comp_stat_leonteif as two types',
    'constrained_optimization': "new filler (fixed ConstrainedOptimizationProblem) with code templates: story LPs and bare QPs; the source had templates but no filler",
    'shephards_lemma': 'new generator replacing the old untemplated shephards_lemma fillers (code templates)',
    'income_elasticity': 'new generator replacing the old untemplated income_elasticity filler (code templates)',
    'engel_curves': 'new generator replacing the old untemplated engel_curves filler (code templates)',
    'linear_system_eq': 'rebuilt filler with code templates (the source had a filler but no templates)',
    'exponents': 'rebuilt filler with code templates (the source had a filler but no templates)',
    'finding_tfp': 'new generator replacing the old finding_tfp filler, which had no templates (code templates)',
    'substitutes_complements': 'new generator: substitutes or complements from a linear demand or a quasi-linear utility, and the cross-price elasticity (code templates)',
    'mrs_tangency': 'new generator: MRS at the optimum, the tangency bundle, and which way to move along the budget line (code templates)',
    'present_value': 'new generator: present values of an amount, a stream or a perpetuity (code templates)',
    'copeland_method': 'new generator: Copeland winner from a profile or a one-on-one table (code templates)',
    'hicksian_duality': 'new generator: Marshallian demand and the Hicksian demand of its utility level, two parts (code templates)',
    'tax_incidence': 'new generator: the same per-unit tax collected from sellers and from buyers, two parts (code templates)',
    'factor_demand': 'new generator: profit maximization and cost minimization of the optimal output, two parts (code templates)',
    'independence': 'new generator: asymmetric decoys with the recovered decoy scenarios (filler_tools/decoy_templates.json)',
    'ambiguity_aversion': 'new generator from the recovered Ellsberg templates (filler_tools/ellsberg_templates.json)',
    'pareto_sc': 'new generator: hand-written Yes/No and which-to-choose templates (code templates)',
    'sunk_cost': 'new generator: 20 hand-written scenarios with named slots (code templates)',
    'maximize_expected_utility': 'new generator: plain type only, utility linear in money (code templates)',
    'micro_utility': 'new filler (STEER 1.2.a); the source had templates but no filler',
    'categorical_syllogism': 'new generator: premise pairs over the syllogistic figures, keys by model checking (code templates)',
    'conditional_syllogism': 'new generator: hand-written rule bank, the four argument forms, keys by truth table (code templates)',
    'transitivity': 'new generator: chained pairwise preferences, keys by reachability (code templates)',
    'top_trading': 'new generator: random strict rankings, keys by Top Trading Cycles (code templates)',
    'prob_basic': 'new generator: simple events with exact fractions (code templates)',
    'first_order_false_belief': 'new generator: unexpected-transfer stories, keys by belief tracking (code templates)',
    'second_order_false_belief': 'new generator: two moves, the observer may not know what the owner learned; keys by nested belief tracking (code templates)',
    'endowment_effect': 'new generator: mirrored trade offers graded by mirror_trade (code templates)',
    'consumption_changes': 'new generator: direction and size of the change in a CES expenditure share after a price rise (code templates)',
    'mrs_labor_supply': 'new generator replacing the old untemplated mrs_labor_supply filler (code templates)',
    'optimal_portfolio_choice': 'new generator replacing the old optimal_portfolio_choice filler, which had no templates (code templates)',
    'elasticity_of_substitution': 'new generator replacing the old elasticity_of_substitution filler, which had no templates (code templates)',
    'entry_exit_decisions': 'new generator replacing the old entry_exit_decisions filler, which had no templates (code templates)',
    'fixed_factor_aggregation': 'new generator replacing the old fixed_factor_aggregation filler, which had no templates (code templates)',
    'factor_shares_equilibrium': 'new generator replacing the old factor_shares_equilibrium filler, which had no templates (code templates)',
    'elastic_inelastic_supply': 'new generator: price changes after cost and demand shocks with fixed or perfectly elastic supply (code templates)',
    'non_distortionary_taxes': 'new generator: the lump-sum or fixed-factor tax among distorting taxes, and the zero-deadweight-loss per-unit tax (code templates)',
}


def test_reference_covers_every_sampleable_element():
    here = set(ElementSampler()._sampleable) - set(ADDED_SINCE_PORT)
    assert here == set(ELEMENTS), {'only here': sorted(here - set(ELEMENTS)), 'only in source': sorted(set(ELEMENTS) - here)}


@pytest.mark.parametrize('element', ELEMENTS)
def test_same_samples_as_source(element):
    if element in CHANGED_SINCE_PORT:
        pytest.skip(f'changed after the port: {CHANGED_SINCE_PORT[element]}')
    got = digests(ElementSampler, SamplerConfig, REF['n'], [element])
    want = {k: v for k, v in REF['digests'].items() if k.rsplit('/', 1)[0] == element}
    assert got == want
