# Built-in packages
# External packages
import pandas as pd
# Local packages
from .fill_registry import FillRegister
function_registry = FillRegister()

social_choice_functions = pd.DataFrame.from_records([
    {
        "method": "Majority Rule", 
        "description": "The alternative that receives the most votes from the individuals is chosen as the social choice.", 
        "functional_form": "f(x) = argmax(x)", 
        "pareto": True, 
        "monotonicity": True, 
        "transitivity": True
    },
    {
        "method": "Plurality Rule", 
        "description": "The alternative that receives the highest number of votes from the individuals is chosen as the social choice.", 
        "functional_form": "f(x) = argmax(count(x))", 
        "pareto": True, 
        "monotonicity": True, 
        "transitivity": False
    }, 
    {
        "method": "Borda Count", 
        "description": "Each individual ranks the alternatives, and the alternative with the highest cumulative rank is chosen as the social choice.", 
        "functional_form": "f(x) = argmax(sum(rank(x)))", 
        "pareto": True, 
        "monotonicity": True, 
        "transitivity": True
    },
    {
        "method": "Approval Voting", 
        "description": "Each individual can approve or disapprove of multiple alternatives, and the alternative with the highest approval count is chosen as the social choice.", 
        "functional_form": "f(x) = argmax(count(approve(x)))", 
        "pareto": True, 
        "monotonicity": True, 
        "transitivity": True
    },
    {
        "method": "Range Voting", 
        "description": "Each individual assigns a score to each alternative, and the alternative with the highest average score is chosen as the social choice.", 
        "functional_form": "f(x) = argmax(avg(score(x)))", 
        "pareto": True, 
        "monotonicity": True, 
        "transitivity": True
    },
    {
        "method": "Instant Runoff Voting", 
        "description": "Each individual ranks the alternatives, and the alternative with the lowest number of last-place rankings is chosen as the social choice. If there is a tie, the alternative with the highest number of second-place rankings is chosen, and so on.", 
        "functional_form": "f(x) = argmin(rank(x))", 
        "pareto": True, 
        "monotonicity": False, 
        "transitivity": True
    },
    {
        "method": "Dictatorship", 
        "description": "The social choice is always the preferred alternative of a single individual, regardless of the preferences of others.", 
        "functional_form": "f(x) = x_dictator", 
        "pareto": False, 
        "monotonicity": False, 
        "transitivity": False
    },
    {
        "method": "Anti-Plurality Rule", 
        "description": "Each individual votes for all but one alternative, and the alternative with the fewest votes is chosen as the social choice.", 
        "functional_form": "f(x) = argmin(count(x))", 
        "pareto": True, 
        "monotonicity": True, 
        "transitivity": False
    },
    {
        "method": "Copeland's Method", 
        "description": "Each alternative gets one point for each other alternative it beats in a head-to-head competition, and the alternative with the most points is chosen as the social choice.", 
        "functional_form": "f(x) = argmax(sum(points(x)))", 
        "pareto": True, 
        "monotonicity": True, 
        "transitivity": True
    },
    {
        "method": "Simpson-Kramer Method", 
        "description": "The alternative that would win by the largest margin in a head-to-head competition against the alternative it is weakest against is chosen as the social choice.",
        "functional_form": "f(x) = argmax(min(margin(x, y) for y in alternatives))", 
        "pareto": True, 
        "monotonicity": True, 
        "transitivity": True
    },
    {
        "method": "Ranked Pairs", 
        "description": "The alternative that wins the most head-to-head competitions is chosen as the social choice. If there is a tie, the alternative that wins the second most head-to-head competitions is chosen, and so on.", 
        "functional_form": "f(x) = argmax(sum(wins(x, y) for y in alternatives))", 
        "pareto": True, 
        "monotonicity": True, 
        "transitivity": True
    },
    {
        "method": "Dodgson's Method", 
        "description": "The alternative that would require the fewest changes in individual rankings to become a Condorcet winner is chosen as the social choice.", 
        "functional_form": "f(x) = argmin(sum(changes(x) to become Condorcet winner))", 
        "pareto": True, 
        "monotonicity": False, 
        "transitivity": True
    },
    {
        "method": "Smith/Minimax Method", 
        "description": "The alternative that would lose by the smallest margin in a head-to-head competition against the alternative it is weakest against is chosen as the social choice.", 
        "functional_form": "f(x) = argmin(max(margin(x, y) for y in alternatives))", 
        "pareto": True, 
        "monotonicity": True, 
        "transitivity": True
    },
    {
        "method": "Baldwin's Method", 
        "description": "Alternatives are eliminated one by one based on their Borda count until only one alternative remains, which is chosen as the social choice.", 
        "functional_form": "f(x) = last remaining alternative after iterative elimination based on Borda count", 
        "pareto": True, 
        "monotonicity": False, 
        "transitivity": True
    },
    {
        "method": "Coombs' Method", 
        "description": "Alternatives are eliminated one by one based on their plurality count until only one alternative remains, which is chosen as the social choice.", 
        "functional_form": "f(x) = last remaining alternative after iterative elimination based on plurality count", 
        "pareto": True, 
        "monotonicity": False, 
        "transitivity": True
    },
    {
        "method": "Bucklin's Method", 
        "description": "The alternative that receives a majority of first-place votes is chosen as the social choice. If no alternative receives a majority, the alternative that receives a majority when second-place votes are included is chosen, and so on.", 
        "functional_form": "f(x) = first alternative to receive a majority of votes when votes are counted in order of preference", 
        "pareto": True, 
        "monotonicity": True, 
        "transitivity": True
    },
    {
        "method": "Random Choice",
        "description": "A random alternative is chosen as the social choice.",
        "functional_form": "f(x) = random.choice(x)",
        "pareto": False,
        "monotonicity": True,
        "transitivity": True
    },
    {
        "method": "Random Dictatorship",
        "description": "A random individual's preferred alternative is chosen as the social choice, regardless of the preferences of others.",
        "functional_form": "f(x) = x_dictator_random",
        "pareto": False,
        "monotonicity": False,
        "transitivity": False
    },
    {
        "method": "Plurality with Runoff",
        "description": "If no alternative receives a majority of votes, a runoff election is held between the two alternatives with the highest number of votes.",
        "functional_form": "f(x) = argmax(count(x)) if max(count(x)) > len(x) / 2 else argmax(count(runoff(x)))",
        "pareto": False,
        "monotonicity": True,
        "transitivity": True
    }
])





#########################################################################################
#########################################################################################
##                                                                                     ##
##                                                                                     ##
##                        Generate Social Choice Axiom Values                          ##
##                                                                                     ##
##                                                                                     ##
#########################################################################################
#########################################################################################

@function_registry.register('sc_pareto')
def gen_pareto_values(metadata):
    # sample_indices = rng.integers(low = 0, high = len(social_choice_functions), size = 3)
    options = social_choice_functions.query('pareto == False').sample(3)['description'].tolist()
    correct_answer = social_choice_functions.query('pareto == True').sample(1)['description'].tolist()[0]
    options += correct_answer

    return {
        'options_list': [options],
        'correct_answer': [[correct_answer == option for option in options]]
    }

@function_registry.register('sc_monotonicity')
def gen_monotonicity_values(metadata):
    options = social_choice_functions.query('monotonicity == False').sample(3)['description'].tolist()
    correct_answer = social_choice_functions.query('monotonicity == True').sample(1)['description'].tolist()[0]
    options += correct_answer

    return {
        'options_list': [options],
        'correct_answer': [[correct_answer == option for option in options]]
    }

@function_registry.register('sc_transitivity')
def gen_transitivity_values(metadata):
    options = social_choice_functions.query('transitivity == False').sample(3)['description'].tolist()
    correct_answer = social_choice_functions.query('transitivity == True').sample(1)['description'].tolist()[0]
    options += correct_answer

    return {
        'options_list': [options],
        'correct_answer': [[correct_answer == option for option in options]]
    }