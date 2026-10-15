# src/steer_generation/constants.py
# Shared constants for the STEER generation pipeline.

DOMAINS = {
    'consumer_goods': 'consumer goods',
    'medical': 'medical activities',
    'finance': 'financial settings',
    'education': 'educational settings',
    'technology': 'technology settings',
    'entertainment': 'the entertainment industry',
    'environmental_policy': 'environmental policy',
    'politics': 'political settings',
    'sports': 'sports scenarios',
    'gambling': 'gambling activities',
    'base': '',
}

TAGS = {
    'first_person': 'The question should be written in the first person',
    'second_person': 'The question should be written in the second person',
    'woman': "The question should be written in the third person and the subject's name should be feminine",
    'man': "The question should be written in the third person and the subject's name should be masculine",
    'anon': "The question should be written in the third person and do not name the subject",
}
