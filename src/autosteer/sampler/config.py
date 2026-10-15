from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class SamplerConfig:
    """
    Partial configuration for ElementSampler. Any field left as None will be
    sampled uniformly from the values available for the chosen element.

    Fields
    ------
    element_name : str, optional
        Name of the element (e.g. "deriving_demand"). Sampled uniformly across
        all elements that have both a template and a filler function if omitted.
    domain : str, optional
        Domain context (e.g. "consumer_goods", "finance"). Must be a value
        present in the element's manifest entry.
    type : str, optional
        Sub-type within the element (e.g. "good_1", "good_2").
    difficulty_level : str or int, optional
        Difficulty level. Stored as strings in the manifest; int values are
        automatically coerced to str for comparison.
    tags : str, optional
        Perspective tag (e.g. "first_person", "woman", "anon").
    pool : str, optional
        Which template pool to draw from: "questions" (default, the test
        templates) or "examples" (the few-shot example templates).
    """
    element_name: Optional[str] = None
    domain: Optional[str] = None
    type: Optional[str] = None
    difficulty_level: Optional[Union[str, int]] = None
    tags: Optional[str] = None
    pool: Optional[str] = None
