"""Relative difficulty: the difficulty users see.

Templates carry raw difficulty levels (after public_difficulty, below): STEER's are grade-like (0 to 13),
other elements use small integers of their own. Raw levels mean different things in different elements, so every
public surface (the dataset `difficulty` column, /api/sample and /api/elements, the site) shows a relative level
instead: an element's distinct raw levels, sorted ascending, numbered 1..k, so 1 is the element's easiest level and
k its hardest. Relative levels are comparable only within one element, not across elements.

The sampler, its filters, the template data and the manifest keep raw levels; these helpers translate at the edge.
An element's raw levels are the manifest's `difficulty_levels` (pass `manifest` to use a given one).
"""
from __future__ import annotations

import os


# Published difficulty is 0/1/2. Some elements use their own internal levels, which the fillers branch on; they are
# mapped only in the sampler output (and in SamplerConfig filtering / the manifest).
DIFFICULTY_MAPS: dict[str, dict[int, int]] = {
    # 0: 2 goods, prices given; 1: 2 goods, functional form; 3: 3 goods, linear utility, prices given;
    # 4: 3 goods, prices given; 2: 3 goods, functional form
    'deriving_hicksian_demand': {0: 0, 1: 1, 3: 1, 4: 1, 2: 2},
    # 5: single prospect; 6: scenarios with a population / several options
    'expected_value_counts_aug2024': {5: 0, 6: 1},
}


def public_difficulty(element: str, level) -> int:
    level = int(level)
    return DIFFICULTY_MAPS.get(element, {}).get(level, level)

_CACHE: dict = {}


def _manifest() -> dict:
    """load_manifest(), reloaded only when manifest.json changes (load_manifest hashes the code on every call)."""
    from . import manifest as m

    def stamp():
        try:
            path = m.manifest_path()
            return path, os.stat(path).st_mtime_ns
        except OSError:
            return None

    key = stamp()
    if key is None or _CACHE.get("key") != key:
        loaded = m.load_manifest()
        _CACHE.update(key=stamp(), manifest=loaded)
        return loaded
    return _CACHE["manifest"]


def relative_scale(raw_levels) -> dict[int, int]:
    """{raw: relative} for one element's raw levels: sorted ascending (numerically), numbered from 1."""
    return {raw: i for i, raw in enumerate(sorted({int(x) for x in raw_levels}), start=1)}


def raw_levels(element: str, manifest: dict | None = None) -> list[int]:
    """The element's raw difficulty levels, ascending."""
    entry = (manifest if manifest is not None else _manifest()).get("elements", {}).get(element)
    if entry is None:
        raise KeyError(f"unknown element {element!r}")
    return sorted({int(x) for x in entry.get("difficulty_levels", [])})


def level_count(element: str, manifest: dict | None = None) -> int:
    """k: how many difficulty levels the element has (its relative levels are 1..k)."""
    return len(raw_levels(element, manifest))


def relative_levels(element: str, manifest: dict | None = None) -> list[int]:
    return list(range(1, level_count(element, manifest) + 1))


def to_relative(element: str, raw, manifest: dict | None = None) -> int:
    """Relative level (1..k) of a raw level of the element."""
    scale = relative_scale(raw_levels(element, manifest))
    try:
        return scale[int(raw)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{element}: {raw!r} is not one of its raw difficulty levels {sorted(scale)}") from None


def parse_relative(element: str, level, manifest: dict | None = None) -> int:
    """Check a relative level given by a user (int or digit string); returns it as an int or raises ValueError
    with a message fit to show (it names the element's range)."""
    k = level_count(element, manifest)
    text = str(level).strip()
    if isinstance(level, bool) or not text.lstrip("+").isdigit() or not 1 <= int(text) <= k:
        span = "1 (it has one level)" if k == 1 else f"1 to {k} (1 = its easiest)"
        raise ValueError(f"difficulty_level for {element} must be {span}, got {level!r}; "
                         "levels are relative to each element")
    return int(text)


def to_raw(element: str, level, manifest: dict | None = None) -> int:
    """Raw level of a relative level (1..k) of the element; ValueError if out of range."""
    return raw_levels(element, manifest)[parse_relative(element, level, manifest) - 1]


def public_entry(element: str, entry: dict | None) -> dict | None:
    """A manifest entry as shown publicly: `difficulty_levels` as the relative levels "1".."k"."""
    if entry is None or "difficulty_levels" not in entry:
        return entry
    k = len({int(x) for x in entry["difficulty_levels"]})
    return {**entry, "difficulty_levels": [str(i) for i in range(1, k + 1)]}
