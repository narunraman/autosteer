import functools
import hashlib
import logging
import random
import re
import string
import threading
from typing import Optional

import numpy as np

from autosteer.filler_tools import code_templates, function_registry
from autosteer.filler_tools.utilities import filler_utils
from autosteer.filler_tools.fill_registry import validate_filler_output
from autosteer.formatting import format_fill_values, format_value, extract_fstring_keys
from autosteer.steer_dataset import Dataset

from .config import SamplerConfig
from .manifest import load_manifest, get_sampleable_elements, load_raw_templates
from .template_fixes import apply_template_fixes, public_difficulty



def seed_everything(seed: int) -> None:
    """
    Seed every random source the filler functions use: the stdlib `random`
    module, numpy's global RNG, and the shared `filler_utils.rng` generator
    (reseeded in place, because filler modules import it by name).
    """
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    filler_utils.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state


SEED_MAX = 2 ** 63 - 1
# Options that refer to other options by position are never shuffled.
POSITIONAL_RE = re.compile(
    r"\b(all|none|both|neither) of (the )?(above|these|them)\b|\boption [A-H]\b|\b[A-D] and [A-D]\b",
    re.IGNORECASE,
)


def derive_seed(*parts) -> int:
    """A seed in [0, 2**63) from any parts, stable across processes and Python versions (sha256, not hash())."""
    digest = hashlib.sha256('\x1f'.join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], 'big') & SEED_MAX


def parse_template_id(template_id, element_name: str, pool: str) -> int:
    """Template row id from '<element>/<pool>/<n>' or a bare n."""
    if isinstance(template_id, (int, np.integer)):
        return int(template_id)
    parts = str(template_id).split('/')
    if len(parts) == 3:
        if parts[0] != element_name or parts[1] != pool:
            raise ValueError(f"template_id {template_id!r} is not a {pool!r} template of {element_name!r}")
        return int(parts[2])
    if len(parts) == 1:
        return int(parts[0])
    raise ValueError(f"malformed template_id {template_id!r}; expected '<element>/<pool>/<n>'")


class _KeepMissing(dict):
    """format_map mapping that leaves unknown {placeholders} untouched."""
    def __missing__(self, key):
        return '{' + key + '}'


def _fill_text(text: str, fill_values: dict) -> str:
    """Fill {placeholders} in an option text from a template; leave unknown ones."""
    if not isinstance(text, str) or '{' not in text:
        return text
    try:
        return string.Formatter().vformat(text, (), _KeepMissing(fill_values))
    except (ValueError, IndexError, KeyError):
        return text


def _normalize_options(options_list: list, correct_answer: list) -> tuple[list[list], list[list]]:
    """
    Filler functions return options_list/correct_answer with inconsistent nesting.
    Normalize both to shape [sub_id][option_id] regardless of source.
    """
    # Detect extra nesting: [[[ ... ]]] instead of [[ ... ]]
    if (
        isinstance(options_list[0], list)
        and len(options_list[0]) == 1
        and isinstance(options_list[0][0], list)
    ):
        options_list = [inner[0] for inner in options_list]
        correct_answer = [inner[0] for inner in correct_answer]

    return options_list, correct_answer


def _fill_question(template_text: str, fill_values: dict) -> str:
    """Substitute {placeholder} keys in template_text with fill_values."""
    fill_values = {k: v for k, v in fill_values.items()
                   if k not in ('options_list', 'correct_answer')}
    return template_text.format(**fill_values)


def _has_duplicate_options(options_list: list[list]) -> bool:
    for opts in options_list:
        texts = [str(format_value(o)) if not isinstance(o, str) else format_value(o) for o in opts]
        if len(set(texts)) != len(texts):
            return True
    return False


def _build_result(
    element_name: str,
    metadata: dict,
    question_texts: list[str],
    options_list: list[list],
    correct_answer: list[list],
    rng: Optional[random.Random] = None,
) -> dict:
    """
    Package filled questions and options into the output dict.

    For single-sub-question elements the return is flat:
      {question, options, answer, metadata}

    For multi-sub-question elements:
      {sub_questions: [{question, options, answer}], metadata}
    """
    sub_questions = []
    for sub_id, (q_text, opts, ans_mask) in enumerate(
        zip(question_texts, options_list, correct_answer)
    ):
        opts_str = [str(format_value(o)) for o in opts]
        ans_mask_bool = [bool(a) for a in ans_mask]
        if rng is not None:
            # Shuffle option order so the correct position is uniform.
            order = list(range(len(opts_str)))
            rng.shuffle(order)
            opts_str = [opts_str[i] for i in order]
            ans_mask_bool = [ans_mask_bool[i] for i in order]
        correct_idx = ans_mask_bool.index(True) if True in ans_mask_bool else None
        answer_text = opts_str[correct_idx] if correct_idx is not None else None

        sub_questions.append({
            'question': q_text,
            'options': [
                {'text': text, 'correct': correct}
                for text, correct in zip(opts_str, ans_mask_bool)
            ],
            'answer': answer_text,
        })

    result_metadata = {
        'element': element_name,
        'domain': metadata['domain'],
        'type': metadata['type'],
        'difficulty_level': public_difficulty(element_name, metadata['difficulty_level']),
        'tags': metadata['tags'],
    }

    if len(sub_questions) == 1:
        return {**sub_questions[0], 'metadata': result_metadata}

    return {'sub_questions': sub_questions, 'metadata': result_metadata}


# Seeding and the fillers use process-wide RNGs, so everything that seeds or calls a filler holds this lock: the API,
# the element pages' examples, developer mode and the template loader would otherwise change each other's draws.
RNG_LOCK = threading.RLock()
# Loaded templates, shared by every sampler in the process (the API's, the element pages', developer mode's):
# (element, manifest build) -> Dataset. Loading the large elements takes seconds (welfare_theorem_1: ~10 s).
_SHARED_TEMPLATES: dict = {}
# Templates dropped at load as unfillable, per element and pool: {reason: [question_id, ...]} (see _drop_unfillable).
DROPPED: dict = {}
log = logging.getLogger(__name__)


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with RNG_LOCK:
            return fn(*args, **kwargs)
    return wrapper


# Elements whose questions are balanced over one field: when the request leaves the field open, its value is drawn
# first (evenly, or with these weights), then a template that has it. The released data balanced these fields by
# repeating templates (independence_risk held each of its 18 vacation templates 19 times, to match its 337 meal
# templates); the repeats were deleted from the data (October 2026) and the balance is kept here.
SAMPLING_BALANCE: dict[str, tuple[str, dict | None]] = {
    'independence_risk': ('domain', None),
    'compute_expectations': ('domain', None),
    'subgame_nash': ('difficulty_level', None),
    'complement_rule': ('difficulty_level', None),
    # mult_div: as released, half positive and a quarter each negative and both (only the explicit-arithmetic
    # templates take negative numbers; the stories are about prices and counts)
    'mult_div': ('type', {'positive': 2, 'negative': 1, 'both': 1}),
}


class ElementSampler:
    """
    Samples economics MCQ elements on the fly by combining a question template
    with a filler function. No LLM is involved — all generation is deterministic
    and numeric.

    Usage
    -----
    sampler = ElementSampler()

    # Fully random
    item = sampler.sample()

    # Partial config — missing fields are sampled uniformly
    item = sampler.sample(SamplerConfig(element_name='deriving_demand', domain='finance'))

    # Batch
    items = sampler.sample_batch(n=10, config=SamplerConfig(tags='first_person'))

    # Seeded example
    sampler = ElementSampler(seed=0)   # reproducible questions and option order

    Option order is shuffled for every sampled question (uniform correct
    position) unless shuffle_options=False. Samples whose options collide
    after rounding to 2 decimals are redrawn (up to max_attempts times).
    Templates are corrected at load time (template_fixes.py), and templates
    with placeholders the filler never fills are skipped.
    """

    def __init__(
        self,
        manifest: Optional[dict] = None,
        seed: Optional[int] = None,
        shuffle_options: bool = True,
        max_attempts: int = 50,
    ):
        self._manifest = manifest or load_manifest()
        self._sampleable = get_sampleable_elements(self._manifest)
        self._template_cache: dict[str, Dataset] = {}
        self.shuffle_options = shuffle_options
        self.max_attempts = max_attempts
        self.seed(seed)

    def seed(self, seed: Optional[int]) -> None:
        """Reseed the sampler (template/option choice) and all filler RNGs."""
        if seed is not None:
            self._rng = random.Random(seed)
            seed_everything(seed)
        else:
            # Unseeded sampler: draw template and option order from the global
            # `random` module, so callers that seed the global RNGs (e.g. the
            # release exporter's seed_everything) get reproducible output. A
            # private random.Random(None) was seeded from the OS every run.
            self._rng = random

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @_locked
    def sample(self, config: Optional[SamplerConfig] = None) -> dict:
        """
        Sample one element. Any unspecified config fields are sampled uniformly.

        Returns a dict with keys: question, options, answer, metadata
        (or sub_questions, metadata for multi-part elements).
        """
        if config is None:
            config = SamplerConfig()
        return self._sample_one(config)

    @_locked
    def sample_batch(self, n: int, config: Optional[SamplerConfig] = None) -> list[dict]:
        """
        Sample n elements, each independently. Config fields are re-sampled
        independently for each item unless specified.
        """
        if config is None:
            config = SamplerConfig()
        return [self._sample_one(config) for _ in range(n)]

    def template_ids(self, element_name: str, pool: str = 'questions', **filters) -> list[int]:
        """Sorted template row ids of an element's pool, optionally filtered by domain, type,
        difficulty_level (public scale) or tags."""
        q, _ = self._pool_frames(self._load_template(element_name), pool)
        mask = self._filter_mask(q, element_name, SamplerConfig(element_name=element_name, pool=pool, **filters))
        return sorted(int(x) for x in q[mask]['question_id'].unique().tolist())

    @_locked
    def sample_question(
        self,
        element_name: str,
        seed: int,
        template_id=None,
        pool: str = 'questions',
        **filters,
    ) -> dict:
        """
        One question determined by (element, template, seed) alone, independent of anything sampled before.

        Without template_id, the template is drawn from the pool (after the optional filters
        domain, type, difficulty_level, tags; a template text with several fill settings counts once, and the fields
        of SAMPLING_BALANCE are balanced) with an RNG derived from the seed. The filler then runs after
        seed_everything(seed), and each part's options are shuffled with an RNG derived from (seed, part),
        except parts whose options refer to each other by position. So sample_question(e, seed) and
        sample_question(e, seed, template_id=<the template it chose>) return the same question.

        Returns {element, pool, template_id, seed, metadata, parts}; each part has question, options (texts in
        presented order), correct (bools), order (template index of each presented option) and shuffled.
        """
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) <= SEED_MAX:
            raise ValueError(f"seed must be an integer in [0, 2**63), got {seed!r}")
        seed = int(seed)
        if element_name not in self._sampleable:
            raise ValueError(f"Element {element_name!r} cannot be sampled (no templates or no filler).")
        template = self._load_template(element_name)
        q, options = self._pool_frames(template, pool)
        if template_id is None:
            mask = self._filter_mask(
                q, element_name, SamplerConfig(element_name=element_name, pool=pool, **filters))
            if not mask.any():
                raise ValueError(f"No template questions found for element {element_name!r} matching {filters}.")
            chosen = self._choose_id(q, q[mask], element_name, SamplerConfig(element_name=element_name, pool=pool, **filters),
                                     random.Random(derive_seed('template', seed)))
        else:
            chosen = parse_template_id(template_id, element_name, pool)
        q_df = q[q['question_id'] == chosen].sort_values('sub_id')
        if q_df.empty:
            raise ValueError(f"{element_name!r} has no {pool!r} template {chosen}.")
        opts_df = options[options['question_id'] == chosen].sort_values(['sub_id', 'option_id'])
        row = q_df.iloc[0]
        metadata = {'domain': row['domain'], 'type': row['type'],
                    'difficulty_level': row['difficulty_level'], 'tags': row['tags']}

        seed_everything(seed)
        question_texts, options_list, correct_answer = self._fill_template(element_name, q_df, opts_df, metadata)
        parts = []
        for part, (text, opts, mask) in enumerate(zip(question_texts, options_list, correct_answer)):
            texts = [str(format_value(o)) for o in opts]
            correct = [bool(a) for a in mask]
            order = list(range(len(texts)))
            shuffled = self.shuffle_options and len(texts) > 1 and not any(POSITIONAL_RE.search(t) for t in texts)
            if shuffled:
                random.Random(derive_seed('options', seed, part)).shuffle(order)
            parts.append({
                'question': text,
                'options': [texts[i] for i in order],
                'correct': [correct[i] for i in order],
                'order': order,
                'shuffled': shuffled,
            })
        return {
            'element': element_name,
            'pool': pool,
            'template_id': f'{element_name}/{pool}/{chosen}',
            'seed': seed,
            'metadata': {
                'domain': metadata['domain'],
                'type': metadata['type'],
                'difficulty_level': public_difficulty(element_name, metadata['difficulty_level']),
                'tags': metadata['tags'],
            },
            'parts': parts,
        }

    def rebuild_manifest(self) -> None:
        """Force-rebuild the manifest and reload it."""
        from .manifest import build_manifest
        self._manifest = build_manifest(verbose=True)
        self._sampleable = get_sampleable_elements(self._manifest)
        self._template_cache.clear()
        _SHARED_TEMPLATES.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_one(self, config: SamplerConfig) -> dict:
        element_name = self._resolve_element(config)
        template = self._load_template(element_name)
        q_template_df, opts_template_df, metadata = self._pick_template_question(
            template, config, element_name
        )

        question_texts, options_list, correct_answer = self._fill_template(
            element_name, q_template_df, opts_template_df, metadata
        )

        return _build_result(
            element_name, metadata, question_texts, options_list, correct_answer,
            rng=self._rng if self.shuffle_options else None,
        )

    def _fill_template(self, element_name: str, q_template_df, opts_template_df, metadata: dict) -> tuple:
        """Call the filler (redrawing on duplicate options) and fill the template's texts and options."""
        # Fillers whose answer depends on the wording (e.g. which outcome is
        # asked for) may read the template text; others ignore this key.
        filler_metadata = dict(metadata)
        filler_metadata['question_text'] = ' '.join(q_template_df['question_text'].tolist())
        for _ in range(self.max_attempts):
            fill_values = self._call_filler(element_name, filler_metadata)

            if 'options_list' in fill_values:
                raw_opts = fill_values.pop('options_list')
                raw_ans = fill_values.pop('correct_answer')
                options_list, correct_answer = _normalize_options(raw_opts, raw_ans)
            else:
                sub_ids = q_template_df['sub_id'].unique()
                options_list = []
                correct_answer = []
                for sid in sub_ids:
                    sid_opts = opts_template_df[
                        opts_template_df['sub_id'] == sid
                    ]
                    options_list.append(sid_opts['option_text'].tolist())
                    correct_answer.append(sid_opts['correct_answer'].tolist())

            fill_values = format_fill_values(fill_values)
            # Template-provided options may contain {placeholders} too.
            options_list = [[_fill_text(o, fill_values) for o in opts] for opts in options_list]
            if not _has_duplicate_options(options_list):
                break
        else:
            raise RuntimeError(
                f"Filler for '{element_name}' produced duplicate options in "
                f"{self.max_attempts} attempts (metadata {metadata})."
            )

        question_texts = [
            _fill_question(row['question_text'], fill_values)
            for _, row in q_template_df.iterrows()
        ]
        return question_texts, options_list, correct_answer

    def _resolve_element(self, config: SamplerConfig) -> str:
        if config.element_name is not None:
            if config.element_name not in self._manifest['elements']:
                raise ValueError(
                    f"Element '{config.element_name}' not found in manifest. "
                    f"Available: {sorted(self._manifest['elements'].keys())}"
                )
            entry = self._manifest['elements'][config.element_name]
            if not entry['has_filler']:
                raise ValueError(
                    f"Element '{config.element_name}' has no filler function. "
                    f"Elements without fillers cannot be sampled deterministically."
                )
            return config.element_name
        return self._rng.choice(self._sampleable)

    def _load_template(self, element_name: str) -> Dataset:
        if element_name not in self._template_cache:
            key = (element_name, self._manifest.get('generated_at'))
            dataset = _SHARED_TEMPLATES.get(key)
            if dataset is None:
                # the corrections are deterministic and slow: outside the lock, so draws go on meanwhile (code
                # templates are built by running code, so those are loaded under it)
                pure = not (code_templates.has_code_templates(element_name) or code_templates.is_composite(element_name))
                fixed = apply_template_fixes(element_name, load_raw_templates(element_name)) if pure else None
                with RNG_LOCK:
                    dataset = _SHARED_TEMPLATES.get(key)
                    if dataset is None:
                        if fixed is None:
                            fixed = apply_template_fixes(element_name, load_raw_templates(element_name))
                        dataset = self._drop_unfillable(element_name, fixed)
                        # templates loaded under an older manifest are not needed by new samplers: let them go
                        for old in [k for k in _SHARED_TEMPLATES if k[1] != key[1]]:
                            del _SHARED_TEMPLATES[old]
                        _SHARED_TEMPLATES[key] = dataset
            self._template_cache[element_name] = dataset
        return self._template_cache[element_name]

    def warm(self, min_templates: int = 2000) -> list[str]:
        """Load the elements with at least min_templates templates (largest first), so the first request for one
        does not wait for it. Returns the elements loaded."""
        big = sorted((e for e in self._sampleable
                      if (self._manifest['elements'].get(e) or {}).get('template_question_count', 0) >= min_templates),
                     key=lambda e: -self._manifest['elements'][e]['template_question_count'])
        for e in big:
            try:
                self._load_template(e)
            except Exception:  # an element that cannot load fails on its own request, as before
                pass
        return big

    def _drop_unfillable(self, element_name: str, dataset: Dataset, tries: int = 5) -> Dataset:
        """
        Drop templates that use a {placeholder} the filler never returns for
        that template's metadata (checked by calling the filler `tries` times
        per metadata combination). RNG states are saved and restored so that
        loading a template does not change what is sampled afterwards.
        """
        if code_templates.has_code_templates(element_name):
            # written with their filler; the filler needs the template text, which the probe does not pass
            return dataset
        fn = function_registry.get(element_name)
        saved = (random.getstate(), np.random.get_state(), filler_utils.rng.bit_generator.state)
        keys_cache: dict[tuple, set] = {}
        errors: dict[tuple, str] = {}
        reasons: dict = {}
        try:
            # Probe from a fixed state, so which templates are kept never depends on what was sampled before.
            seed_everything(derive_seed('drop_unfillable', element_name))
            for q_attr, o_attr in (('questions', 'options'), ('example_questions', 'example_options')):
                q = getattr(dataset, q_attr)
                if q is None or len(q) == 0:
                    continue
                if 'question_id' not in q.columns:
                    q = q.reset_index()
                bad = set()
                for _, row in q.iterrows():
                    md = (row['domain'], row['type'], row['difficulty_level'], row['tags'])
                    if md not in keys_cache:
                        keys = set()
                        meta = dict(zip(('domain', 'type', 'difficulty_level', 'tags'), md))
                        meta['difficulty_level'] = int(meta['difficulty_level'])
                        for _ in range(tries):
                            try:
                                keys |= set(fn(dict(meta)) or {})
                            except Exception as err:  # noqa: BLE001
                                errors[md] = f'{type(err).__name__}: {err}'
                        keys_cache[md] = keys
                    try:
                        need = extract_fstring_keys(row['question_text'])
                    except ValueError:
                        bad.add(row['question_id'])
                        reasons[row['question_id']] = 'template text is not a valid format string'
                        continue
                    missing = need - keys_cache[md]
                    if missing:
                        bad.add(row['question_id'])
                        reasons[row['question_id']] = (f'filler error for {md}: {errors[md]}' if md in errors
                                                       else f'filler never returns {sorted(missing)} for {md}')
                if bad and len(bad) >= q['question_id'].nunique():
                    # Dropping everything means the filler is broken (e.g. an
                    # environment change), not that some templates are unfillable.
                    example = next(iter(errors.items()), None)
                    raise RuntimeError(
                        f"every {q_attr} template of {element_name!r} is unfillable"
                        + (f"; filler error for {example[0]}: {example[1]}" if example else '')
                    )
                if bad:
                    # what was dropped and why, so it is visible (DROPPED[element][pool]; logged once per load)
                    report = DROPPED.setdefault(element_name, {})[q_attr] = {}
                    for qid in sorted(bad):
                        report.setdefault(reasons.get(qid, 'unknown'), []).append(int(qid))
                    log.info('%s: %d %s templates dropped as unfillable: %s', element_name, len(bad), q_attr,
                             {r: len(v) for r, v in report.items()})
                    setattr(dataset, q_attr, q[~q['question_id'].isin(bad)].reset_index(drop=True))
                    o = getattr(dataset, o_attr)
                    if o is not None and len(o):
                        if 'question_id' not in o.columns:
                            o = o.reset_index()
                        setattr(dataset, o_attr, o[~o['question_id'].isin(bad)].reset_index(drop=True))
        finally:
            random.setstate(saved[0])
            np.random.set_state(saved[1])
            filler_utils.rng.bit_generator.state = saved[2]
        return dataset

    def _pick_template_question(
        self, template: Dataset, config: SamplerConfig, element_name: str
    ) -> tuple:
        """
        Filter the template by any specified config fields, then pick a random
        valid question_id. Returns (q_df, opts_df, metadata) where metadata is
        read from the chosen row — guaranteeing only real combinations are used.
        """
        q, options = self._pool_frames(template, config.pool)
        mask = self._filter_mask(q, element_name, config)

        filtered = q[mask]
        if filtered.empty:
            raise ValueError(
                f"No template questions found for element '{element_name}' "
                f"matching config {config}."
            )

        chosen_id = self._choose_id(q, filtered, element_name, config, self._rng)

        q_df = q[q['question_id'] == chosen_id].sort_values('sub_id')
        opts_df = options[
            options['question_id'] == chosen_id
        ].sort_values(['sub_id', 'option_id'])

        row = q_df.iloc[0]
        metadata = {
            'domain': row['domain'],
            'type': row['type'],
            'difficulty_level': row['difficulty_level'],
            'tags': row['tags'],
        }

        return q_df, opts_df, metadata

    def _choose_id(self, q, filtered, element_name: str, config: SamplerConfig, rng) -> int:
        """The question_id of the template to fill, among `filtered` (rows of q). A field of SAMPLING_BALANCE that the
        request leaves open is drawn first. Templates that share their text and level differ only in what the filler
        is told (mult_div's sign, the functional form of diminishing_marginal_product): one template with several
        fill settings, so a text is drawn first, then one of its settings. Elements with neither draw uniformly over
        the templates, as before."""
        balance = SAMPLING_BALANCE.get(element_name)
        if balance and getattr(config, balance[0], None) is None:
            field, weights = balance
            values = sorted(filtered[field].astype(str).unique())
            if len(values) > 1:
                pick = rng.choices(values, weights=[(weights or {}).get(v, 1) for v in values])[0]
                filtered = filtered[filtered[field].astype(str) == pick]
        # (numpy over arrays computed once per element and pool; the draws are those of the list version: ids sorted,
        # a text's members sorted, texts ordered by their smallest member, the same rng calls in the same order)
        ids = np.unique(filtered['question_id'].to_numpy().astype(np.int64))
        all_ids, group_of = self._text_group_arrays(q, element_name, config.pool)
        pos = np.searchsorted(all_ids, ids)
        if pos.size and (pos[-1] >= len(all_ids) or not np.array_equal(all_ids[pos], ids)):
            raise RuntimeError(f"{element_name}: filtered templates that are not in the pool")
        groups = group_of[pos]
        distinct, first = np.unique(groups, return_index=True)
        if len(distinct) == len(ids):
            return int(ids[rng.randrange(len(ids))])
        text = distinct[np.argsort(first, kind='stable')][rng.randrange(len(distinct))]
        members = ids[groups == text]
        return int(members[rng.randrange(len(members))])

    def _text_group_arrays(self, q, element_name: str, pool) -> tuple:
        """Every question_id of q, sorted, and the text group (see _text_groups) of each."""
        cache = self.__dict__.setdefault('_text_group_arrays_cache', {})
        key = (element_name, pool)
        if key not in cache or cache[key][0] is not q:
            groups = self._text_groups(q, element_name, pool)
            all_ids = np.array(sorted(groups), dtype=np.int64)
            cache[key] = (q, all_ids, np.array([groups[i] for i in all_ids.tolist()], dtype=np.int64))
        return cache[key][1], cache[key][2]

    def _text_groups(self, q, element_name: str, pool) -> dict:
        """question_id -> the smallest question_id with the same text (every part, spaces collapsed) and level."""
        cache = self.__dict__.setdefault('_text_group_cache', {})
        key = (element_name, pool)
        if key not in cache or cache[key][0] is not q:
            qs = q.sort_values(['question_id', 'sub_id'], kind='stable')
            text = qs['question_text'].astype(str).str.split().str.join(' ')
            g = text.groupby(qs['question_id']).agg(tuple).to_frame('text')
            g['level'] = qs.groupby('question_id')['difficulty_level'].first().astype(str)
            first = g.reset_index().groupby(['text', 'level'])['question_id'].transform('min')
            cache[key] = (q, dict(zip(g.index.tolist(), first.tolist())))
        return cache[key][1]

    @staticmethod
    def _pool_frames(template: Dataset, pool) -> tuple:
        if pool in (None, 'questions'):
            q, options = template.questions, template.options
        elif pool == 'examples':
            q, options = template.example_questions, template.example_options
        else:
            raise ValueError(f"Unknown pool {pool!r}; use 'questions' or 'examples'.")
        if q is None or options is None:
            raise ValueError(f"No template questions found in pool {pool!r}.")
        if 'question_id' not in q.columns:
            q = q.reset_index()
        if 'question_id' not in options.columns:
            options = options.reset_index()
        return q, options

    @staticmethod
    def _filter_mask(q, element_name: str, config: SamplerConfig):
        mask = q['question_id'].notna()

        if config.domain is not None:
            mask = mask & (q['domain'] == config.domain)
        if config.type is not None:
            mask = mask & (q['type'] == config.type)
        if config.difficulty_level is not None:
            mask = mask & (q['difficulty_level'].map(lambda v: public_difficulty(element_name, v)).astype(str)
                           == str(config.difficulty_level))
        if config.tags is not None:
            mask = mask & (q['tags'] == config.tags)
        return mask

    def _call_filler(self, element_name: str, metadata: dict) -> dict:
        fn = function_registry.get(element_name)
        result = fn(metadata)
        if result is None:
            raise RuntimeError(
                f"Filler function for '{element_name}' returned None for metadata {metadata}."
            )
        validate_filler_output(element_name, result)
        return result
