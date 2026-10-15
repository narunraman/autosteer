"""Hugging Face release packages for STEER and STEER-ME, built locally. Nothing is uploaded.

    autosteer release build --out out/release-2026-10-07
    autosteer release check out/release-2026-10-07

`build` runs the exporter (test and few_shot splits of every released element, see autosteer.release.export) into
<out>/export, then lays out one directory per Hugging Face dataset repo:

    <out>/steer/        -> narunraman/steer
    <out>/steer-me/     -> narunraman/steer_me
        README.md           dataset card with the YAML metadata (configs, license, task categories, size)
        ELEMENTS.md         per element: how its questions are made, what varies, grading, counts
        elements.csv        the same, machine-readable (lists joined with ';')
        elements.jsonl
        provenance.json     generator commit, seed, data files commit, export rule, date, file checksums
        data/<element>/test.parquet, data/<element>/few_shot.parquet

Configs: `default` (all elements) plus one per element, named after the element. Setting and module are columns.
Like the exporter, `build` refuses a checkout with uncommitted changes. `--from-export DIR` packages an existing
export instead of generating one (it must come from the current commit). `check` validates a built release:
schema of every Parquet file, counts against web/content/dataset_counts.json, few_shot/test disjointness (template
ids, template texts, questions), every released element present, and (when the `datasets` package is importable)
every config loads with datasets.load_dataset("parquet", data_files=...).
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import glob
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

from autosteer.release.export import DATASET_DIRS, DEFAULT_SEED

REPOS = {"STEER": "narunraman/steer", "STEER-ME": "narunraman/steer_me"}
REPO_DIRS = {"STEER": "steer", "STEER-ME": "steer-me"}
SITE = "https://steer-benchmark.cs.ubc.ca"
EVAL_REPO = "https://github.com/narunraman/STEER"
PAPERS = {
    "STEER": ("STEER: Assessing the Economic Rationality of Large Language Models", "ICML 2024",
              "https://arxiv.org/abs/2402.09552"),
    "STEER-ME": ("STEER-ME: Assessing the Microeconomic Reasoning of Large Language Models",
                 "NeurIPS 2025 Datasets and Benchmarks Track", "https://arxiv.org/abs/2502.13119"),
}
SITE_PAGES = {"STEER": "steer", "STEER-ME": "steer-me"}
LICENSE = "cc-by-4.0"
LICENSE_NAME = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
SPLITS = ("test", "few_shot")
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"
COUNTS_JSON = Path(__file__).resolve().parent.parent / "web" / "content" / "dataset_counts.json"
RULE_RE = re.compile(r"^[a-z_]+:\d+(,\d+)*$")

# Consistency rules (autosteer.release.consistency): rule id -> when the choices of the two parts are consistent.
RULES = {
    "same_brand": "Part 0 offers two brands (price and quality) and \"no preference\"; part 1 adds a decoy that is "
                  "worse than one of them on both. Consistent: the same brand in both parts, or no preference in "
                  "both.",
    "ellsberg_seu": "Ellsberg's three-colour urn: two pairs of bets, one with known and one with unknown odds, each "
                    "with an \"indifferent\" option. Consistent: the pairs that one subjective probability can "
                    "explain (the known-odds bet in exactly one part, or indifferent in both).",
    "mirror_trade": "You own item X and are offered Y; then you own Y and are offered X (Yes / No / indifferent). "
                    "Consistent: Yes in exactly one part, or indifferent in both.",
    "common_ratio_same_side": "Part 0: a sure amount or a lottery; part 1: both scaled down by the same "
                              "probability. Consistent: the sure option and its scaled-down version, or the lottery in both parts.",
    "reflection_same_side": "Part 0: a sure gain or a lottery over gains; part 1: the mirrored losses, with equal "
                            "expected values. Consistent: the same side in both parts.",
    "pairwise_complete": "Four or five fully described alternatives (sure outcomes, or lotteries); one part per pair, "
                         "each with the options prefer one, prefer the other, indifferent, and cannot decide / need "
                         "more information. Consistent: no pair answered \"cannot decide\" (the answers are complete). "
                         "Reported besides: whether the answered pairs are transitive.",
}
STATED_PRIOR = ("In type `stated_prior` the question states which option the reader chose in an earlier, related "
                "choice; the keyed option is the one on the same side.")

# Columns of the released tables (autosteer.release.schema.SCHEMA), for the cards.
def _difficulty_fact(levels) -> str:
    """ELEMENTS.md: an element's (relative) difficulty levels, "difficulty levels 1–3" or "difficulty level 1"."""
    levels = sorted(levels)
    if len(levels) == 1:
        return f"difficulty level {levels[0]} (one level)"
    if levels == list(range(levels[0], levels[-1] + 1)):
        return f"difficulty levels {levels[0]}\u2013{levels[-1]}"
    return f"difficulty levels {', '.join(map(str, levels))}"


FIELDS = [
    ("element", "string", "element id; also the config name"),
    ("question_id", "string", "`<element>/<base_id>/<part>`, unique in the dataset"),
    ("base_id", "string", "one question (16 hex digits of a content hash); shared by its parts"),
    ("part", "int32", "0-based part index within `base_id`; most questions have one part"),
    ("template_id", "string", "`<element>/<pool>/<n>`: the template the question was filled from"),
    ("question_text", "string", "the prompt"),
    ("options", "list<string>", "answer options in presentation order, without letter labels"),
    ("grading", "string", "`answer` or `consistency`"),
    ("correct_index", "int32", "0-based index of the correct option; null for consistency-graded rows"),
    ("consistency_group", "string", "rows scored together (`<element>/<base_id>`); null for answer-graded rows"),
    ("consistency_rule", "string", "`<rule_id>:<allowed>`; null for answer-graded rows"),
    ("type", "string", "question type within the element (`default` if the element has one)"),
    ("domain", "string", "subject matter of the wording (e.g. `consumer_goods`)"),
    ("perspective", "string", "`first_person`, `second_person`, `anon`, `man`, `woman`, `ai_assistant`; null if none"),
    ("tags", "list<string>", "other template descriptors (may be empty)"),
    ("difficulty", "int32", "difficulty level within the element: 1 is its easiest, up to its number of levels; not comparable across elements"),
    ("setting", "string", "setting of the taxonomy"),
    ("module", "string", "module of the taxonomy"),
    ("generator", "string", "`autosteer@<version>+<commit>` that generated the row"),
    ("seed", "int64", "with `element` and `template_id`, regenerates the question"),
]


# --------------------------------------------------------------------------------------------------------------
# Inputs: element catalogue in paper order (as on the site), citations (from the site's Cite section)
# --------------------------------------------------------------------------------------------------------------
def bibtex() -> dict[str, str]:
    """BibTeX of both papers, from the Cite section of the site (templates/guide_welcome.html)."""
    html = (TEMPLATES_DIR / "guide_welcome.html").read_text(encoding="utf-8")
    out = {}
    for bench, pre_id in (("STEER", "bib-steer"), ("STEER-ME", "bib-steerme")):
        m = re.search(rf'<pre id="{pre_id}">\{{% raw %\}}(.*?)\{{% endraw %\}}</pre>', html, re.S)
        if not m:
            raise SystemExit(f"no BibTeX for {bench} in {TEMPLATES_DIR / 'guide_welcome.html'}")
        out[bench] = m.group(1).strip()
    return out


def paper_layout(benchmark: str) -> list[dict]:
    """Settings -> modules -> elements in paper order, as the site shows them (autosteer.web.site.tree)."""
    from autosteer.web import site

    return site.tree(benchmark)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _size(n: int) -> str:
    for lo, hi, label in ((0, 1_000, "n<1K"), (1_000, 10_000, "1K<n<10K"), (10_000, 100_000, "10K<n<100K"),
                          (100_000, 1_000_000, "100K<n<1M")):
        if lo <= n < hi:
            return label
    return "1M<n<10M"


def _fmt(n: int) -> str:
    return f"{n:,}"


# --------------------------------------------------------------------------------------------------------------
# Element records
# --------------------------------------------------------------------------------------------------------------
def element_records(benchmark: str, manifest: dict, data_dir: Path) -> list[dict]:
    """One record per released element of the benchmark, in paper order, with values read from its files."""
    import pyarrow.parquet as pq

    from autosteer.release.rows import catalog

    cat = catalog()
    released = {e for e, r in manifest["elements"].items() if r["benchmark"] == benchmark and r.get("file")}
    records, placed = [], set()
    for s in paper_layout(benchmark):
        for m in s["modules"]:
            for e in m["elements"]:
                name = e.get("element")
                if not name or name not in released or e.get("shared") or name in placed:
                    continue
                placed.add(name)
                d, rec = cat[name], manifest["elements"][name]
                test = pq.read_table(data_dir / name / "test.parquet").to_pylist()
                few_path = data_dir / name / "few_shot.parquet"
                few = pq.read_table(few_path).to_pylist() if few_path.exists() else []
                gradings = sorted({r["grading"] for r in test})
                fs = rec.get("few_shot", {})
                records.append({
                    "element": name,
                    "display_name": d["display_name"],
                    "benchmark": benchmark,
                    "paper_id": e.get("id") or "",
                    "paper_name": (e["display_name"] if benchmark == "STEER" else d.get("paper_name"))
                    or d["display_name"],
                    "setting": d["setting"],
                    "module": d["module"],
                    "summary": d.get("summary", ""),
                    "description": d.get("description", ""),
                    "grading": gradings[0] if len(gradings) == 1 else "mixed",
                    "grading_rule": grading_text(test),
                    "consistency_types": sorted({r["type"] for r in test if r["grading"] == "consistency"}),
                    "types": sorted({r["type"] for r in test}),
                    "domains": sorted({r["domain"] for r in test}),
                    "perspectives": sorted({r["perspective"] for r in test if r["perspective"]}),
                    "difficulties": sorted({r["difficulty"] for r in test}),
                    "test_questions": len({r["base_id"] for r in test}),
                    "test_rows": len(test),
                    "few_shot_questions": len({r["base_id"] for r in few}),
                    "few_shot_rows": len(few),
                    "few_shot_source": "+".join(s for s in fs.get("sources", []) if s != "none")
                    + ("+partial" if "none" in fs.get("sources", []) and few else "") or "none",
                    "aliases": d.get("aliases", []),
                    "example": d.get("example", ""),
                })
    missing = sorted(released - placed)
    if missing:
        raise SystemExit(f"{benchmark}: released elements the site does not place: {missing}")
    return records


def grading_text(rows: list[dict]) -> str:
    """How an element's rows are graded, from the rows themselves (public wording; no working notes)."""
    parts = []
    answer_types = sorted({r["type"] for r in rows if r["grading"] == "answer"})
    if answer_types:
        parts.append("One correct option per question part (`correct_index`); score is accuracy.")
        if "stated_prior" in answer_types:
            parts.append(STATED_PRIOR)
    by_rule: dict[str, set] = {}
    for r in rows:
        if r["grading"] == "consistency":
            by_rule.setdefault(r["consistency_rule"].split(":", 1)[0], set()).add(r["type"])
    for rule, types in sorted(by_rule.items()):
        where = f" (type {', '.join(f'`{t}`' for t in sorted(types))})" if answer_types else ""
        parts.append(f"Consistency-graded{where} with rule `{rule}`: {RULES.get(rule, '')} Score: share of "
                     "consistent groups.")
    return " ".join(parts)


def covered_and_shared(benchmark: str) -> tuple[list[dict], list[dict]]:
    """Paper elements covered by another element, and (STEER-ME) STEER elements shared with its Foundations."""
    covered, shared = [], []
    for s in paper_layout(benchmark):
        for m in s["modules"]:
            for e in m["elements"]:
                if e.get("status") == "covered":
                    covered.append({"paper_id": e["id"], "paper_name": e["display_name"], "covered_by": e["covered_by"]})
                if e.get("shared"):
                    shared.append({"element": e["element"], "display_name": e["display_name"]})
    return covered, shared


def held_elements(benchmark: str) -> list[dict]:
    """Paper elements the site lists as held (no released data)."""
    out = []
    for s in paper_layout(benchmark):
        for m in s["modules"]:
            for e in m["elements"]:
                if e.get("status") == "held" and not e.get("shared"):
                    out.append({"paper_id": e.get("id") or "", "paper_name": e["display_name"]})
    return out


# --------------------------------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------------------------------
CSV_COLUMNS = ["element", "display_name", "benchmark", "paper_id", "paper_name", "setting", "module", "summary",
               "description", "grading", "grading_rule", "consistency_types", "types", "domains", "perspectives",
               "difficulties", "test_questions", "test_rows", "few_shot_questions", "few_shot_rows",
               "few_shot_source", "aliases"]


def write_elements(repo: Path, records: list[dict]) -> None:
    with (repo / "elements.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: r[k] for k in CSV_COLUMNS}, ensure_ascii=False) + "\n")
    with (repo / "elements.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in records:
            w.writerow([";".join(map(str, r[k])) if isinstance(r[k], list) else r[k] for k in CSV_COLUMNS])


def _by_setting(records: list[dict]) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {}
    for r in records:
        out.setdefault(r["setting"], {}).setdefault(r["module"], []).append(r)
    return out


def elements_md(benchmark: str, records: list[dict]) -> str:
    covered, shared = covered_and_shared(benchmark)
    lines = [f"# {benchmark} elements", "",
             f"How the questions of each element of [`{REPOS[benchmark]}`](README.md) are made and graded. "
             "The same information, machine-readable, is in `elements.csv` and `elements.jsonl`. "
             f"Element pages with sample questions: {SITE}/{SITE_PAGES[benchmark]}.", ""]
    for setting, modules in _by_setting(records).items():
        lines += [f"## {setting}", ""]
        for module, recs in modules.items():
            lines += [f"### {module}", ""]
            for r in recs:
                pid = f"{r['paper_id']} " if r["paper_id"] else ""
                lines += [f"#### {pid}{r['display_name']} (`{r['element']}`)", ""]
                if r["paper_name"] != r["display_name"]:
                    lines += [f"Paper name: {r['paper_name']}.", ""]
                lines += [r["summary"], "", f"**How the questions are made.** {r['description']}", "",
                          f"**Grading.** {r['grading_rule']}", ""]
                facts = [f"types: {', '.join(r['types'])}", f"domains: {', '.join(r['domains'])}",
                         _difficulty_fact(r["difficulties"])]
                if r["perspectives"]:
                    facts.append(f"perspectives: {', '.join(r['perspectives'])}")
                facts.append(f"test: {_fmt(r['test_questions'])} questions ({_fmt(r['test_rows'])} rows); "
                             f"few_shot: {r['few_shot_questions']} questions")
                if r["aliases"]:
                    facts.append("earlier ids: " + ", ".join(f"`{a}`" for a in r["aliases"]))
                lines += [f"- {x}" for x in facts] + [""]
    if covered:
        lines += ["## Paper elements covered by another element", "",
                  "| paper id | paper element | covered by |", "|---|---|---|"]
        lines += [f"| {c['paper_id']} | {c['paper_name']} | `{c['covered_by']}` |" for c in covered] + [""]
    if shared:
        lines += ["## Elements shared with STEER", "",
                  "STEER-ME's Foundations include these STEER elements. They are published once, in "
                  f"[`{REPOS['STEER']}`](https://huggingface.co/datasets/{REPOS['STEER']}): "
                  + ", ".join(f"`{s['element']}`" for s in shared) + ".", ""]
    return "\n".join(lines)


def _yaml_configs(records: list[dict]) -> list[str]:
    def files(pattern_dir: str, has_few: bool) -> list[str]:
        out = ["  data_files:", "  - split: test", f"    path: data/{pattern_dir}/test.parquet"]
        if has_few:
            out += ["  - split: few_shot", f"    path: data/{pattern_dir}/few_shot.parquet"]
        return out

    lines = ["configs:", "- config_name: default", "  default: true"] + files("*", True)
    for r in records:
        lines += [f"- config_name: {r['element']}"] + files(r["element"], r["few_shot_rows"] > 0)
    return lines


def card(benchmark: str, records: list[dict], prov: dict) -> str:
    repo, (title, venue, arxiv) = REPOS[benchmark], PAPERS[benchmark]
    n_test_q = sum(r["test_questions"] for r in records)
    n_test = sum(r["test_rows"] for r in records)
    n_few_q = sum(r["few_shot_questions"] for r in records)
    n_few = sum(r["few_shot_rows"] for r in records)
    no_few = [r["element"] for r in records if not r["few_shot_rows"]]
    consistency = [r for r in records if r["grading"] != "answer"]
    _, shared = covered_and_shared(benchmark)
    first = records[0]["element"]
    tags = (["economics", "rationality", "decision-making", "game-theory", "social-choice", "reasoning", "benchmark"]
            if benchmark == "STEER" else ["economics", "microeconomics", "reasoning", "benchmark"])
    out = ["---", f"pretty_name: {benchmark}", f"license: {LICENSE}", "language:", "- en", "task_categories:",
           "- multiple-choice", "- question-answering", "tags:", *[f"- {t}" for t in tags], "size_categories:",
           f"- {_size(n_test + n_few)}", *_yaml_configs(records), "---", ""]

    if benchmark == "STEER":
        what = ("STEER measures how well large language models reason about economic decisions, from arithmetic "
                "and probability through single-agent choice under certainty and risk, strategic reasoning in "
                "games, and decisions made on behalf of others (social choice and mechanism design). Each "
                "*element* of the paper's taxonomy is one skill, for example applying Bayes' rule or finding a "
                "pure-strategy Nash equilibrium; some elements test for behavioural biases (certainty effect, "
                "sunk cost, endowment effect) by checking that related answers are consistent.")
    else:
        what = ("STEER-ME measures how well large language models reason about microeconomics in non-strategic "
                "settings: consumption, production, and markets with many agents (demand and supply, elasticities, "
                "surplus, equilibrium prices, taxes and the welfare theorems). Each *element* of the paper's "
                "taxonomy is one skill, for example applying the law of demand or computing a deadweight loss.")
    out += [f"# {benchmark}", "", what, "",
            f"- Paper: {title} ({venue}), [{arxiv.rsplit('/', 1)[-1]}]({arxiv})",
            f"- Website, with element pages and sample questions: {SITE}/{SITE_PAGES[benchmark]}",
            f"- Evaluation code: [narunraman/STEER]({EVAL_REPO})", ""]
    if benchmark == "STEER":
        out += [f"The companion benchmark STEER-ME is [`{REPOS['STEER-ME']}`]"
                f"(https://huggingface.co/datasets/{REPOS['STEER-ME']}).", ""]
    else:
        out += [f"STEER-ME's Foundations also use {len(shared)} STEER elements ("
                + ", ".join(f"`{s['element']}`" for s in shared)
                + f"); they are published once, in [`{REPOS['STEER']}`](https://huggingface.co/datasets/"
                f"{REPOS['STEER']}).", ""]

    out += ["## Contents", "",
            f"{len(records)} elements; `test` has {_fmt(n_test_q)} questions ({_fmt(n_test)} rows), "
            f"`few_shot` {_fmt(n_few_q)} questions ({_fmt(n_few)} rows). A row is one question part: most questions "
            "have one part; consistency-graded questions have several.", "",
            "## Loading", "", "```python", "from datasets import load_dataset", "",
            f'ds = load_dataset("{repo}", "{first}")   # one element: splits test and few_shot',
            f'everything = load_dataset("{repo}")          # all elements (config "default")',
            "```", "",
            "There is one config per element, named after the element, and a `default` config with all of them. "
            "Every row carries its `setting` and `module`, so a setting or module is a filter: "
            "`everything[\"test\"].filter(lambda r: r[\"module\"] == \"...\")`.", "",
            "## How the questions are generated", "",
            "Each element has a set of question templates written for it, which vary the question type, the "
            "domain (subject matter of the story), the difficulty and, where relevant, the perspective (first "
            "person, a named third person, ...). A filler program for the element draws the numbers or other "
            "content for a template, computes the correct option, and builds the distractors from typical "
            "mistakes; options are then shuffled (unless an option refers to others by position, such as 'none of the above'). Every question is determined by `element`, `template_id` and "
            f"`seed`, so it can be regenerated. This release was generated by `{prov['generator']}`; "
            "[ELEMENTS.md](ELEMENTS.md) describes each element's templates, "
            "answer key and distractors.", "",
            "Sample size: a *cell* is one combination of type, domain and difficulty within an element. Each "
            f"cell has {prov['rule']['per_cell']} test questions, and every element has at least "
            f"{prov['rule']['floor']} (when there are fewer than {prov['rule']['floor'] // prov['rule']['per_cell']} "
            "cells, the floor is spread evenly over them). Questions that fail automatic checks (no or several "
            "correct options, duplicate options, unfilled placeholders) or repeat an earlier question are "
            "discarded and redrawn.", "",
            "## Splits", "",
            "- `test`: the benchmark.",
            f"- `few_shot`: {prov['rule']['few_shot']['per_type']} questions per question type of each element, "
            "for use as in-context examples. They come from templates that are never used for `test`: the "
            "element's own example templates where it has them, otherwise templates held out of `test` (only "
            "templates whose removal leaves every test cell covered). No `few_shot` question or template text "
            "occurs in `test`, and no `few_shot` template has the text of a template used for `test`."]
    if no_few:
        out += [f"  {len(no_few)} element{'' if len(no_few) == 1 else 's'} {'has' if len(no_few) == 1 else 'have'} no `few_shot` split because each of " f"{'its' if len(no_few) == 1 else 'their'} templates is the only "
                "one of some test cell: " + ", ".join(f"`{e}`" for e in no_few) + "."]
    out += ["", "## Fields", "", "| field | type | meaning |", "|---|---|---|"]
    out += [f"| `{f}` | {t} | {m} |" for f, t, m in FIELDS]
    out += ["", "`difficulty` is relative to the element: its levels are numbered from 1 (the element's easiest) up "
            "to its number of levels, and many elements have a single level. Levels can be compared within an "
            "element, not across elements.", ""]

    out += ["## Grading", "",
            "**Answer-graded** rows (`grading == \"answer\"`) have one correct option, `options[correct_index]`. "
            "The score of an element is accuracy over its questions; a model's answer is the index of the option "
            "it chooses."]
    if consistency:
        out += ["",
                "**Consistency-graded** rows (`grading == \"consistency\"`) have no correct option. They test for "
                "a behavioural bias: the parts of one question (rows with the same `consistency_group`) are "
                "presented separately, and the choices count as consistent if the tuple of chosen option indices, "
                "in `part` order, is one of the allowed tuples listed in `consistency_rule` "
                "(`<rule_id>:<allowed>`, tuples written as digit strings separated by commas). The score is the "
                "share of consistent groups.", "", "```python",
                "def consistent(choices, rule):  # choices: chosen option index per part, in part order",
                "    allowed = rule.split(\":\", 1)[1].split(\",\")",
                "    return \"\".join(map(str, choices)) in allowed",
                "```", "",
                "Example: `same_brand:02,11` means choosing option 0 in part 0 and option 2 in part 1, or option 1 "
                "in both, is consistent.", "",
                "`pairwise_complete` (completeness, completeness_risk) is the exception: its comma-separated tokens "
                "are not allowed tuples but one token per part, in `part` order, of six digits `abpqic`: the part "
                "compares alternatives `a` and `b` (0-based, in the order the question describes them), and `p`, "
                "`q`, `i`, `c` are the option indices of \"prefer a\", \"prefer b\", \"indifferent\" and "
                "\"cannot decide\". The group is consistent (complete) if no part's choice is its `c`; the "
                "transitivity of the answered pairs is a secondary measure "
                "(`autosteer.release.consistency.score_transitivity`).", "",
                "The rules:", "", "| rule | elements | consistent when |", "|---|---|---|"]
        rules: dict[str, list[str]] = {}
        for r in consistency:
            for rule in sorted({x.split("`")[1] for x in r["grading_rule"].split("rule ")[1:]}):
                rules.setdefault(rule, []).append(f"`{r['element']}`" + (
                    f" (type {', '.join(r['consistency_types'])})" if r["grading"] == "mixed" else ""))
        out += [f"| `{k}` | {', '.join(v)} | {RULES.get(k, '').split('Consistent: ', 1)[-1]} |"
                for k, v in sorted(rules.items())]
        if any("stated_prior" in r["types"] for r in records):
            out += ["", STATED_PRIOR]
    out += [""]

    out += ["## Elements", "",
            f"Per setting, in paper order. [ELEMENTS.md](ELEMENTS.md) has the full description of every element "
            "(how its questions are generated, what varies, how the answer and distractors are computed, how it "
            "is graded); `elements.csv` and `elements.jsonl` have the same information for programs.", ""]
    for setting, modules in _by_setting(records).items():
        out += [f"**{setting}**", ""]
        if benchmark == "STEER":
            out += ["| id | config | element | grading | test questions |", "|---|---|---|---|---|"]
        else:
            out += ["| module | config | element | grading | test questions |", "|---|---|---|---|---|"]
        for module, recs in modules.items():
            for r in recs:
                first_col = r["paper_id"] if benchmark == "STEER" else module
                out.append(f"| {first_col} | `{r['element']}` | {r['display_name']} | {r['grading']} | "
                           f"{_fmt(r['test_questions'])} |")
        out.append("")

    out += ["## Changes from earlier releases", ""]
    if benchmark == "STEER":
        out += ["The data used for the ICML 2024 paper was released in July 2024 as per-element pandas pickles in "
                f"[narunraman/STEER]({EVAL_REPO}). This release replaces it:", "",
                "- Every element was regenerated in October 2026 from templates and filler programs, with the "
                "sample size rule above; scores on this release are not directly comparable with those reported "
                "in the paper.",
                "- Answer keys are computed by the fillers; rows without exactly one correct option are not "
                "released. Consistency-graded elements state their groups and rules explicitly "
                "(`consistency_group`, `consistency_rule`).",
                "- One schema for all elements (the fields above), Parquet files, `few_shot` from held-out "
                "templates, no explanations.",
                "- A few elements of the paper's taxonomy are not in this release, because another element tests the "
                "same behaviour or because they have no questions; the "
                f"[website]({SITE}/{SITE_PAGES[benchmark]}) marks each one."]
    else:
        out += ["This release replaces the first version of this dataset (May 2025: `train.csv` and `test.csv`), "
                "which stays available as revision `v1` of this repository:", "",
                "- Every element was regenerated in October 2026 with the sample size rule above. `test` rows are "
                "not comparable with the 2025 data or with the numbers in the paper.",
                "- `train` is now `few_shot`, drawn from templates that are never used for `test`.",
                "- One schema shared with STEER (the fields above), Parquet files, one config per element.",
                "- Strategic elements (e.g. `backward_induction`, `best_response`, `feasibility`, `trigger`) "
                "belong to STEER and are published there, as are the Foundations elements shared with STEER.",
                "- Duplicate questions are removed; answer keys are computed by the fillers and rows without "
                "exactly one correct option are not released."]
    out += [f"- Provenance of this build (generator commit, seed, data files commit, date, checksums): "
            "[provenance.json](provenance.json).", ""]

    out += ["## License", "", f"{LICENSE_NAME}. You may share and adapt the data with attribution; please cite "
            "the paper below.", "",
            "## Citation", "", "```bibtex", bibtex()[benchmark], "```", ""]
    return "\n".join(out)


def provenance(benchmark: str, manifest: dict, repo: Path, records: list[dict], command: str,
               export_manifest: Path) -> dict:
    from autosteer.release.rows import git_state

    files = {}
    for p in sorted((repo / "data").rglob("*.parquet")):
        import pyarrow.parquet as pq

        files[str(p.relative_to(repo))] = {"sha256": _hash(p), "bytes": p.stat().st_size,
                                           "rows": pq.ParquetFile(p).metadata.num_rows}
    return {
        "benchmark": benchmark,
        "repo_id": REPOS[benchmark],
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "exported_at": manifest["created_at"],
        "generator": manifest["generator"],
        "packaged_at_commit": git_state().get("sha"),
        "git": {"sha": manifest["git"].get("sha"), "short_sha": manifest["git"].get("short_sha"),
                "dirty": manifest["git"].get("dirty")},
        "seed": manifest["seed"],
        "rule": manifest["rule"],
        "data_files": manifest["data_files"],
        "command": command,
        "export_command": manifest.get("command"),
        "export_manifest_sha256": _hash(export_manifest),
        "elements": len(records),
        "totals": {k: sum(r[k] for r in records)
                   for k in ("test_questions", "test_rows", "few_shot_questions", "few_shot_rows")},
        "not_ok": {e: r["status"] for e, r in manifest["elements"].items()
                   if r["benchmark"] == benchmark and r["status"] != "ok"},
        "files": files,
    }


# --------------------------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------------------------
# Files that do not affect generated rows: an export from an earlier commit can be packaged if only these changed.
PACKAGING_ONLY = ("src/autosteer/release/package.py", "src/autosteer/cli.py", "src/autosteer/web/", "tests/",
                  "README.md", "docs/")


def generation_changes(export_sha: str | None, head_sha: str | None) -> list[str]:
    """Files changed between the export's commit and HEAD that could change generated rows (['?'] if unknown)."""
    import subprocess

    from autosteer.release.rows import PACKAGE_DIR

    if not export_sha or not head_sha:
        return ["?"]
    if export_sha == head_sha:
        return []
    root = str(PACKAGE_DIR.parent.parent)
    anc = subprocess.run(["git", "-C", root, "merge-base", "--is-ancestor", export_sha, head_sha])
    if anc.returncode != 0:
        return [f"{export_sha[:7]} is not an ancestor of HEAD"]
    diff = subprocess.run(["git", "-C", root, "diff", "--name-only", export_sha, head_sha],
                          capture_output=True, text=True).stdout.split()
    return [f for f in diff if not f.startswith(PACKAGING_ONLY)]


def build(args: argparse.Namespace) -> int:
    from autosteer.release import export
    from autosteer.release.rows import git_state

    out = Path(args.out).expanduser()
    git = git_state()
    if git.get("dirty") and not args.allow_dirty:
        raise SystemExit("refusing to build a release from a checkout with uncommitted changes to tracked files "
                         f"({', '.join(git['dirty_files'][:5])}); commit them or pass --allow-dirty")
    command = " ".join(["autosteer release build", *map(str, getattr(args, "argv", []))])
    if args.from_export:
        export_dir = Path(args.from_export).expanduser()
    else:
        export_dir = out / "export"
        ns = argparse.Namespace(out=str(export_dir), seed=args.seed, elements=args.elements, per_cell=args.per_cell,
                                floor=args.floor, max_attempts_per_question=args.max_attempts_per_question,
                                few_shot_per_type=args.few_shot_per_type, max_held_out=args.max_held_out,
                                workers=args.workers, allow_dirty=args.allow_dirty,
                                allow_unverified_data=args.allow_unverified_data, argv=["(release build)"])
        rc = export.run(ns)
        if rc != 0 and not args.allow_failures:
            raise SystemExit(f"export failed (exit {rc}); see {export_dir / 'manifest.json'}")
    manifest_path = export_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    changed = generation_changes(manifest["git"].get("sha"), git.get("sha"))
    if changed and not args.allow_dirty:
        raise SystemExit(f"{export_dir} was exported at {manifest['git'].get('short_sha')}; generation code changed "
                         f"since ({', '.join(changed[:5])}); export again or pass --allow-dirty")
    if manifest["rule"].get("few_shot", {}).get("per_type", 0) <= 0:
        raise SystemExit(f"{export_dir} has no few_shot split (exported with --few-shot-per-type 0)")
    failed = sorted(e for e, r in manifest["elements"].items() if r["status"] == "fail")
    if failed and not args.allow_failures:
        raise SystemExit(f"export has failed elements {failed}; fix them or pass --allow-failures")

    summary = {}
    for bench in ("STEER", "STEER-ME"):
        repo = out / REPO_DIRS[bench]
        if repo.exists():
            shutil.rmtree(repo)
        (repo / "data").mkdir(parents=True)
        for e, r in sorted(manifest["elements"].items()):
            if r["benchmark"] != bench or not r.get("file"):
                continue
            (repo / "data" / e).mkdir()
            shutil.copy2(export_dir / r["file"], repo / "data" / e / "test.parquet")
            if r.get("few_shot", {}).get("file"):
                shutil.copy2(export_dir / r["few_shot"]["file"], repo / "data" / e / "few_shot.parquet")
        records = element_records(bench, manifest, repo / "data")
        write_elements(repo, records)
        (repo / "ELEMENTS.md").write_text(elements_md(bench, records), encoding="utf-8")
        prov = provenance(bench, manifest, repo, records, command, manifest_path)
        (repo / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
        (repo / "README.md").write_text(card(bench, records, prov), encoding="utf-8")
        summary[bench] = {"dir": str(repo), "elements": len(records), **prov["totals"],
                          "bytes": sum(f["bytes"] for f in prov["files"].values())}
    for bench, s in summary.items():
        print(f"{bench}: {s['elements']} elements, test {s['test_questions']} questions / {s['test_rows']} rows, "
              f"few_shot {s['few_shot_questions']} / {s['few_shot_rows']}, {s['bytes'] / 1e6:.1f} MB -> {s['dir']}")
    print("nothing uploaded; check with: autosteer release check", out)
    return 0


# --------------------------------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------------------------------
def _configs(readme: Path) -> list[dict]:
    text = readme.read_text(encoding="utf-8")
    front = text.split("---\n", 2)[1]
    try:
        import yaml
    except ImportError:  # fall back to the layout this module writes
        configs, cur = [], None
        for line in front.splitlines():
            if line.startswith("- config_name: "):
                cur = {"config_name": line.split(": ", 1)[1], "data_files": []}
                configs.append(cur)
            elif line.startswith("  - split: ") and cur is not None:
                cur["data_files"].append({"split": line.split(": ", 1)[1]})
            elif line.startswith("    path: ") and cur is not None:
                cur["data_files"][-1]["path"] = line.split(": ", 1)[1]
        return configs
    return yaml.safe_load(front)["configs"]


def check(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    from autosteer.release.export import select_elements
    from autosteer.release.rows import element_info
    from autosteer.release.schema import SCHEMA, validate
    from autosteer.sampler import ElementSampler

    out = Path(args.dir).expanduser()
    problems: list[str] = []
    warnings: list[str] = []
    counts = json.loads(COUNTS_JSON.read_text())["elements"]
    chosen, _ = select_elements()
    sampler = ElementSampler()
    report = {}
    try:
        import datasets

        datasets.disable_progress_bars()
        datasets.logging.set_verbosity_error()
        have_datasets = True
    except ImportError:
        have_datasets = False
        warnings.append("datasets not importable: config loading not checked "
                        "(run with `uv run --with datasets autosteer release check ...`)")
    seen_elements: dict[str, str] = {}
    for bench in ("STEER", "STEER-ME"):
        repo = out / REPO_DIRS[bench]
        for f in ("README.md", "ELEMENTS.md", "elements.csv", "elements.jsonl", "provenance.json"):
            if not (repo / f).exists():
                problems.append(f"{bench}: missing {f}")
        rows_by = {"test": 0, "few_shot": 0}
        elements = sorted(p.name for p in (repo / "data").iterdir() if p.is_dir())
        listed = [json.loads(x)["element"] for x in (repo / "elements.jsonl").read_text().splitlines()]
        if sorted(listed) != elements:
            problems.append(f"{bench}: elements.jsonl and data/ disagree")
        for e in elements:
            seen_elements[e] = bench
            if element_info(e)["benchmark"] != bench:
                problems.append(f"{e}: in {bench} but elements.json says {element_info(e)['benchmark']}")
            tabs = {}
            for split in SPLITS:
                p = repo / "data" / e / f"{split}.parquet"
                if not p.exists():
                    if split == "test":
                        problems.append(f"{e}: no test.parquet")
                    continue
                t = pq.read_table(p)
                if not t.schema.equals(SCHEMA, check_metadata=False):
                    problems.append(f"{e}/{split}: schema differs from autosteer.release.schema.SCHEMA")
                errs = validate(t)
                problems += [f"{e}/{split}: {x}" for x in errs]
                rows = t.to_pylist()
                for r in rows:
                    if r["grading"] == "consistency" and not RULE_RE.match(r["consistency_rule"] or ""):
                        problems.append(f"{e}/{split}: malformed consistency_rule {r['consistency_rule']!r}")
                        break
                tabs[split] = rows
                rows_by[split] += len(rows)
            test, few = tabs.get("test", []), tabs.get("few_shot", [])
            nq = len({r["base_id"] for r in test})
            want = counts.get(e)
            if want is None:
                warnings.append(f"{e}: not in dataset_counts.json")
            elif (want["questions"], want["rows"]) != (nq, len(test)):
                warnings.append(f"{e}: test {nq} questions / {len(test)} rows, dataset_counts.json "
                                f"{want['questions']} / {want['rows']}")
            # few_shot vs test: template ids, template texts, questions
            t_ids, f_ids = {r["template_id"] for r in test}, {r["template_id"] for r in few}
            if t_ids & f_ids:
                problems.append(f"{e}: {len(t_ids & f_ids)} templates in both test and few_shot")
            if {r["base_id"] for r in test} & {r["base_id"] for r in few}:
                problems.append(f"{e}: a few_shot question is also in test")
            # texts with numbers in them must differ (texts without, e.g. "Which would you choose?", repeat)
            numbered = {r["question_text"] for r in test if any(ch.isdigit() for ch in r["question_text"])}
            if numbered & {r["question_text"] for r in few}:
                problems.append(f"{e}: a few_shot question text is also a test question text")
            if few:
                parts: dict[str, list] = {}
                for pool in ("questions", "examples"):
                    if any(f"/{pool}/" in x for x in t_ids | f_ids):
                        q = sampler._pool_frames(sampler._load_template(e), pool)[0]
                        for qid, sub, text in q.sort_values(["question_id", "sub_id"])[
                                ["question_id", "sub_id", "question_text"]].itertuples(index=False):
                            parts.setdefault(f"{e}/{pool}/{int(qid)}", []).append(str(text))
                tt = {tuple(parts[i]) for i in t_ids}
                ft = {tuple(parts[i]) for i in f_ids}
                if tt & ft:
                    problems.append(f"{e}: {len(tt & ft)} template texts used by both test and few_shot")
                shared = {x for s in tt for x in s} & {x for s in ft for x in s}
                if shared:  # e.g. a follow-up part worded the same in several templates
                    warnings.append(f"{e}: {len(shared)} part texts of different templates occur in both splits")
        report[bench] = {"elements": len(elements), **{f"{k}_rows": v for k, v in rows_by.items()},
                         "bytes": sum(p.stat().st_size for p in repo.rglob("*") if p.is_file())}
        # configs
        configs = _configs(repo / "README.md")
        names = {c["config_name"] for c in configs}
        if names != set(elements) | {"default"}:
            problems.append(f"{bench}: configs and elements disagree: {sorted(names ^ (set(elements) | {'default'}))}")
        for c in configs:
            for df in c["data_files"]:
                if not glob.glob(str(repo / df["path"])):
                    problems.append(f"{bench}/{c['config_name']}: {df['path']} matches no file")
        if have_datasets:
            from datasets import load_dataset

            for c in configs:
                files = {df["split"]: sorted(glob.glob(str(repo / df["path"]))) for df in c["data_files"]}
                try:
                    ds = load_dataset("parquet", data_files=files)
                except Exception as ex:  # noqa: BLE001
                    problems.append(f"{bench}/{c['config_name']}: load_dataset failed: {type(ex).__name__}: {ex}")
                    continue
                for split, fl in files.items():
                    want = sum(pq.ParquetFile(f).metadata.num_rows for f in fl)
                    if ds[split].num_rows != want:
                        problems.append(f"{bench}/{c['config_name']}/{split}: {ds[split].num_rows} rows, files {want}")
            report[bench]["configs_loaded"] = len(configs)
    missing = sorted(set(chosen) - set(seen_elements))
    extra = sorted(set(seen_elements) - set(chosen))
    if missing and not getattr(args, "partial", False):
        problems.append(f"released elements missing: {missing}")
    if extra:
        problems.append(f"elements that are not released (held or unlisted): {extra}")
    print(json.dumps({"report": report, "warnings": warnings, "problems": problems}, indent=1))
    return 1 if problems else 0


# --------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------
def add_arguments(ap: argparse.ArgumentParser) -> None:
    sub = ap.add_subparsers(dest="release_command", required=True)
    b = sub.add_parser("build", help="export both benchmarks and lay out the Hugging Face dataset repos locally")
    b.add_argument("--out", required=True, help="local output directory (nothing is uploaded)")
    b.add_argument("--from-export", default="", help="package this existing export instead of exporting")
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--elements", nargs="*", help="subset of elements (testing only; a release has all)")
    b.add_argument("--per-cell", type=int, default=20)
    b.add_argument("--floor", type=int, default=200)
    b.add_argument("--few-shot-per-type", type=int, default=5)
    b.add_argument("--max-held-out", type=int, default=5)
    b.add_argument("--max-attempts-per-question", type=int, default=20)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--allow-dirty", action="store_true", help="build from a checkout with uncommitted changes")
    b.add_argument("--allow-unverified-data", action="store_true")
    b.add_argument("--allow-failures", action="store_true", help="package although some elements failed")
    b.set_defaults(release_func=build)
    c = sub.add_parser("check", help="validate a built release directory")
    c.add_argument("dir")
    c.add_argument("--partial", action="store_true", help="a build of some elements: skip the completeness check")
    c.set_defaults(release_func=check)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="autosteer release", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    args = ap.parse_args(argv)
    args.argv = list(argv if argv is not None else sys.argv[1:])
    return args.release_func(args)


if __name__ == "__main__":
    raise SystemExit(main())
