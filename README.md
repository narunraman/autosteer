<h1>
  <img src=".github/assets/autosteer_lockup_light.svg#gh-light-mode-only" alt="autoSTEER" height="44">
  <img src=".github/assets/autosteer_lockup_dark.svg#gh-dark-mode-only" alt="autoSTEER" height="44">
</h1>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23406989.svg)](https://doi.org/10.5281/zenodo.23406989)

autoSTEER is the tool behind the <a href="https://steer-benchmark.cs.ubc.ca"><img src=".github/assets/steer_badge.svg" height="14" alt=""></a> <a href="https://steer-benchmark.cs.ubc.ca">STEER</a> benchmarks for the economic reasoning
of language models. It generates fresh questions for every element of STEER and STEER-ME, serves the benchmark
site, and has a developer mode for writing new elements.

## Run it

autoSTEER runs from a clone of this repository; the template data (about 25 MB) is in `data/`. You need
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/narunraman/autosteer.git
cd autosteer
uv run autosteer dev
```

This opens the site on your machine with developer mode at `/developer`. The steps that use a language model
(seed candidates, templates, the solve check) need a key for one provider, set in your environment before starting
(for example `export OPENAI_API_KEY=...`, or keep them in a `.env` file and start with
`uv run --env-file .env autosteer dev`) or entered on the AI setup page for the session. OpenAI, Azure OpenAI,
Anthropic, Google Gemini and any OpenAI-compatible endpoint (vLLM, a local server) are supported.

Sample questions from Python:

```python
from autosteer.sampler.generator import ElementSampler

sampler = ElementSampler()
question = sampler.sample_question("consumer_surplus", seed=1)
```

A question depends only on the element, the seed and the autoSTEER version: the same call returns the same
question in any process, whatever was drawn before or alongside it. (From 0.2.0; earlier versions drew from
process-wide generators, so their questions for a seed differ.)

The same questions are served by the <a href="https://steer-benchmark.cs.ubc.ca/reference"><img src=".github/assets/steer_badge.svg" height="14" alt=""></a> <a href="https://steer-benchmark.cs.ubc.ca/reference">API</a>, and fixed samples are on
Hugging Face as <a href="https://huggingface.co/datasets/narunraman/steer"><img src=".github/assets/hf_logo.svg" height="14" alt=""></a> <a href="https://huggingface.co/datasets/narunraman/steer"><code>narunraman/steer</code></a> and
<a href="https://huggingface.co/datasets/narunraman/steer_me"><img src=".github/assets/hf_logo.svg" height="14" alt=""></a> <a href="https://huggingface.co/datasets/narunraman/steer_me"><code>narunraman/steer_me</code></a>. To evaluate a model, use
<a href="https://github.com/narunraman/STEER#gh-light-mode-only"><img src=".github/assets/github_logo_light.svg" height="14" alt=""></a><a href="https://github.com/narunraman/STEER#gh-dark-mode-only"><img src=".github/assets/github_logo_dark.svg" height="14" alt=""></a> <a href="https://github.com/narunraman/STEER">steer-bench</a>.

## Write a new element

In developer mode, an element is built in four steps: example seeds, seed candidates (a model rewrites your seeds
for other domains and perspectives; you accept each one), templates, and review. Every answer is computed by the
element's filler in `src/autosteer/filler_tools/`; models only help with wording. The guide pages in developer mode
explain each step.

To contribute an element, open a pull request with its filler and solver test, its folders under `data/`
(`example_seeds/<id>`, `generated_seeds/<id>`, `templates/<id>`) and its review record (`data/review/<id>.json`).

## Data

`data/` holds one folder per element in `templates/`, `example_seeds/` and `generated_seeds/`, each a
`dataset.json` with one Parquet file per table. `uv run autosteer data verify` checks the folder against the list
of expected files (`src/autosteer/data_files.json`). Set `AUTOSTEER_DATA_DIR` to work in another folder.

## Tests

```bash
uv run pytest          # the default tier (a few minutes)
uv run pytest -m ""    # everything, including the slow sweeps over every element (about half an hour)
```

## License

The code is MIT-licensed (`LICENSE`). The template data in `data/` is licensed under CC BY 4.0
(`data/LICENSE.md`).

## Citation

If you use autoSTEER or its questions, please cite the STEER and STEER-ME papers:

```bibtex
@inproceedings{ramansteer,
  author    = {Narun Krishnamurthi Raman and Taylor Lundy and
               Samuel Joseph Amouyal and Yoav Levine and
               Kevin Leyton{-}Brown and Moshe Tennenholtz},
  title     = {{STEER:} {A}ssessing the {E}conomic {R}ationality of
               {L}arge {L}anguage {M}odels},
  booktitle = {Forty-first International Conference on Machine Learning,
               {ICML} 2024, Vienna, Austria, July 21-27, 2024},
  publisher = {OpenReview.net},
  year      = {2024},
  url       = {https://openreview.net/forum?id=nU1mtFDtMX}
}
```

```bibtex
@inproceedings{raman2025steerme,
  title     = {{STEER-ME}: Assessing the Microeconomic
               Reasoning of Large Language Models},
  author    = {Raman, Narun and Lundy, Taylor and
               Amin, Thiago and Perla, Jesse and
               Leyton-Brown, Kevin},
  booktitle = {Advances in Neural Information Processing
               Systems},
  volume    = {38},
  year      = {2025},
  note      = {Datasets and Benchmarks Track},
  url       = {https://arxiv.org/abs/2502.13119}
}
```

To cite autoSTEER itself (this DOI always points to the latest release; each release on [Zenodo](https://doi.org/10.5281/zenodo.23406989) has its own):

```bibtex
@software{raman_autosteer,
  author    = {Raman, Narun},
  title     = {{autoSTEER: the question generator and site for STEER and STEER-ME}},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23406989},
  url       = {https://doi.org/10.5281/zenodo.23406989}
}
```
