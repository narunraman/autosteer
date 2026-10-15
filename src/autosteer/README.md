# The autosteer package

Where things are, for working on the code. How to run autoSTEER is in the [repository README](../../README.md).

| Path | What it does |
|---|---|
| `cli.py` | The `autosteer` command: `autosteer dev`, `autosteer data verify`, ... |
| `sampler/` | Turning templates into questions: `generator.py` (`ElementSampler`), `manifest.py` (a summary of every element's templates), difficulty levels. The templates in data/ are stored corrected (scripts/migrations/). |
| `filler_tools/` | The fill functions, one `*_tools.py` per family of elements: they pick the values for a template and compute the answer. Its [README](filler_tools/README.md) explains how to write one. |
| `web/` | The site and developer mode: `app.py` (puts the app together), `routes/` (the API, and developer mode's seeds, candidates and template generation), `common.py` (what those routes share), `site.py` (guide and benchmark pages), `public.py` (the public site's switches), `devmode.py` and `review.py` (developer mode's pages and review), `content/` (the site's text), `templates/` and `static/`. |
| `release/` | The fixed datasets: rows, schema, consistency checks and the exporter. |
| `services/`, `schemas/` | Developer mode's back end: generating seed candidates and templates with a model, and storing the records. |
| `llm_client.py` | The model providers: OpenAI, Azure OpenAI, Anthropic, Gemini and OpenAI-compatible endpoints. |
| `instruction_parts.py`, `json_utils.py` | The instructions for the model, and parsing its replies. |
| `formatting.py` | Formatting filled values: rounding, placeholders, duplicate options. |
| `steer_dataset.py`, `dataset_io.py` | The `Dataset` class (question, option and instruction tables) and its Parquet format on disk. |
| `data_files.py` | The pinned list of data files (`autosteer data verify`). |
