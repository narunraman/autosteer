# autoSTEER template data

The data directory the generator reads (`AUTOSTEER_DATA_DIR`). One folder per element in each of:

- **templates/**: the templates the sampler fills to make questions (`questions` / `options` tables), and the
  few-shot pool (`example_questions` / `example_options`).
- **example_seeds/**: the hand-written example seeds an element's AI pipeline starts from.
- **generated_seeds/**: seed candidates a model wrote from the example seeds, kept with their review state
  (`review.json`).

Each folder holds `dataset.json` (settings and how to read the tables) and one Parquet file per table. They are
read and written by `autosteer.dataset_io`; `pandas.read_parquet` opens any table on its own.

The data was first assembled from the STEER-generation project (the commit is recorded in
`autosteer/data_files.json`): converted from its pickle files to Parquet, with duplicated example seeds, seed
candidates and templates removed (October 2026).

`manifest.json` is rebuilt by the sampler whenever the set of elements changes. `autosteer data verify` checks
this directory against the pinned list of files (`autosteer/data_files.json` in the package).
