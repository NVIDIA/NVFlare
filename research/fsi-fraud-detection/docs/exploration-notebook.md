# Exploration Notebook

## Location

`notebooks/data_generation_exploration.ipynb`

## Purpose

The exploration notebook provides an interactive, step-by-step walkthrough
of the entire synthetic data generation and anomaly injection pipeline. It
serves as:

- **Living documentation** for how the library components compose.
- **A development sandbox** for experimenting with configurations,
  inspecting intermediate outputs, and validating new features.
- **An onboarding tool** for understanding the data generation workflow
  without reading the full library source.

## Structure

The notebook is organised into 26 cells (12 markdown, 14 code) covering
the following sections:

### 1. Introduction (Cells 1-2)

Describes the motivation for synthetic payment data in federated learning:
lack of real payment datasets, need for controllable distributions per site,
and the architecture (library + orchestration layer).

Includes a component table summarising all `data_generation/` modules.

### 2. Setup and Configuration (Cells 3-6)

- Imports the three provider types (`FakerSyntheticDataProvider`,
  `RandomChoiceDataProvider`, `UniformDistributionDataProvider`), the
  attribute graph builders, and the dataset generator.
- Loads a per-site YAML config (e.g. `config/site1.yml`).
- Extracts distribution parameters from `anomaly_generation_config.field`
  and selects the first entry from `dataset_generation_config` for
  exploration.

### 3. Dependency Graph (Cells 7-8)

- Builds the full attribute graph via `get_per_participant_attributes()`,
  `get_payment_core_attributes()`, and `get_payment_amount_attributes()`.
- Prints the attribute count and a sample of independent vs. dependent
  attributes.

### 4. Provider Mapping (Cells 9-10)

- Creates provider instances and maps each attribute to its required
  provider type using `typing.get_type_hints()`.
- Prints a summary of provider-to-attribute assignments.

### 5. Topological Sort (Cells 11-12)

- Runs `topological_sort()` on the graph.
- Displays the sorted order to verify dependency resolution.

### 6. Data Generation (Cells 13-14)

- Calls `generate()` with the sorted graph, providers, and configured row
  count.
- Inspects the resulting DataFrame shape and head.

### 7. Row Shuffle (Cells 15-16)

- Demonstrates the shuffle-and-reset pattern:
  `df.sample(frac=1, random_state=SEED).reset_index(drop=True)`.
- Compares row ordering before and after.

### 8. Static Data (Cells 17-19)

- Loads country-currency mappings and exchange rates via
  `country_static_data.load_static_data()`.
- Displays the static reference tables.

### 9. Anomaly Generation (Cells 20-24)

- Explains the four anomaly types (tower mismatch, young account + high
  amount, stale activity, high event count).
- Builds `Type1Config` and `Type2Config` from site fields.
- Calls `add_fraud_columns()` then `inject_all()` with the site's
  `fraud_insertion_rule_stack`.
- Applies `apply_fraud_with_probability()` for label noise.
- Prints fraud row counts before and after thinning.
- Displays a DataFrame slice showing anomalous feature values alongside
  fraud flags.

### 10. Summary (Cell 25)

Recaps the six stages completed and references the `main.py` CLI tool for
bulk generation across all sites.

## How to Run

```bash
# Start JupyterLab
uv run jupyter lab

# Open notebooks/data_generation_exploration.ipynb
# Execute cells sequentially (Shift+Enter)
```

The notebook requires the same dependencies as the main project. Static
data (exchange rates) will be fetched on first run and cached to
`~/.cache/fsi_static`.

## Relationship to `main.py`

The notebook exercises the same library functions that `main.py` invokes.
The key differences are:

| Aspect        | Notebook                                | `main.py`                              |
| ------------- | --------------------------------------- | -------------------------------------- |
| Scope         | Single site, first dataset config entry | All sites, all dataset config entries  |
| Output        | In-memory DataFrames, inline display    | CSV files on disk                      |
| Iteration     | Manual cell-by-cell execution           | Automated loop over sites and configs  |
| Configuration | Loaded once, explored interactively     | Loaded per-site in the generation loop |

Both paths produce functionally equivalent output for the same
configuration and seed.
