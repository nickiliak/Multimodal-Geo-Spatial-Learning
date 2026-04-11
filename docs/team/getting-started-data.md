# Getting Started with the Data

> **HPC only.** The dataset lives on DTU HPC storage. This setup will not work on a local machine.

## Prerequisites

- SSH access to DTU HPC
- Repo cloned on HPC
- `uv` installed (used to run Jupyter)

## Setup

From the repo root, run:

```bash
bash scripts/setup_data.sh
```

## What the script does

### 1. Validates the HPC data path

Checks that `/dtu/blackhole/02/137570/MML` exists. If not (i.e. you're running off HPC), it exits with an error:

```
ERROR: /dtu/blackhole/02/137570/MML not found. Run this on HPC.
```

### 2. Creates a symlink

Creates `data/MML_Data` pointing to the HPC data directory:

```
data/MML_Data -> /dtu/blackhole/02/137570/MML
```

If the symlink already exists, the script skips this step and confirms.

### 3. Optionally re-runs the EDA notebook (optional)

Prompts whether to execute `notebooks/team/02_eda.ipynb`:

```
EDA notebook already executed. Re-run? [y/N]
```

Default is **No** — press Enter to skip. Enter `y` to re-execute the notebook in-place using `uv run jupyter nbconvert`.

## Accessing the data in code

After setup, reference data via the symlink:

```python
data_path = "data/MML_Data"
```

The symlink resolves to the HPC path transparently.

## Note on local development

The symlink target (`/dtu/blackhole/02/137570/MML`) does not exist outside of DTU HPC. Any code that reads from `data/MML_Data` will fail locally. Keep data-loading code HPC-only or guard it with a path existence check.
