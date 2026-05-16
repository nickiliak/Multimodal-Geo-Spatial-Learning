# Tomorrow: paste HPC output + this prompt to Claude

## Step 1 — run on HPC, paste all output

```bash
head -5 ~/MML_Data/train/mml_train.csv
head -5 ~/MML_Data/train/mml_train_ground.csv
head -5 ~/MML_Data/train/mml_train_satellite.csv
head -5 ~/MML_Data/train/mml_train_text.csv

head -5 ~/MML_Data/index/mml_index_ground.csv
head -5 ~/MML_Data/index/mml_index_satellite.csv

head -5 ~/MML_Data/query/mml_query.csv
head -5 ~/MML_Data/query/mml_query_ground.csv
head -5 ~/MML_Data/query/mml_query_satellite.csv
head -5 ~/MML_Data/query/mml_query_text.csv

head -5 ~/MML_Data/additional_info/mmlandmarks.csv

wc -l ~/MML_Data/train/mml_train.csv
wc -l ~/MML_Data/train/mml_train_ground.csv
wc -l ~/MML_Data/train/mml_train_satellite.csv
wc -l ~/MML_Data/train/mml_train_text.csv
wc -l ~/MML_Data/index/mml_index_ground.csv
wc -l ~/MML_Data/index/mml_index_satellite.csv
wc -l ~/MML_Data/query/mml_query.csv
wc -l ~/MML_Data/query/mml_query_ground.csv
```

---

## Step 2 — send this prompt to Claude (after pasting the output above)

---

> Using the CSV output above, create a Jupyter notebook at `notebooks/02_eda.ipynb` for exploratory data analysis of the MMLandmarks dataset.
>
> **Context:**
> - Dataset: MMLandmarks — multimodal landmark dataset (ground images, satellite images, text descriptions)
> - Splits: `train/`, `query/`, `index/` — each with manifest CSVs and hex-sharded image folders
> - Data root: `data/MML_Data/` (symlinked on HPC to `/dtu/blackhole/02/137570/MML/`)
> - Metadata: `data/MML_Data/additional_info/mmlandmarks.csv` — columns include landmark_id, lat, lon, bounding box, category, hierarchical_category, WikipediaPage, QID
> - Downstream task: geo-localization with GeoClip baseline — so lat/lon quality and category distribution matter most
>
> **The notebook must cover:**
>
> 1. **Setup** — imports, set `DATA_ROOT = Path("../data/MML_Data")`, verify path exists
> 2. **Split sizes** — table of row counts per split and modality (use the wc -l numbers)
> 3. **Manifest schema** — load each CSV, print `.head()` and `.dtypes`, confirm join keys across modalities
> 4. **Metadata analysis** — load `mmlandmarks.csv`, null check, describe lat/lon/bbox columns
> 5. **Category distribution** — bar chart top-30 categories (log scale y-axis), print Gini coefficient or top-5 share to quantify imbalance
> 6. **Geographic distribution** — scatter plot lat/lon, color by category or density; note any outliers
> 7. **Images per landmark** — merge manifest with metadata, histogram of image count per landmark (log scale), print median/max/min
> 8. **Multimodal completeness** — for train split, check how many landmark_ids have ground AND satellite AND text; report % complete
> 9. **Bounding box precision** — histogram of bbox area `(max_lat-min_lat)*(max_lon-min_lon)`, flag landmarks with suspiciously large boxes
> 10. **Sample images** — load 9 random ground images and 9 satellite images (3x3 grid each) using PIL; derive file path from manifest columns using the hex-prefix structure `{modality}/{filename[0]}/{filename}/`
> 11. **Sample text** — print 5 random text file contents (derive path same way)
> 12. **Summary table** — markdown cell at the end with key numbers filled in
>
> Use `matplotlib` for plots, `pandas` for data, `PIL` for images. Save figures to `outputs/eda/`. Make each section a clearly labelled markdown cell. Use `pathlib.Path` throughout, no hardcoded strings.
