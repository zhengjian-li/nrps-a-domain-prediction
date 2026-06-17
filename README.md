# NRPS A-Domain Substrate Prediction

## Publication

**Title:** A data-driven rediscovery of the specificity-conferring code of adenylation domains in nonribosomal peptide synthetases

**Journal:** BioRxiv (2026)  
**DOI:** [TODO]

## Online Tool

🤗 **Huggingface Space:**  
https://huggingface.co/spaces/UdS-LSV/nrps-a-domain-prediction

---

## Layout

```
nrps-a-domain-prediction/
├── config/
│   └── training.yaml                 # Reference hyperparameters (see note below)
├── data/                             # Inputs, tracked in git (~55 MB)
│   ├── sequence/
│   │   ├── dataset.fasta             # Domain sequences
│   │   ├── dataset.tsv               # Labels: domain_id, substrates, type, source, split
│   │   ├── dataset_msa.tsv           # MSA feature table: grsA_N residue columns
│   │   ├── msa/                      # Alignment + id↔msa mapping
│   │   ├── benchmark_bacterial.fasta
│   │   └── benchmark_fungal.fasta
│   ├── residue/                      # GrsA-numbered position sets (one integer per line)
│   │   ├── stachelhaus.txt           #   Stachelhaus code (8 positions)
│   │   ├── 8angstrom.txt             #   residues within 8 Å of the substrate
│   │   ├── ig13f.txt / ig15b.txt     #   information-gain selected (fungal / bacterial)
│   │   └── ig*_{union,intersect}_*.txt   #   combinations with 8Å / Stachelhaus
│   └── substrate/                    # Substrate label lists (all / common / benchmark, per kingdom)
└── script/
    ├── featurisation/
    │   ├── featurisation_models.py   # InformationGain / MutualInformation / ChiSquared scorers
    │   └── pipeline_k_selection.py   # rank positions → CV top-k selection → benchmark eval
    └── model_training/
        └── rf_substrate_classifier.py   # train the final RF classifiers
```

> Trained `.pkl` models are not tracked in git (see `.gitignore`); they are written
> to `model/` when you run the training script.

---

## Setup

Requires Python ≥ 3.9 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

The dataset is already in the repo, so you can run featurisation and training
immediately.

---

## Data

Each domain has a row in `data/sequence/dataset.tsv`:

| column | meaning |
|--------|---------|
| `domain_id`  | unique A-domain identifier |
| `substrates` | Python-literal list of substrate names (multi-label) |
| `type`       | kingdom (`bacterial` / `fungal`) |
| `source`     | provenance of the entry |
| `split`      | `train` or `benchmark` |

`dataset_msa.tsv` carries the same `domain_id` plus one column
per aligned position (`grsA_1 … grsA_N`), each holding the single-letter amino
acid (or `-` for a gap) at that GrsA-numbered position. The two tables are joined
on `domain_id`. Residue features are **one-hot encoded** over the amino-acid
alphabet, so no false ordinal relationship is imposed.

---

## 1. Residue-position selection

`pipeline_k_selection.py` ranks positions by an information-theoretic score, then
uses cross-validation on the train split to pick how many top positions (`k`) to
keep, and finally evaluates that set on the benchmark split.

```bash
uv run python script/featurisation/pipeline_k_selection.py \
    --train-kingdom fungal --test-kingdom fungal --method ig
```

Key options (`--help` for all):

| flag | default | meaning |
|------|---------|---------|
| `--train-kingdom` | `fungal` | `bacterial` / `fungal` / `all` |
| `--test-kingdom`  | `fungal` | `bacterial` / `fungal` |
| `--method`        | all five | `ig`, `mi_avg`, `mi_max`, `chi2_avg`, `chi2_max` |
| `--min-k/--max-k/--step` | 5 / 40 / 1 | range of position counts to sweep |
| `--cv-folds`      | 5 | stratified CV folds |
| `--n-seeds`       | 5 | RF seeds averaged per evaluation point |
| `--k-criterion`   | `peak` | `peak` (max CV acc) or `knee` (elbow of the curve) |

Outputs (rankings, per-k CV curves, plots, and a summary) are written to
`data/residue/pipeline_k_selection_peak_benchmark_substrates/`. The
information-gain–selected sets shipped in `data/residue/` (`ig13f`, `ig15b`) were
produced this way.

The three scorers live in `featurisation_models.py`:

- **InformationGain** — `G(t) = H(C) − H(C|t)` for each binary `(position, amino acid)` term.
- **MutualInformation** — `I(t;c)`, reported as a class-weighted average or per-class max.
- **ChiSquared** — `χ²(t,c)` from a 2×2 contingency table, again averaged or maxed over classes.

---

## 2. Training the classifiers

`rf_substrate_classifier.py` trains one `RandomForestClassifier` per
`(residue_set, kingdom)` pair listed in its `CONFIGS` table — the Stachelhaus, 8 Å,
information-gain, and union/intersection residue sets, for both bacterial and
fungal kingdoms.

```bash
uv run python script/model_training/rf_substrate_classifier.py
```

For each config it:

1. loads residue positions from `data/residue/<set>.txt` and the benchmark
   substrate list from `data/substrate/benchmark_substrates_<kingdom>.txt`;
2. builds the train set (bacterial models train on bacterial rows; fungal models
   train on all kingdoms), exploding multi-label rows to one row per substrate;
3. one-hot encodes the residues, fits the forest, and evaluates 1-hit accuracy on
   the benchmark split;
4. saves a self-contained bundle to `model/rf_substrate_<set>_<kingdom>.pkl`
   (model + one-hot encoder + label encoder + positions + included substrates) for
   later inference.

A summary of train/benchmark accuracy per model is written to
`data/results_rf_substrate.tsv`.

---

## Method summary

- **Features:** residues at defined GrsA-numbered positions, one-hot encoded over the amino-acid alphabet.
- **Residue sets:** Stachelhaus (8), 8 Å contacts, information-gain sets (`ig13f` fungal, `ig15b` bacterial), and their unions/intersections.
- **Model:** scikit-learn `RandomForestClassifier` with `class_weight="balanced"`.
- **Evaluation:** 1-hit accuracy on the held-out `benchmark` split, restricted to substrates seen during training.

---

## License

See [LICENSE](LICENSE).
