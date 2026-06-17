"""
rf_substrate_classifier.py
--------------------------
Train Random Forest substrate classifiers for NRPS A-domains.

Models trained:
  bacterial kingdom (train on bacterial data only):
    - 8angstrom, stachelhaus, ig15b
    - ig15b_union_8angstrom, ig15b_intersect_8angstrom
    - ig15b_union_stachelhaus, ig15b_intersect_stachelhaus

  fungal kingdom (train on all kingdoms' data):
    - 8angstrom, stachelhaus, ig13f
    - ig13f_union_8angstrom, ig13f_intersect_8angstrom
    - ig13f_union_stachelhaus, ig13f_intersect_stachelhaus

Substrate labels are restricted to benchmark_substrates_{kingdom}.txt.
Train rows are exploded (one row per substrate) then filtered to included substrates.
Benchmark evaluation uses 1-hit accuracy, denominator = all benchmark rows.

Output: model/rf_substrate_{residue_set}_{kingdom}.pkl
"""

from __future__ import annotations

import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent.parent
MSA_TSV   = ROOT / "data" / "sequence" / "dataset_msa.tsv"
SPLIT_TSV = ROOT / "data" / "sequence" / "dataset.tsv"
RES_DIR   = ROOT / "data" / "residue"
SUB_DIR   = ROOT / "data" / "substrate"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ── Model configurations ───────────────────────────────────────────────────────
# (residue_set, kingdom)
CONFIGS = [
    # bacterial
    ("8angstrom",                    "bacterial"),
    ("stachelhaus",                  "bacterial"),
    ("ig15b",                        "bacterial"),
    ("ig15b_union_8angstrom",        "bacterial"),
    ("ig15b_intersect_8angstrom",    "bacterial"),
    ("ig15b_union_stachelhaus",      "bacterial"),
    ("ig15b_intersect_stachelhaus",  "bacterial"),
    # fungal
    ("8angstrom",                    "fungal"),
    ("stachelhaus",                  "fungal"),
    ("ig13f",                        "fungal"),
    ("ig13f_union_8angstrom",        "fungal"),
    ("ig13f_intersect_8angstrom",    "fungal"),
    ("ig13f_union_stachelhaus",      "fungal"),
    ("ig13f_intersect_stachelhaus",  "fungal"),
]

BENCHMARK_SUB_FILES = {
    "bacterial": SUB_DIR / "benchmark_substrates_bacterial.txt",
    "fungal":    SUB_DIR / "benchmark_substrates_fungal.txt",
}

RESIDUE_FILES = {name: RES_DIR / f"{name}.txt" for name, _ in CONFIGS}

DISPLAY_NAMES = {
    "8angstrom":                   "8Å",
    "stachelhaus":                 "Stachelhaus",
    "ig15b":                       "IG15B",
    "ig15b_union_8angstrom":       "IG15B ∪ 8Å",
    "ig15b_intersect_8angstrom":   "IG15B ∩ 8Å",
    "ig15b_union_stachelhaus":     "IG15B ∪ Stachelhaus",
    "ig15b_intersect_stachelhaus": "IG15B ∩ Stachelhaus",
    "ig13f":                       "IG13F",
    "ig13f_union_8angstrom":       "IG13F ∪ 8Å",
    "ig13f_intersect_8angstrom":   "IG13F ∩ 8Å",
    "ig13f_union_stachelhaus":     "IG13F ∪ Stachelhaus",
    "ig13f_intersect_stachelhaus": "IG13F ∩ Stachelhaus",
}

RESULTS_TSV = ROOT / "data" / "results_rf_substrate.tsv"

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_positions(residue_set: str) -> list[int]:
    path = RESIDUE_FILES[residue_set]
    return [int(l.strip()) for l in path.read_text().splitlines() if l.strip().isdigit()]


def load_benchmark_substrates(kingdom: str) -> set[str]:
    path = BENCHMARK_SUB_FILES[kingdom]
    return {l.strip() for l in path.read_text().splitlines() if l.strip()}


def load_data(positions: list[int]) -> pd.DataFrame:
    feat_cols = [f"grsA_{p}" for p in positions]
    msa = pd.read_csv(MSA_TSV, sep="\t", usecols=["domain_id", "type"] + feat_cols, dtype=str)
    tsv = pd.read_csv(SPLIT_TSV, sep="\t",
                      usecols=["domain_id", "substrates", "split"], dtype=str)
    tsv["subs"] = tsv["substrates"].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )
    df = msa.merge(tsv[["domain_id", "subs", "split"]], on="domain_id", how="inner")
    return df, feat_cols


def build_train_df(df: pd.DataFrame, feat_cols: list[str],
                   kingdom: str, included: set[str]) -> pd.DataFrame:
    pool = df[df["split"] == "train"].copy()
    if kingdom == "bacterial":
        pool = pool[pool["type"] == "bacterial"]
    # fungal: keep all kingdoms

    # Step 1: explode multi-label rows into one row per substrate
    # Step 2: filter out substrates not in included
    return (
        pool.explode("subs")
        .rename(columns={"subs": "substrate"})
        .loc[lambda d: d["substrate"].isin(included),
             ["domain_id", "substrate"] + feat_cols]
        .reset_index(drop=True)
    )


def build_benchmark_df(df: pd.DataFrame, feat_cols: list[str],
                       kingdom: str, included: set[str]) -> pd.DataFrame:
    pool = df[(df["split"] == "benchmark") & (df["type"] == kingdom)].copy()

    rows = []
    for _, row in pool.iterrows():
        subs = [s for s in row["subs"] if s in included]
        if len(subs) >= 1:
            rows.append({"domain_id": row["domain_id"], "true_subs": subs,
                         **{c: row[c] for c in feat_cols}})
    return pd.DataFrame(rows)


# ── Train one model ────────────────────────────────────────────────────────────

def train_model(residue_set: str, kingdom: str) -> dict | None:
    print(f"\n{'='*60}")
    print(f"  Residue set : {residue_set}   Kingdom : {kingdom}")
    print(f"{'='*60}")

    positions  = load_positions(residue_set)
    included   = load_benchmark_substrates(kingdom)
    feat_cols  = [f"grsA_{p}" for p in positions]
    print(f"  Positions   : {len(positions)}  |  Included substrates : {len(included)}")

    df, feat_cols = load_data(positions)

    train_df = build_train_df(df, feat_cols, kingdom, included)
    bench_df = build_benchmark_df(df, feat_cols, kingdom, included)
    print(f"  Train rows  : {len(train_df)}  |  Benchmark rows : {len(bench_df)}")
    # print(f"  Train class dist:\n{train_df['substrate'].value_counts().to_string()}")

    if train_df.empty or bench_df.empty:
        print("  WARNING: empty split — skipping.")
        return None

    # ── Encode ────────────────────────────────────────────────────────────────
    AA_VOCAB = sorted({aa for col in feat_cols for aa in train_df[col].dropna().unique()})
    ohe = OneHotEncoder(
        categories=[AA_VOCAB] * len(feat_cols),
        handle_unknown="ignore",
        sparse_output=False,
    )
    le = LabelEncoder()

    X_train = ohe.fit_transform(train_df[feat_cols].fillna("-"))
    y_train = le.fit_transform(train_df["substrate"])

    X_bench = ohe.transform(bench_df[feat_cols].fillna("-"))

    # ── Train ─────────────────────────────────────────────────────────────────
    #   bacterial: n_estimators=500, max_features=0.3, min_samples_leaf=2
    #   fungal:    n_estimators=500, max_features=0.3, min_samples_leaf=3
    clf = RandomForestClassifier(
        n_estimators=500,
        max_features=0.3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate: train accuracy ───────────────────────────────────────────────
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    print(f"\n  Train accuracy     : {train_acc:.4f}")

    # ── Evaluate: benchmark (1-hit, known-only denominator) ───────────────────
    train_classes = set(le.classes_)
    known_mask = bench_df["true_subs"].apply(
        lambda subs: any(s in train_classes for s in subs)
    )
    bench_known = bench_df[known_mask].reset_index(drop=True)
    X_bench_known = X_bench[known_mask.values]

    bench_preds = le.inverse_transform(clf.predict(X_bench_known))
    hits = sum(
        pred in true_subs
        for pred, true_subs in zip(bench_preds, bench_known["true_subs"])
    )
    bench_acc = hits / len(bench_known)
    print(f"  Benchmark accuracy : {bench_acc:.4f}  "
          f"({hits}/{len(bench_known)}, total={len(bench_df)})")

    # ── Benchmark classification report (single-label known rows only) ────────
    single_mask = bench_known["true_subs"].apply(len) == 1
    if single_mask.sum() > 0:
        y_true_single = le.transform(
            [ts[0] for ts in bench_known.loc[single_mask, "true_subs"]]
        )
        y_pred_single = clf.predict(X_bench_known[single_mask.values])
        print(f"\n  Classification report (single-label known benchmark rows, "
              f"n={single_mask.sum()}):")
        present_labels = sorted(set(y_true_single) | set(y_pred_single))
        # print(classification_report(
        #     y_true_single, y_pred_single,
        #     labels=present_labels,
        #     target_names=le.classes_[present_labels],
        #     zero_division=0,
        # ))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = MODEL_DIR / f"rf_substrate_{residue_set}_{kingdom}.pkl"
    joblib.dump({
        "model":        clf,
        "ohe":          ohe,
        "label_encoder": le,
        "positions":    positions,
        "residue_set":  residue_set,
        "kingdom":      kingdom,
        "included_substrates": sorted(included),
    }, out_path)
    print(f"\n  Saved → {out_path}")
    return {
        "residue_set":       DISPLAY_NAMES.get(residue_set, residue_set),
        "kingdom":           kingdom,
        "n_positions":       len(positions),
        "train_acc":         round(train_acc, 4),
        "benchmark_acc":     round(bench_acc, 4),
        "benchmark_hits":    hits,
        "benchmark_known":   len(bench_known),
        "benchmark_total":   len(bench_df),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rows = []
    for residue_set, kingdom in CONFIGS:
        result = train_model(residue_set, kingdom)
        if result is not None:
            rows.append(result)

    results_df = pd.DataFrame(rows, columns=[
        "residue_set", "kingdom", "n_positions",
        "train_acc", "benchmark_acc",
        "benchmark_hits", "benchmark_known", "benchmark_total",
    ])
    results_df.to_csv(RESULTS_TSV, sep="\t", index=False)
    print(f"\nAll models trained. Results saved → {RESULTS_TSV}")
    print(results_df.to_string(index=False))
