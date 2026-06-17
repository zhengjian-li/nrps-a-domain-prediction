from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "script" / "featurisation"))

from featurisation_models import InformationGain, MutualInformation, ChiSquared

# ── Paths ─────────────────────────────────────────────────────────────────────

MSA_TSV   = ROOT / "data" / "sequence" / "dataset_msa.tsv"
SPLIT_TSV = ROOT / "data" / "sequence" / "dataset.tsv"
SUB_DIR   = ROOT / "data" / "substrate"
OUT_DIR   = ROOT / "data" / "residue" / "pipeline_k_selection_peak_benchmark_substrates"

BENCHMARK_SUB_FILES = {
    "bacterial": SUB_DIR / "benchmark_substrates_bacterial.txt",
    "fungal":    SUB_DIR / "benchmark_substrates_fungal.txt",
}

# BENCHMARK_SUB_FILES = {
#     "bacterial": SUB_DIR / "common_substrates_bacterial.txt",
#     "fungal":    SUB_DIR / "common_substrates_fungal.txt",
# }

VALID_METHODS = ["ig", "mi_avg", "mi_max", "chi2_avg", "chi2_max"]


# ── Step 1: Data loading ───────────────────────────────────────────────────────

def load_benchmark_substrates(test_kingdom: str) -> set[str]:
    path = BENCHMARK_SUB_FILES[test_kingdom]
    return {l.strip() for l in path.read_text().splitlines() if l.strip()}


def load_splits(train_kingdom: str, test_kingdom: str, included: set[str]
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (train_df, benchmark_df).

    Both splits:
      - substrates restricted to `included` (benchmark substrates of test_kingdom)

    train_df:     split == "train", single-label only (exactly one matching substrate)
                  train_kingdom == "bacterial" → type == "bacterial" only
                  train_kingdom == "fungal"    → type == "fungal" only
                  train_kingdom == "all"       → all kingdoms
    benchmark_df: split == "benchmark", type == test_kingdom
                  multi-label rows are kept; true_subs stores the full label list.
                  Evaluation uses 1-hit accuracy (prediction in true_subs).
    """
    msa  = pd.read_csv(MSA_TSV, sep="\t", dtype=str)
    tsv  = pd.read_csv(SPLIT_TSV, sep="\t",
                       usecols=["domain_id", "substrates", "split", "type"],
                       dtype=str)
    tsv["subs"] = tsv["substrates"].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )
    # MSA also has a 'type' column; drop it before merge to avoid _x/_y suffix
    msa_merge = msa.drop(columns=["type"], errors="ignore")
    df = msa_merge.merge(tsv[["domain_id", "subs", "split", "type"]],
                         on="domain_id", how="inner")

    pos_cols = [c for c in msa.columns if c.startswith("grsA_")]

    # ── train ─────────────────────────────────────────────────────────────────
    tr_pool = df[df["split"] == "train"].copy()
    if train_kingdom in ("bacterial", "fungal"):
        tr_pool = tr_pool[tr_pool["type"] == train_kingdom]
    # all: keep every kingdom

    # Step 1: explode multi-label rows into one row per substrate
    # Step 2: filter out substrates not in included
    train_df = (
        tr_pool.explode("subs")
        .rename(columns={"subs": "substrate"})
        .loc[lambda d: d["substrate"].isin(included),
             ["domain_id", "substrate"] + pos_cols]
        .reset_index(drop=True)
    )

    # ── benchmark ─────────────────────────────────────────────────────────────
    bm_pool = df[(df["split"] == "benchmark") & (df["type"] == test_kingdom)].copy()

    bm_rows = []
    for _, row in bm_pool.iterrows():
        subs = [s for s in row["subs"] if s in included]
        if len(subs) >= 1:
            bm_rows.append({"domain_id": row["domain_id"],
                             "true_subs": subs,       # list; may be multi-label
                             **{c: row[c] for c in pos_cols}})
    benchmark_df = pd.DataFrame(bm_rows)

    return train_df, benchmark_df


# ── Step 2: Position ranking on train ─────────────────────────────────────────

def _to_ranked_positions(scores: dict[tuple[str, str], float]) -> list[int]:
    """
    Collapse per-(position, aa) scores to per-position scores (take max over aa)
    and return positions sorted best → worst.
    """
    pos_score: dict[str, float] = {}
    for (pos, aa), v in scores.items():
        if pos not in pos_score or v > pos_score[pos]:
            pos_score[pos] = v
    ranked = sorted(pos_score, key=lambda p: pos_score[p], reverse=True)
    return [int(p) for p in ranked]


def compute_all_rankings(train_df: pd.DataFrame,
                         substrates: list[str],
                         methods: list[str]) -> dict[str, list[int]]:
    """
    Run only the requested scoring methods on train_df.
    Returns {method_name: [position, ...]} sorted best → worst.
    """
    # Rename grsA_N → N for featurisation_models
    pos_cols = [c for c in train_df.columns if c.startswith("grsA_")]
    feat_df  = train_df.rename(columns={c: c.split("_")[1] for c in pos_cols})
    feat_df  = pd.concat([feat_df, pd.get_dummies(feat_df["substrate"])], axis=1)

    need_ig   = "ig"       in methods
    need_mi   = any(m in methods for m in ("mi_avg", "mi_max"))
    need_chi2 = any(m in methods for m in ("chi2_avg", "chi2_max"))

    raw: dict[str, dict] = {}
    if need_ig:
        ig_model = InformationGain(feat_df, substrates)
        raw["ig"] = ig_model.get_all_information_gain()
    if need_mi:
        mi_model = MutualInformation(feat_df, substrates)
        if "mi_avg" in methods:
            raw["mi_avg"] = mi_model.get_mi_avg()
        if "mi_max" in methods:
            raw["mi_max"] = mi_model.get_mi_max()
    if need_chi2:
        chi_model = ChiSquared(feat_df, substrates)
        if "chi2_avg" in methods:
            raw["chi2_avg"] = chi_model.get_chi2_avg()
        if "chi2_max" in methods:
            raw["chi2_max"] = chi_model.get_chi2_max()

    return {name: _to_ranked_positions(scores) for name, scores in raw.items()}


def save_rankings(rankings: dict[str, list[int]], run_tag: str,
                  out_dir: Path) -> None:
    for name, positions in rankings.items():
        rows = [{"rank": i + 1, "position": p} for i, p in enumerate(positions)]
        pd.DataFrame(rows).to_csv(
            out_dir / f"ranking_{name}_{run_tag}.csv", index=False
        )


# ── Step 3: CV to select k ────────────────────────────────────────────────────

def _build_xy(df: pd.DataFrame,
              positions: list[int],
              le: LabelEncoder | None = None,
              ohe: OneHotEncoder | None = None,
              fit: bool = True):
    """
    Build (X, y, le, ohe) from a DataFrame.
    If fit=True, fit le and ohe on df; otherwise transform only.
    """
    feat_cols = [f"grsA_{p}" for p in positions]
    X_raw = df[feat_cols].fillna("-").astype(str)

    if fit:
        le  = LabelEncoder().fit(df["substrate"])
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X   = ohe.fit_transform(X_raw)
    else:
        X = ohe.transform(X_raw)

    y = le.transform(df["substrate"])
    return X, y, le, ohe


def cv_accuracy(train_df: pd.DataFrame,
                positions: list[int],
                n_folds: int,
                seeds: list[int],
                n_estimators: int,
                rf_n_jobs: int = 1) -> float:
    """
    Stratified k-fold CV accuracy averaged over folds and seeds.

    Classes with fewer than n_folds samples are excluded from CV to avoid
    StratifiedKFold warnings and unstable splits.  The final model (step 3)
    still trains on all data including these rare classes.

    rf_n_jobs: n_jobs passed to RandomForestClassifier.
               Set to 1 when the caller already parallelises over k.
    """
    counts = train_df["substrate"].value_counts()
    rare   = set(counts[counts < n_folds].index)
    if rare:
        cv_df = train_df[~train_df["substrate"].isin(rare)]
    else:
        cv_df = train_df

    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    y_all = LabelEncoder().fit_transform(cv_df["substrate"])  # for stratify

    fold_accs = []
    for fold_idx, (tr_idx, val_idx) in enumerate(
            skf.split(cv_df, y_all)):
        tr  = cv_df.iloc[tr_idx]
        val = cv_df.iloc[val_idx]

        # Fit encoders on fold train
        feat_cols = [f"grsA_{p}" for p in positions]
        X_tr_raw  = tr[feat_cols].fillna("-").astype(str)
        X_val_raw = val[feat_cols].fillna("-").astype(str)

        le  = LabelEncoder().fit(tr["substrate"])
        # Classes in val not seen in tr would be ignored by handle_unknown
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_tr  = ohe.fit_transform(X_tr_raw)
        X_val = ohe.transform(X_val_raw)
        y_tr  = le.transform(tr["substrate"])

        # Eval val labels — only rows whose substrate was seen in tr fold
        val_known_mask = val["substrate"].isin(tr["substrate"].unique())
        val_known      = val[val_known_mask]
        if len(val_known) == 0:
            continue
        X_val_known = ohe.transform(
            val_known[feat_cols].fillna("-").astype(str)
        )
        y_val_known = le.transform(val_known["substrate"])

        seed_accs = []
        for seed in seeds:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed,
                n_jobs=rf_n_jobs,
            )
            rf.fit(X_tr, y_tr)
            preds = rf.predict(X_val_known)
            seed_accs.append((preds == y_val_known).mean())

        fold_accs.append(float(np.mean(seed_accs)))

    return float(np.mean(fold_accs)) if fold_accs else 0.0


def _find_knee(cv_df: pd.DataFrame) -> int:
    """
    Kneedle-style knee detection: find the k whose CV accuracy has the
    maximum perpendicular distance from the line connecting the first and
    last points of the curve.
    """
    x = cv_df["k"].values.astype(float)
    y = cv_df["cv_acc"].values.astype(float)

    # Normalise both axes to [0, 1]
    x_n = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    y_n = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y

    # Perpendicular distance from each point to the line (x_n[0],y_n[0])→(x_n[-1],y_n[-1])
    x0, y0, x1, y1 = x_n[0], y_n[0], x_n[-1], y_n[-1]
    dx, dy = x1 - x0, y1 - y0
    line_len = np.sqrt(dx ** 2 + dy ** 2)
    if line_len == 0:
        return int(cv_df.iloc[0]["k"])

    distances = np.abs(dy * x_n - dx * y_n + x1 * y0 - y1 * x0) / line_len
    return int(cv_df.iloc[int(np.argmax(distances))]["k"])


def select_k_by_cv(train_df: pd.DataFrame,
                   ranked_positions: list[int],
                   min_k: int,
                   max_k: int,
                   step: int,
                   n_folds: int,
                   seeds: list[int],
                   n_estimators: int,
                   k_criterion: str = "peak",
                   n_jobs: int = 1) -> tuple[int, pd.DataFrame]:
    """
    For k in [min_k, max_k] (step size `step`), compute CV accuracy.
    Return (best_k, results_df).

    k_criterion : "peak" – k with maximum CV accuracy (default)
                  "knee"  – elbow/knee of the CV curve (Kneedle method)
    n_jobs      : number of parallel workers for the k sweep.
                  -1 = use all CPU cores.  RF is single-threaded when n_jobs != 1
                  to avoid nested parallelism.
    """
    max_k   = min(max_k, len(ranked_positions))
    ks      = list(range(min_k, max_k + 1, step))
    rf_jobs = 1 if n_jobs != 1 else 2   # avoid nested parallelism

    def _eval_k(k):
        acc = cv_accuracy(train_df, ranked_positions[:k],
                          n_folds, seeds, n_estimators, rf_n_jobs=rf_jobs)
        return k, acc

    if n_jobs == 1:
        results = []
        for k in tqdm(ks, desc="CV k-selection"):
            k_val, acc = _eval_k(k)
            results.append((k_val, acc))
            tqdm.write(f"  k={k_val:3d}  cv_acc={acc:.4f}")
    else:
        print(f"  Parallelising over {len(ks)} k values (n_jobs={n_jobs}) …")
        results = Parallel(n_jobs=n_jobs)(delayed(_eval_k)(k) for k in ks)
        results.sort(key=lambda x: x[0])
        for k_val, acc in results:
            print(f"  k={k_val:3d}  cv_acc={acc:.4f}")

    results_df = pd.DataFrame(results, columns=["k", "cv_acc"])

    if k_criterion == "knee":
        best_k = _find_knee(results_df)
    else:
        best_k = int(results_df.loc[results_df["cv_acc"].idxmax(), "k"])

    return best_k, results_df


# ── Step 4: Final model & benchmark evaluation ─────────────────────────────────

def train_and_evaluate(train_df: pd.DataFrame,
                       benchmark_df: pd.DataFrame,
                       positions: list[int],
                       seeds: list[int],
                       n_estimators: int) -> dict:
    """
    Train RF on full train_df with `positions`, evaluate on benchmark_df.
    Returns dict with per-seed and mean/std accuracies.
    """
    feat_cols  = [f"grsA_{p}" for p in positions]
    X_tr_raw   = train_df[feat_cols].fillna("-").astype(str)

    le   = LabelEncoder().fit(train_df["substrate"])
    y_tr = le.transform(train_df["substrate"])

    # Benchmark rows where at least one true label was seen in train
    train_classes = set(train_df["substrate"].unique())
    bm_known_mask = benchmark_df["true_subs"].apply(
        lambda subs: any(s in train_classes for s in subs)
    )
    bm_known = benchmark_df[bm_known_mask].reset_index(drop=True)

    accs = []
    for seed in seeds:
        ohe  = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_tr = ohe.fit_transform(X_tr_raw)

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=2,
        )
        rf.fit(X_tr, y_tr)

        X_bm_k = ohe.transform(bm_known[feat_cols].fillna("-").astype(str))
        preds   = le.inverse_transform(rf.predict(X_bm_k))

        # 1-hit: prediction counts as correct if it appears in true_subs
        hits = sum(p in subs for p, subs in zip(preds, bm_known["true_subs"]))
        accs.append(hits / len(benchmark_df))

    return {
        "mean":  float(np.mean(accs)),
        "std":   float(np.std(accs)),
        "accs":  accs,
        "n_benchmark_total":  len(benchmark_df),
        "n_benchmark_known":  int(bm_known_mask.sum()),
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_cv_curve(cv_df: pd.DataFrame,
                  best_k: int,
                  run_tag: str,
                  method: str,
                  out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = cv_df["k"].values
    y = cv_df["cv_acc"].values

    ax.plot(x, y, color="#2563EB", linewidth=1.8, zorder=3)
    ax.axvline(best_k, color="#DC2626", linestyle="--", linewidth=1.2,
               label=f"Best k={best_k}")

    best_acc = float(cv_df.loc[cv_df["k"] == best_k, "cv_acc"].iloc[0])
    ax.scatter([best_k], [best_acc], color="#DC2626", s=80, zorder=5)
    ax.annotate(
        f"Best k={best_k}\nCV acc={best_acc:.2%}",
        xy=(best_k, best_acc),
        xytext=(best_k + max(1, len(x) // 15), best_acc - 0.02),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2),
        fontsize=9, color="#DC2626",
    )

    ax.set_xlabel("Number of features (positions)", fontsize=11)
    ax.set_ylabel("CV accuracy", fontsize=11)
    ax.set_title(
        f"CV k-selection — {run_tag} / {method}",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  CV plot saved → {out_path.relative_to(ROOT)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Feature-selection pipeline: rank → CV k-select → benchmark eval."
    )
    ap.add_argument("--train-kingdom", default="fungal",
                    choices=["bacterial", "fungal", "all"],
                    help="Kingdom(s) to train on: bacterial / fungal / all ")
    ap.add_argument("--test-kingdom",  default="fungal",
                    choices=["bacterial", "fungal"],
                    help="Kingdom to evaluate on")
    ap.add_argument("--method",    default=None,
                    choices=VALID_METHODS + ["all"],
                    help="Ranking method (default: all five)")
    ap.add_argument("--min-k",     type=int, default=5,
                    help="Minimum k to try in CV (default: 5)")
    ap.add_argument("--max-k",     type=int, default=40,
                    help="Maximum k to try in CV (default: 40)")
    ap.add_argument("--step",      type=int, default=1,
                    help="Step size for k sweep (default: 1)")
    ap.add_argument("--cv-folds",  type=int, default=5,
                    help="Number of CV folds (default: 5)")
    ap.add_argument("--n-seeds",   type=int, default=5,
                    help="RF seeds per evaluation point (default: 5)")
    ap.add_argument("--n-estimators", type=int, default=300,
                    help="Number of RF trees (default: 300)")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="Parallel workers for CV k sweep: -1 = all cores (default)")
    ap.add_argument("--k-criterion", default="peak",
                    choices=["peak", "knee"],
                    help="How to pick best k: 'peak' = max CV acc (default), "
                         "'knee' = elbow of the CV curve")
    ap.add_argument("--skip-ranking", action="store_true",
                    help="Load pre-computed rankings from out_dir instead of recomputing")
    args = ap.parse_args()

    methods  = VALID_METHODS if (args.method is None or args.method == "all") \
               else [args.method]
    seeds    = list(range(args.n_seeds))
    run_tag  = f"{args.train_kingdom}_test_{args.test_kingdom}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Train kingdom : {args.train_kingdom}")
    print(f"  Test kingdom  : {args.test_kingdom}")
    print(f"  Methods       : {methods}")
    print(f"  k range       : [{args.min_k}, {args.max_k}]  step={args.step}")
    print(f"  CV folds      : {args.cv_folds}   seeds: {seeds}")
    print(f"{'='*60}")

    included     = load_benchmark_substrates(args.test_kingdom)
    train_df, benchmark_df = load_splits(args.train_kingdom, args.test_kingdom, included)
    substrates   = sorted(train_df["substrate"].unique().tolist())

    counts     = train_df["substrate"].value_counts()
    rare_count = int((counts < args.cv_folds).sum())

    print(f"\n  Benchmark substrates : {len(included)}")
    print(f"  Train rows           : {len(train_df)}")
    print(f"  Train classes        : {len(substrates)}"
          + (f"  ({rare_count} rare, excluded from CV)" if rare_count else ""))
    print(f"  Benchmark rows       : {len(benchmark_df)}")

    # ── Step 1: Rankings ──────────────────────────────────────────────────────
    if args.skip_ranking:
        print("\n[Step 1] Loading pre-computed rankings …")
        rankings: dict[str, list[int]] = {}
        for name in methods:
            csv = OUT_DIR / f"ranking_{name}_{run_tag}.csv"
            if not csv.exists():
                raise FileNotFoundError(
                    f"Ranking file not found: {csv}\n"
                    f"Re-run without --skip-ranking to compute them."
                )
            rankings[name] = pd.read_csv(csv)["position"].tolist()
    else:
        print("\n[Step 1] Computing position rankings on train set …")
        rankings = compute_all_rankings(train_df, substrates, methods)
        save_rankings(rankings, run_tag, OUT_DIR)
        print(f"  Rankings saved to {OUT_DIR.relative_to(ROOT)}")

    # ── Step 2: CV k-selection ────────────────────────────────────────────────
    print("\n[Step 2] Selecting best k by cross-validation (train only) …")
    summary_rows = []

    for method in methods:
        print(f"\n  --- Method: {method} ---")
        ranked_positions = rankings[method]
        best_k, cv_df = select_k_by_cv(
            train_df        = train_df,
            ranked_positions= ranked_positions,
            min_k           = args.min_k,
            max_k           = args.max_k,
            step            = args.step,
            n_folds         = args.cv_folds,
            seeds           = seeds,
            n_estimators    = args.n_estimators,
            k_criterion     = args.k_criterion,
            n_jobs          = args.n_jobs,
        )
        cv_df.to_csv(OUT_DIR / f"cv_{method}_{run_tag}.csv", index=False)
        plot_cv_curve(cv_df, best_k, run_tag, method,
                      OUT_DIR / f"cv_plot_{method}_{run_tag}.png")
        print(f"  → Best k = {best_k}  "
              f"(CV acc = {cv_df.loc[cv_df['k']==best_k,'cv_acc'].iloc[0]:.4f}"
              f"  criterion={args.k_criterion})")

        # ── Step 3+4: Final model & benchmark eval ────────────────────────────
        print(f"[Step 3+4] Training final model (k={best_k}) & evaluating on benchmark …")
        top_positions = ranked_positions[:best_k]
        result = train_and_evaluate(
            train_df     = train_df,
            benchmark_df = benchmark_df,
            positions    = top_positions,
            seeds        = seeds,
            n_estimators = args.n_estimators,
        )

        print(f"  Benchmark accuracy : {result['mean']:.4f} ± {result['std']:.4f}")
        print(f"  (evaluated on {result['n_benchmark_known']} / "
              f"{result['n_benchmark_total']} benchmark rows with known substrates)")

        summary_rows.append({
            "train_kingdom":      args.train_kingdom,
            "test_kingdom":       args.test_kingdom,
            "method":             method,
            "k_criterion":        args.k_criterion,
            "best_k":             best_k,
            "cv_acc":             float(cv_df.loc[cv_df["k"]==best_k,"cv_acc"].iloc[0]),
            "benchmark_acc_mean": result["mean"],
            "benchmark_acc_std":  result["std"],
            "n_benchmark_total":  result["n_benchmark_total"],
            "n_benchmark_known":  result["n_benchmark_known"],
            "top_positions":      " ".join(str(p) for p in top_positions),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_df   = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / f"summary_{run_tag}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for _, row in summary_df.iterrows():
        print(f"  {row['method']:12s}  best_k={int(row['best_k']):3d}  "
              f"cv={row['cv_acc']:.4f}  "
              f"benchmark={row['benchmark_acc_mean']:.4f}±{row['benchmark_acc_std']:.4f}")
    print(f"\n  Full summary saved → {summary_path.relative_to(ROOT)}")
    print()


if __name__ == "__main__":
    main()
