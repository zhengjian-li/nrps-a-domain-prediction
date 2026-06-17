"""
Position-level feature scoring models for NRPS A-domain residue selection.

Three scoring methods are provided:
  - InformationGain  : G(t) = H(C) - H(C|t)
  - MutualInformation: I(t; C) = sum_c p(t,c) log[p(t,c)/(p(t)p(c))],
                       averaged over classes as sum_c p(c) I(t,c)
  - ChiSquared       : χ²(t, c) via 2×2 contingency table,
                       averaged over classes as sum_c p(c) χ²(t,c)

All classes expect a DataFrame whose position columns are named with plain
integers (e.g. "235", "236") and one-hot encoded substrate columns whose
names match the `substrates` list passed at construction.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Shared base ───────────────────────────────────────────────────────────────

class _FeatureBase:
    """Common setup shared by all scoring classes."""

    def __init__(self, df: pd.DataFrame, substrates: list[str]) -> None:
        self.df         = df
        self.substrates = substrates
        self.positions  = [col for col in df.columns if col.isdigit()]
        self.position_aa   = self._build_position_aa()
        self.term_masks    = self._build_term_masks()

    def _build_position_aa(self) -> dict[str, list[str]]:
        """Map each position column to the amino acids observed there (excl. gaps)."""
        result = {}
        for pos in self.positions:
            aas = self.df[pos].unique().tolist()
            if "-" in aas:
                aas.remove("-")
            result[pos] = aas
        return result

    def _build_term_masks(self) -> dict[tuple[str, str], pd.Series]:
        """Boolean mask for every (position, aa) term."""
        masks = {}
        for pos in self.positions:
            for aa in self.position_aa[pos]:
                masks[(pos, aa)] = self.df[pos] == aa
        return masks


# ── Information Gain ──────────────────────────────────────────────────────────

class InformationGain(_FeatureBase):
    r"""
    Information gain of a binary term t (position p has amino acid a):

        G(t) = H(C)
             - P(t)   · H(C | t)
             - P(~t)  · H(C | ~t)

    where H(C) = -Σ P(c_i) log P(c_i).
    Returns one score per (position, aa) pair via `get_all_information_gain()`.
    """

    def __init__(self, df: pd.DataFrame, substrates: list[str]) -> None:
        super().__init__(df, substrates)
        self.label_masks = {c: (df[c] == 1) for c in substrates}
        self.p_c   = {c: self.label_masks[c].mean() for c in substrates}
        self.p_t   = {k: v.mean() for k, v in self.term_masks.items()}

    def _entropy(self) -> float:
        return -sum(p * np.log2(p) for p in self.p_c.values() if p > 0)

    def _cond_entropy(self, mask: pd.Series) -> float:
        if mask.sum() == 0:
            return 0.0
        h = 0.0
        for c in self.substrates:
            p = self.label_masks[c][mask].mean()
            if p > 0:
                h -= p * np.log2(p)
        return h

    def _score(self, pos: str, aa: str) -> float:
        mask_t    = self.term_masks[(pos, aa)]
        mask_tbar = ~mask_t
        p_t    = self.p_t[(pos, aa)]
        p_tbar = 1.0 - p_t
        return (self._entropy()
                - p_t    * self._cond_entropy(mask_t)
                - p_tbar * self._cond_entropy(mask_tbar))

    def get_all_information_gain(self) -> dict[tuple[str, str], float]:
        scores = {}
        for pos in tqdm(self.positions, desc="InformationGain"):
            for aa in self.position_aa[pos]:
                scores[(pos, aa)] = self._score(pos, aa)
        return scores


# ── Mutual Information ────────────────────────────────────────────────────────

class MutualInformation(_FeatureBase):
    r"""
    Mutual information between a binary term t and each class label c:

        I(t; c) = p(t,c) · log[ p(t,c) / (p(t)·p(c)) ]

    `get_mi_avg()` returns the class-probability-weighted average:

        MI_avg(t) = Σ_c p(c) · I(t; c)
    """

    def __init__(self, df: pd.DataFrame, substrates: list[str]) -> None:
        super().__init__(df, substrates)
        self.label_masks = {c: (df[c] == 1) for c in substrates}
        self.p_c  = {c: self.label_masks[c].mean() for c in substrates}
        self.p_t  = {k: v.mean() for k, v in self.term_masks.items()}

        n = len(df)
        self._p_tc: dict[tuple[str, str, str], float] = {
            (pos, aa, c): (self.term_masks[(pos, aa)] & self.label_masks[c]).sum() / n
            for (pos, aa) in tqdm(self.term_masks, desc="MI  p(t,c)")
            for c in substrates
        }

    def _mi(self, pos: str, aa: str, c: str) -> float:
        p_tc = self._p_tc.get((pos, aa, c), 0)
        p_t  = self.p_t.get((pos, aa), 0)
        p_c  = self.p_c.get(c, 0)
        if p_tc == 0 or p_t == 0 or p_c == 0:
            return 0.0
        return float(p_tc * np.log2(p_tc / (p_t * p_c)))

    def get_mi_avg(self) -> dict[tuple[str, str], float]:
        """Weighted-average MI per (position, aa): Σ_c p(c)·I(t,c)."""
        result = {}
        for (pos, aa) in tqdm(self.term_masks, desc="MI  avg"):
            result[(pos, aa)] = sum(
                self._mi(pos, aa, c) * self.p_c[c] for c in self.substrates
            )
        return result

    def get_mi_max(self) -> dict[tuple[str, str], float]:
        """Max MI over classes per (position, aa): max_c I(t,c)."""
        result = {}
        for (pos, aa) in tqdm(self.term_masks, desc="MI  max"):
            result[(pos, aa)] = max(
                self._mi(pos, aa, c) for c in self.substrates
            )
        return result


# ── Chi-Squared ───────────────────────────────────────────────────────────────

class ChiSquared(_FeatureBase):
    r"""
    Chi-squared statistic for a binary term t against each class c via a
    2×2 contingency table (A, B, C, D):

        χ²(t, c) = N·(AD − BC)² / [(A+C)(B+D)(A+B)(C+D)]

    `get_chi2_avg()` returns the class-probability-weighted average:

        χ²_avg(t) = Σ_c p(c) · χ²(t, c)
    """

    def __init__(self, df: pd.DataFrame, substrates: list[str]) -> None:
        super().__init__(df, substrates)
        self.label_masks_1 = {c: (df[c] == 1) for c in substrates}
        self.label_masks_0 = {c: (df[c] == 0) for c in substrates}
        self.p_c = {c: self.label_masks_1[c].mean() for c in substrates}

    def _chi2(self, pos: str, aa: str, c: str) -> float:
        mask_t = self.term_masks[(pos, aa)]
        m1     = self.label_masks_1[c]
        m0     = self.label_masks_0[c]
        A = ( mask_t &  m1).sum()
        B = (~mask_t &  m1).sum()
        C = ( mask_t &  m0).sum()
        D = (~mask_t &  m0).sum()
        N = A + B + C + D
        if N == 0:
            return 0.0
        denom = (A + C) * (B + D) * (A + B) * (C + D)
        if denom == 0:
            return 0.0
        return float(N * (A * D - B * C) ** 2 / denom)

    def get_chi2_avg(self) -> dict[tuple[str, str], float]:
        """Weighted-average χ² per (position, aa): Σ_c p(c)·χ²(t,c)."""
        result = {}
        for pos in tqdm(self.positions, desc="Chi²  avg"):
            for aa in self.position_aa[pos]:
                result[(pos, aa)] = sum(
                    self._chi2(pos, aa, c) * self.p_c[c] for c in self.substrates
                )
        return result

    def get_chi2_max(self) -> dict[tuple[str, str], float]:
        """Max χ² over classes per (position, aa): max_c χ²(t,c)."""
        result = {}
        for pos in tqdm(self.positions, desc="Chi²  max"):
            for aa in self.position_aa[pos]:
                result[(pos, aa)] = max(
                    self._chi2(pos, aa, c) for c in self.substrates
                )
        return result
