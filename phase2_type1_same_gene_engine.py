#!/usr/bin/env python3
"""
phase2_type1_same_gene_engine.py

Phase 2 — Type 1 discovery engine
---------------------------------
This is the Jetson-first, non-IBM, same-gene neighborhood discovery engine.

What it does
------------
1. Loads a disease/panel universe from either:
   - a candidate CSV from data_processed/reports/
   - a panel parquet from data_processed/panels/
   - or the full variant_kg.parquet
2. Optionally filters by disease keyword.
3. Builds same-gene neighborhoods.
4. Ranks/selects up to MAX_VARIANTS_PER_GENE variants per gene.
5. Builds the same h and J style local disease-load model used in the project:
      h[i] from ClinVar significance
      J[i,j] from shared disease annotations + pathogenic bonus
6. Enumerates ALL 2^n states locally on the Jetson (exact scan, not quantum).
7. Saves per-gene and overall reports.

Why this matters
----------------
This lets you do local exact discovery on the Jetson first, before any IBM step.
It is repeatable for ANY disease or panel as long as the input has:
  rsid, gene_symbol, clinvar_significance, clinvar_diseases

Outputs
-------
Writes to:
  data_processed/type1_same_gene/

Main outputs:
  type1_gene_summary.csv
  selected_neighborhoods.csv
  per_gene/<GENE>_selected_variants.csv
  per_gene/<GENE>_state_scan.csv
  type1_report.txt
"""

from __future__ import annotations

from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, PANELS_DIR, VARIANT_KG_PATH


# =============================================================================
# USER CONFIG
# =============================================================================

# Choose one:
#   "report_csv"   -> e.g. eds_pathogenic_candidates.csv
#   "panel_parquet"-> e.g. eds_panel_variants.parquet
#   "variant_kg"   -> full universe (usually only with a disease filter)
SOURCE_MODE = "report_csv"

# If SOURCE_MODE == "report_csv"
SOURCE_REPORT_FILENAME = "eds_pathogenic_candidates.csv"   # or collagen_pathogenic_candidates.csv

# If SOURCE_MODE == "panel_parquet"
SOURCE_PANEL_FILENAME = "eds_panel_variants.parquet"       # or collagen_panel_variants.parquet

# Disease keyword filter (optional). Leave None or "" for no extra filter.
# Example: "Ehlers-Danlos", "classic_type", "kyphoscoliotic", "aneurysm"
DISEASE_KEYWORD = ""

# Optional exact gene whitelist. Leave [] to let the engine discover top genes automatically.
GENE_WHITELIST: List[str] = []

# Only keep rows with significance bucket at or above this level.
# Options by score:
#   pathogenic=5, likely_pathogenic=4, conflicting_pathogenic=3, risk_factor=2, vus_or_uncertain=1
MIN_SEVERITY_SCORE = 3

# Neighborhood sizing
MIN_ROWS_PER_GENE = 2
MAX_GENES_TO_SCAN = 25
MAX_VARIANTS_PER_GENE = 8   # exact state scan will enumerate 2^n states
MAX_STATES_WARNING = 2 ** 12

# Extra ranking preferences
REQUIRE_NONEMPTY_RSID = True
SORT_BY = ["severity_score", "disease_text_len", "rsid"]
SORT_ASC = [False, False, True]

# Output location
OUT_DIR = DATA_PROCESSED / "type1_same_gene"
PER_GENE_DIR = OUT_DIR / "per_gene"


# =============================================================================
# Helpers
# =============================================================================

def to_hashable_string(x) -> str:
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        if x.size == 1:
            return to_hashable_string(x.item())
        return "|".join(sorted({to_hashable_string(v) for v in x.tolist()}))
    if isinstance(x, (list, tuple, set)):
        if len(x) == 0:
            return ""
        if len(x) == 1:
            return to_hashable_string(next(iter(x)))
        return "|".join(sorted({to_hashable_string(v) for v in x}))
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["rsid", "gene_symbol", "clinvar_significance", "clinvar_diseases", "chrom", "ref", "alt"]:
        if col in df.columns:
            df[col] = df[col].apply(to_hashable_string)
    return df


def normalize_significance(sig: str) -> str:
    s = to_hashable_string(sig).lower()
    if not s:
        return "missing"

    if "conflicting" in s and "pathogenic" in s:
        return "conflicting_pathogenic"
    if "pathogenic" in s and "benign" not in s:
        if "likely" in s:
            return "likely_pathogenic"
        return "pathogenic"
    if "benign" in s and "pathogenic" not in s:
        if "likely" in s:
            return "likely_benign"
        return "benign"
    if "uncertain" in s or "vus" in s:
        return "vus_or_uncertain"
    if "risk factor" in s:
        return "risk_factor"
    if "drug response" in s:
        return "drug_response"
    if "association" in s:
        return "association"
    if "protective" in s:
        return "protective"
    if "affects" in s:
        return "affects"
    return "other_or_mixed"


def severity_score(sig: str) -> int:
    bucket = normalize_significance(sig)
    score_map = {
        "pathogenic": 5,
        "likely_pathogenic": 4,
        "conflicting_pathogenic": 3,
        "risk_factor": 2,
        "vus_or_uncertain": 1,
        "drug_response": 1,
        "association": 1,
        "protective": 0,
        "likely_benign": 0,
        "benign": 0,
        "affects": 0,
        "other_or_mixed": 0,
        "missing": 0,
    }
    return score_map.get(bucket, 0)


def split_diseases(val) -> set[str]:
    if not isinstance(val, str) or not val:
        return set()
    raw = str(val).replace("|", ",")
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    return {p for p in parts if "not_provided" not in p.lower() and len(p) > 2}


def load_source_df() -> pd.DataFrame:
    if SOURCE_MODE == "report_csv":
        path = DATA_PROCESSED / "reports" / SOURCE_REPORT_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"Report CSV not found: {path}")
        df = pd.read_csv(path)
        print(f"[LOAD] report_csv: {path}  rows={len(df):,}")
        return normalize_df(df)

    if SOURCE_MODE == "panel_parquet":
        path = PANELS_DIR / SOURCE_PANEL_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"Panel parquet not found: {path}")
        df = pd.read_parquet(path)
        print(f"[LOAD] panel_parquet: {path}  rows={len(df):,}")
        return normalize_df(df)

    if SOURCE_MODE == "variant_kg":
        path = VARIANT_KG_PATH
        if not path.exists():
            raise FileNotFoundError(f"variant_kg not found: {path}")
        df = pd.read_parquet(path)
        print(f"[LOAD] variant_kg: {path}  rows={len(df):,}")
        return normalize_df(df)

    raise ValueError(f"Unknown SOURCE_MODE: {SOURCE_MODE}")


# =============================================================================
# Local model (same h/J concepts as the project)
# =============================================================================

def build_h(subset_df: pd.DataFrame) -> Dict[int, float]:
    h: Dict[int, float] = {}
    for i, row in subset_df.iterrows():
        sig = row.get("clinvar_significance", "")
        bucket = normalize_significance(sig)
        if bucket == "pathogenic":
            h[i] = 2.0
        elif bucket == "likely_pathogenic":
            h[i] = 1.5
        elif bucket == "conflicting_pathogenic":
            h[i] = 1.5
        elif bucket == "risk_factor":
            h[i] = 1.0
        elif bucket == "likely_benign":
            h[i] = -0.5
        elif bucket == "benign":
            h[i] = -1.0
        else:
            h[i] = 0.0
    return h


def build_J(subset_df: pd.DataFrame, h: Dict[int, float]) -> Dict[Tuple[int, int], float]:
    J: Dict[Tuple[int, int], float] = {}
    n = len(subset_df)
    diseases = [split_diseases(subset_df.loc[i, "clinvar_diseases"]) for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            shared = diseases[i] & diseases[j]
            val = 0.0

            if shared:
                val += 0.8
            if h.get(i, 0.0) > 0.5 and h.get(j, 0.0) > 0.5:
                val += 0.4

            if abs(val) > 0:
                J[(i, j)] = val
    return J


def energy_of_state(bits: Tuple[int, ...], h: Dict[int, float], J: Dict[Tuple[int, int], float]) -> float:
    # bit 0 => absent => spin +1
    # bit 1 => present => spin -1
    spins = np.array([+1.0 if b == 0 else -1.0 for b in bits], dtype=float)
    E = 0.0
    for i, h_i in h.items():
        E += h_i * spins[i]
    for (i, j), j_ij in J.items():
        E += j_ij * spins[i] * spins[j]
    return float(E)


def enumerate_states(subset_df: pd.DataFrame) -> pd.DataFrame:
    n = len(subset_df)
    h = build_h(subset_df)
    J = build_J(subset_df, h)

    rows = []
    rsids = subset_df["rsid"].fillna("").astype(str).tolist()
    genes = subset_df["gene_symbol"].fillna("").astype(str).tolist()

    for bits in product([0, 1], repeat=n):
        bitstring = "".join(str(b) for b in bits)
        E = energy_of_state(bits, h, J)

        present_rsids = [rsids[i] for i, b in enumerate(bits) if b == 1]
        present_genes = [genes[i] for i, b in enumerate(bits) if b == 1]

        rows.append({
            "bitstring": bitstring,
            "n_present": int(sum(bits)),
            "energy": E,
            "present_rsids": "|".join(present_rsids),
            "present_genes": "|".join(present_genes),
        })

    out = pd.DataFrame(rows).sort_values(["energy", "n_present"], ascending=[True, False]).reset_index(drop=True)
    return out


# =============================================================================
# Neighborhood construction
# =============================================================================

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["rsid", "gene_symbol", "clinvar_significance", "clinvar_diseases"]:
        if col not in df.columns:
            df[col] = ""

    df["significance_bucket"] = df["clinvar_significance"].apply(normalize_significance)
    df["severity_score"] = df["clinvar_significance"].apply(severity_score)
    df["disease_text_len"] = df["clinvar_diseases"].fillna("").astype(str).str.len()

    if REQUIRE_NONEMPTY_RSID:
        df = df[df["rsid"].fillna("").astype(str).str.strip() != ""]

    if DISEASE_KEYWORD:
        mask = df["clinvar_diseases"].fillna("").astype(str).str.contains(
            re.escape(DISEASE_KEYWORD), case=False, na=False
        )
        df = df[mask].copy()

    df = df[df["severity_score"] >= MIN_SEVERITY_SCORE].copy()

    df["rsid"] = df["rsid"].astype(str).str.replace("^rs", "", regex=True)
    df["gene_symbol"] = df["gene_symbol"].astype(str)

    return df


def build_gene_neighborhoods(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if GENE_WHITELIST:
        df = df[df["gene_symbol"].isin(GENE_WHITELIST)].copy()

    gene_counts = (
        df["gene_symbol"]
        .fillna("")
        .astype(str)
        .value_counts()
        .rename_axis("gene_symbol")
        .reset_index(name="count_rows")
    )

    gene_counts = gene_counts[gene_counts["count_rows"] >= MIN_ROWS_PER_GENE].copy()

    if not GENE_WHITELIST:
        gene_counts = gene_counts.head(MAX_GENES_TO_SCAN)

    selected_neighborhoods = []
    gene_to_df: Dict[str, pd.DataFrame] = {}

    for gene in gene_counts["gene_symbol"].tolist():
        gdf = df[df["gene_symbol"] == gene].copy()

        gdf = gdf.sort_values(SORT_BY, ascending=SORT_ASC)
        gdf = gdf.drop_duplicates(subset=["rsid", "chrom", "pos", "ref", "alt"], keep="first")
        gdf = gdf.head(MAX_VARIANTS_PER_GENE).reset_index(drop=True)

        if len(gdf) < MIN_ROWS_PER_GENE:
            continue

        selected_neighborhoods.append({
            "gene_symbol": gene,
            "count_rows_original": int(len(df[df["gene_symbol"] == gene])),
            "count_rows_selected": int(len(gdf)),
            "estimated_states": int(2 ** len(gdf)),
            "top_significance_buckets": "|".join(gdf["significance_bucket"].value_counts().index.tolist()[:5]),
            "top_rsids": "|".join(gdf["rsid"].astype(str).tolist()[:10]),
        })
        gene_to_df[gene] = gdf

    neighborhood_df = pd.DataFrame(selected_neighborhoods).sort_values(
        ["count_rows_selected", "count_rows_original"], ascending=[False, False]
    ).reset_index(drop=True)

    return neighborhood_df, gene_to_df


# =============================================================================
# Reporting
# =============================================================================

def write_report(summary_df: pd.DataFrame, source_df: pd.DataFrame) -> None:
    lines = []
    lines.append("QuantumGenome — Phase 2 Type 1 Same-Gene Discovery Engine")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"SOURCE_MODE            : {SOURCE_MODE}")
    lines.append(f"SOURCE_REPORT_FILENAME : {SOURCE_REPORT_FILENAME}")
    lines.append(f"SOURCE_PANEL_FILENAME  : {SOURCE_PANEL_FILENAME}")
    lines.append(f"DISEASE_KEYWORD        : {DISEASE_KEYWORD}")
    lines.append(f"MIN_SEVERITY_SCORE     : {MIN_SEVERITY_SCORE}")
    lines.append(f"MAX_GENES_TO_SCAN      : {MAX_GENES_TO_SCAN}")
    lines.append(f"MAX_VARIANTS_PER_GENE  : {MAX_VARIANTS_PER_GENE}")
    lines.append("")
    lines.append(f"Filtered source rows   : {len(source_df):,}")
    lines.append(f"Neighborhoods scanned  : {len(summary_df):,}")
    lines.append("")

    if len(summary_df) > 0:
        lines.append("Top neighborhoods by minimum energy")
        lines.append("-" * 78)
        for _, row in summary_df.head(25).iterrows():
            lines.append(
                f"{row['gene_symbol']:<14} "
                f"selected={int(row['count_rows_selected']):>2}  "
                f"states={int(row['estimated_states']):>7,}  "
                f"min_energy={row['min_energy']:>8.3f}  "
                f"top_bitstring={row['top_bitstring']}"
            )
        lines.append("")

    lines.append("Interpretation:")
    lines.append("- Each gene is treated as its own local neighborhood.")
    lines.append("- The engine selects up to MAX_VARIANTS_PER_GENE variants per gene.")
    lines.append("- It then enumerates ALL 2^n states exactly on the Jetson.")
    lines.append("- Lower energy = more disease-loaded combination under the current local model.")
    lines.append("- This is repeatable for any disease or panel by changing SOURCE_MODE and DISEASE_KEYWORD.")
    lines.append("")

    (OUT_DIR / "type1_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_GENE_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_source_df()
    df = prepare_df(raw_df)

    if df.empty:
        raise ValueError("No rows left after filtering. Loosen DISEASE_KEYWORD or MIN_SEVERITY_SCORE.")

    neighborhood_df, gene_to_df = build_gene_neighborhoods(df)

    if neighborhood_df.empty:
        raise ValueError("No same-gene neighborhoods met the current thresholds.")

    neighborhood_df.to_csv(OUT_DIR / "selected_neighborhoods.csv", index=False)

    summary_rows = []

    for _, nrow in neighborhood_df.iterrows():
        gene = nrow["gene_symbol"]
        gdf = gene_to_df[gene].copy()

        gdf.to_csv(PER_GENE_DIR / f"{gene}_selected_variants.csv", index=False)

        n = len(gdf)
        if 2 ** n > MAX_STATES_WARNING:
            print(f"[WARN] {gene}: selected {n} variants => {2**n:,} states. Still scanning exactly.")

        states_df = enumerate_states(gdf)
        states_df.to_csv(PER_GENE_DIR / f"{gene}_state_scan.csv", index=False)

        best = states_df.iloc[0]
        summary_rows.append({
            "gene_symbol": gene,
            "count_rows_original": int(nrow["count_rows_original"]),
            "count_rows_selected": int(nrow["count_rows_selected"]),
            "estimated_states": int(nrow["estimated_states"]),
            "min_energy": float(best["energy"]),
            "top_bitstring": best["bitstring"],
            "top_present_rsids": best["present_rsids"],
            "top_present_genes": best["present_genes"],
            "top_significance_buckets": nrow["top_significance_buckets"],
            "top_rsids": nrow["top_rsids"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["min_energy", "count_rows_selected"], ascending=[True, False]
    ).reset_index(drop=True)

    summary_df.to_csv(OUT_DIR / "type1_gene_summary.csv", index=False)
    write_report(summary_df, df)

    print("\n" + "=" * 78)
    print("QuantumGenome — Phase 2 Type 1 Same-Gene Discovery Engine")
    print("=" * 78)
    print(f"Filtered source rows  : {len(df):,}")
    print(f"Neighborhoods scanned : {len(summary_df):,}")
    print("")
    print("Top neighborhoods by minimum energy:")
    print(summary_df.head(20).to_string(index=False))
    print("")
    print(f"Outputs written to: {OUT_DIR}")
    print("Most important files:")
    print(f"  - {OUT_DIR / 'type1_report.txt'}")
    print(f"  - {OUT_DIR / 'type1_gene_summary.csv'}")
    print(f"  - {OUT_DIR / 'selected_neighborhoods.csv'}")
    print(f"  - {PER_GENE_DIR}/*_selected_variants.csv")
    print(f"  - {PER_GENE_DIR}/*_state_scan.csv")


if __name__ == "__main__":
    main()
