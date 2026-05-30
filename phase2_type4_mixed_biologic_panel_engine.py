#!/usr/bin/env python3
"""
phase2_type4_mixed_biologic_panel_engine.py

Phase 2 — Type 4 discovery engine
---------------------------------
Jetson-first, non-IBM, mixed biologic panel neighborhood discovery engine.

What it does
------------
1. Loads a disease/panel universe from either:
   - a candidate CSV from data_processed/reports/
   - a panel parquet from data_processed/panels/
   - or the full variant_kg.parquet
2. Optionally filters by disease keyword.
3. Builds reusable mixed biologic panels using one of two modes:
   A) explicit GENE_SETS dictionary
   B) auto-generated panels from a seed-gene list
4. Ranks/selects up to MAX_VARIANTS_PER_PANEL variants across the mixed panel.
5. Builds the same local disease-load model used elsewhere:
      h[i] from ClinVar significance
      J[i,j] from shared disease annotations + pathogenic bonus
6. Enumerates ALL 2^n states locally on the Jetson (exact scan, not quantum).
7. Saves per-panel and overall reports.

Why this matters
----------------
This lets you combine related genes into one neighborhood even when they are:
- not all in the same gene
- not all in the same exact disease string
- not physically near each other

This is the best engine for reusable "biology-first" panel discovery across:
- EDS subtypes
- connective tissue diseases
- vascular panels
- collagen panels
- any future disease family

It is reusable for ANY disease or panel as long as the input has:
  rsid, gene_symbol, clinvar_significance, clinvar_diseases

Outputs
-------
Writes to:
  data_processed/type4_mixed_panel/

Main outputs:
  type4_panel_summary.csv
  selected_panels.csv
  per_panel/<PANEL>_selected_variants.csv
  per_panel/<PANEL>_state_scan.csv
  type4_report.txt
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

# Optional broad pre-filter applied before panel construction.
# Example: "Ehlers-Danlos", "aneurysm", "cardiomyopathy"
DISEASE_KEYWORD = "Ehlers-Danlos"

# Build mixed panels in one of two ways:
#   "explicit" -> use GENE_SETS below exactly as written
#   "seed_auto"-> use SEED_GENES and AUTO_PANEL_SIZE to auto-build focused panels
PANEL_BUILD_MODE = "explicit"

# A) Explicit reusable biologic panels
GENE_SETS: Dict[str, List[str]] = {
    "classical_matrix_panel": ["COL5A1", "COL5A2", "TNXB", "AEBP1"],
    "vascular_matrix_panel": ["COL3A1", "COL1A1", "COL1A2", "TNXB"],
    "typeI_collagen_overlap_panel": ["COL1A1", "COL1A2", "COL3A1", "COL5A1"],
    "kypho_crosslink_panel": ["PLOD1", "FKBP14", "TNXB", "COL1A2"],
    "spondylodysplastic_panel": ["B3GALT6", "B4GALT7", "SLC39A13", "COL1A2"],
    "musculocontractural_panel": ["CHST14", "DSE", "COL5A1", "TNXB"],
    "fragility_overlap_panel": ["ZNF469", "PRDM5", "COL1A1", "COL5A1"],
    "dermatosparaxis_panel": ["ADAMTS2", "COL1A1", "COL1A2", "COL5A1"],
}

# B) Auto-generated panels from seed genes
SEED_GENES: List[str] = ["COL3A1", "COL5A1", "COL1A1", "COL1A2", "PLOD1", "B3GALT6", "TNXB"]
AUTO_PANEL_SIZE = 4

# Restrict to a subset of panels by name if desired
PANEL_NAME_WHITELIST: List[str] = []

# Only keep rows with significance bucket at or above this level.
#   pathogenic=5, likely_pathogenic=4, conflicting_pathogenic=3, risk_factor=2, vus_or_uncertain=1
MIN_SEVERITY_SCORE = 3

# Neighborhood sizing
MIN_ROWS_PER_PANEL = 2
MAX_PANELS_TO_SCAN = 30
MAX_VARIANTS_PER_PANEL = 8
MAX_STATES_WARNING = 2 ** 12

# Selection preferences
REQUIRE_NONEMPTY_RSID = True
SORT_BY = ["severity_score", "gene_symbol", "disease_text_len", "rsid"]
SORT_ASC = [False, True, False, True]

# Output location
OUT_DIR = DATA_PROCESSED / "type4_mixed_panel"
PER_PANEL_DIR = OUT_DIR / "per_panel"


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
    txt = to_hashable_string(val)
    if not txt:
        return set()
    raw = txt.replace("|", ",")
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    return {p for p in parts if "not_provided" not in p.lower() and len(p) > 2}


def normalize_rsid(val) -> str:
    s = to_hashable_string(val)
    if not s:
        return ""
    return re.sub(r"^rs", "", s, flags=re.I)


def safe_panel_name(name: str, max_len: int = 120) -> str:
    n = to_hashable_string(name).strip().replace(" ", "_")
    n = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", n)
    n = re.sub(r"__+", "_", n)
    return n[:max_len] or "UNKNOWN_PANEL"


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
# Panel construction
# =============================================================================

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["rsid", "gene_symbol", "clinvar_significance", "clinvar_diseases"]:
        if col not in df.columns:
            df[col] = ""

    df["significance_bucket"] = df["clinvar_significance"].apply(normalize_significance)
    df["severity_score"] = df["clinvar_significance"].apply(severity_score)
    df["disease_text_len"] = df["clinvar_diseases"].fillna("").astype(str).str.len()
    df["rsid"] = df["rsid"].apply(normalize_rsid)

    if REQUIRE_NONEMPTY_RSID:
        df = df[df["rsid"].fillna("").astype(str).str.strip() != ""]

    if DISEASE_KEYWORD:
        mask = df["clinvar_diseases"].fillna("").astype(str).str.contains(
            re.escape(DISEASE_KEYWORD), case=False, na=False
        )
        df = df[mask].copy()

    df = df[df["severity_score"] >= MIN_SEVERITY_SCORE].copy()
    return df


def auto_build_gene_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build seed-centered panels by pairing each seed with the most represented genes
    in the current filtered universe. This keeps the engine reusable.
    """
    all_gene_counts = (
        df["gene_symbol"]
        .fillna("")
        .astype(str)
        .value_counts()
        .index
        .tolist()
    )

    panels: Dict[str, List[str]] = {}
    for seed in SEED_GENES:
        if seed not in all_gene_counts:
            continue

        genes = [seed]
        for g in all_gene_counts:
            if g == seed:
                continue
            if g not in genes:
                genes.append(g)
            if len(genes) >= AUTO_PANEL_SIZE:
                break

        panels[f"auto_{seed}_panel"] = genes

    return panels


def build_panel_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    if PANEL_BUILD_MODE == "explicit":
        panel_map = {k: list(v) for k, v in GENE_SETS.items()}
    elif PANEL_BUILD_MODE == "seed_auto":
        panel_map = auto_build_gene_sets(df)
    else:
        raise ValueError(f"Unknown PANEL_BUILD_MODE: {PANEL_BUILD_MODE}")

    if PANEL_NAME_WHITELIST:
        wanted = {safe_panel_name(x) for x in PANEL_NAME_WHITELIST}
        panel_map = {k: v for k, v in panel_map.items() if safe_panel_name(k) in wanted}

    return panel_map


def build_mixed_panels(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    panel_map = build_panel_map(df)

    rows = []
    panel_to_df: Dict[str, pd.DataFrame] = {}

    for panel_name, gene_list in panel_map.items():
        pdf = df[df["gene_symbol"].isin(gene_list)].copy()

        pdf = pdf.sort_values(SORT_BY, ascending=SORT_ASC)
        pdf = pdf.drop_duplicates(subset=["rsid", "chrom", "pos", "ref", "alt"], keep="first")
        pdf = pdf.head(MAX_VARIANTS_PER_PANEL).reset_index(drop=True)

        if len(pdf) < MIN_ROWS_PER_PANEL:
            continue

        safe_name = safe_panel_name(panel_name)
        rows.append({
            "panel_name": panel_name,
            "safe_panel_name": safe_name,
            "genes_requested": "|".join(gene_list),
            "genes_observed": "|".join(pdf["gene_symbol"].fillna("").astype(str).value_counts().index.tolist()[:20]),
            "count_rows_selected": int(len(pdf)),
            "estimated_states": int(2 ** len(pdf)),
            "top_rsids": "|".join(pdf["rsid"].astype(str).tolist()[:10]),
        })
        panel_to_df[panel_name] = pdf

    panel_df = pd.DataFrame(rows).sort_values(
        ["count_rows_selected", "panel_name"], ascending=[False, True]
    ).reset_index(drop=True)

    if len(panel_df) > MAX_PANELS_TO_SCAN:
        keep_names = panel_df.head(MAX_PANELS_TO_SCAN)["panel_name"].tolist()
        panel_df = panel_df.head(MAX_PANELS_TO_SCAN).reset_index(drop=True)
        panel_to_df = {k: v for k, v in panel_to_df.items() if k in keep_names}

    return panel_df, panel_to_df


# =============================================================================
# Reporting
# =============================================================================

def write_report(summary_df: pd.DataFrame, source_df: pd.DataFrame) -> None:
    lines = []
    lines.append("QuantumGenome — Phase 2 Type 4 Mixed Biologic Panel Discovery Engine")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"SOURCE_MODE             : {SOURCE_MODE}")
    lines.append(f"SOURCE_REPORT_FILENAME  : {SOURCE_REPORT_FILENAME}")
    lines.append(f"SOURCE_PANEL_FILENAME   : {SOURCE_PANEL_FILENAME}")
    lines.append(f"DISEASE_KEYWORD         : {DISEASE_KEYWORD}")
    lines.append(f"PANEL_BUILD_MODE        : {PANEL_BUILD_MODE}")
    lines.append(f"MIN_SEVERITY_SCORE      : {MIN_SEVERITY_SCORE}")
    lines.append(f"MAX_PANELS_TO_SCAN      : {MAX_PANELS_TO_SCAN}")
    lines.append(f"MAX_VARIANTS_PER_PANEL  : {MAX_VARIANTS_PER_PANEL}")
    lines.append("")
    lines.append(f"Filtered source rows    : {len(source_df):,}")
    lines.append(f"Mixed panels scanned    : {len(summary_df):,}")
    lines.append("")

    if len(summary_df) > 0:
        lines.append("Top mixed panels by minimum energy")
        lines.append("-" * 78)
        for _, row in summary_df.head(25).iterrows():
            lines.append(
                f"{row['panel_name']:<34} "
                f"selected={int(row['count_rows_selected']):>2}  "
                f"states={int(row['estimated_states']):>7,}  "
                f"min_energy={row['min_energy']:>8.3f}  "
                f"top_bitstring={row['top_bitstring']}"
            )
        lines.append("")

    lines.append("Interpretation:")
    lines.append("- Each panel is a reusable mixed biologic gene set.")
    lines.append("- The engine selects up to MAX_VARIANTS_PER_PANEL across that panel.")
    lines.append("- It then enumerates ALL 2^n states exactly on the Jetson.")
    lines.append("- Lower energy = more disease-loaded combination under the current local model.")
    lines.append("- This is reusable for any disease or panel by changing SOURCE_MODE, DISEASE_KEYWORD, and GENE_SETS.")
    lines.append("")

    (OUT_DIR / "type4_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_PANEL_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_source_df()
    df = prepare_df(raw_df)

    if df.empty:
        raise ValueError("No rows left after filtering. Loosen DISEASE_KEYWORD or MIN_SEVERITY_SCORE.")

    selected_panels_df, panel_to_df = build_mixed_panels(df)
    if selected_panels_df.empty:
        raise ValueError("No mixed panels met the current thresholds.")

    selected_panels_df.to_csv(OUT_DIR / "selected_panels.csv", index=False)

    summary_rows = []

    for _, prow in selected_panels_df.iterrows():
        panel_name = prow["panel_name"]
        safe_name = prow["safe_panel_name"]
        pdf = panel_to_df[panel_name].copy()

        pdf.to_csv(PER_PANEL_DIR / f"{safe_name}_selected_variants.csv", index=False)

        n = len(pdf)
        if 2 ** n > MAX_STATES_WARNING:
            print(f"[WARN] {panel_name}: selected {n} variants => {2**n:,} states. Still scanning exactly.")

        states_df = enumerate_states(pdf)
        states_df.to_csv(PER_PANEL_DIR / f"{safe_name}_state_scan.csv", index=False)

        best = states_df.iloc[0]
        summary_rows.append({
            "panel_name": panel_name,
            "safe_panel_name": safe_name,
            "genes_requested": prow["genes_requested"],
            "genes_observed": prow["genes_observed"],
            "count_rows_selected": int(prow["count_rows_selected"]),
            "estimated_states": int(prow["estimated_states"]),
            "min_energy": float(best["energy"]),
            "top_bitstring": best["bitstring"],
            "top_present_rsids": best["present_rsids"],
            "top_present_genes": best["present_genes"],
            "top_rsids": prow["top_rsids"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["min_energy", "count_rows_selected"], ascending=[True, False]
    ).reset_index(drop=True)

    summary_df.to_csv(OUT_DIR / "type4_panel_summary.csv", index=False)
    write_report(summary_df, df)

    print("\n" + "=" * 78)
    print("QuantumGenome — Phase 2 Type 4 Mixed Biologic Panel Discovery Engine")
    print("=" * 78)
    print(f"Filtered source rows : {len(df):,}")
    print(f"Mixed panels scanned : {len(summary_df):,}")
    print("")
    print("Top mixed panels by minimum energy:")
    print(summary_df.head(20).to_string(index=False))
    print("")
    print(f"Outputs written to: {OUT_DIR}")
    print("Most important files:")
    print(f"  - {OUT_DIR / 'type4_report.txt'}")
    print(f"  - {OUT_DIR / 'type4_panel_summary.csv'}")
    print(f"  - {OUT_DIR / 'selected_panels.csv'}")
    print(f"  - {PER_PANEL_DIR}/*_selected_variants.csv")
    print(f"  - {PER_PANEL_DIR}/*_state_scan.csv")


if __name__ == "__main__":
    main()
