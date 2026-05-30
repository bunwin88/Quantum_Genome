#!/usr/bin/env python3
"""
phase2_type3_position_window_engine.py

Phase 2 — Type 3 discovery engine
---------------------------------
Jetson-first, non-IBM, genomic position-window neighborhood discovery engine.

What it does
------------
1. Loads a disease/panel universe from either:
   - a candidate CSV from data_processed/reports/
   - a panel parquet from data_processed/panels/
   - or the full variant_kg.parquet
2. Optionally filters by disease keyword.
3. Chooses anchor variants.
4. Builds genomic neighborhoods around each anchor:
      same chromosome, within +/- WINDOW_BP
5. Ranks/selects up to MAX_VARIANTS_PER_WINDOW variants per neighborhood.
6. Builds the same local disease-load model used elsewhere:
      h[i] from ClinVar significance
      J[i,j] from shared disease annotations + pathogenic bonus
7. Enumerates ALL 2^n states locally on the Jetson (exact scan, not quantum).
8. Saves per-window and overall reports.

Why this matters
----------------
This lets you search local genomic neighborhoods and variant clusters that
may not be obvious from gene labels or disease labels alone.

It is reusable for ANY disease or panel as long as the input has:
  chrom, pos, rsid, gene_symbol, clinvar_significance, clinvar_diseases

Outputs
-------
Writes to:
  data_processed/type3_position_window/

Main outputs:
  type3_window_summary.csv
  selected_neighborhoods.csv
  per_window/<WINDOW_ID>_selected_variants.csv
  per_window/<WINDOW_ID>_state_scan.csv
  type3_report.txt
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
SOURCE_MODE = "panel_parquet"

# If SOURCE_MODE == "report_csv"
SOURCE_REPORT_FILENAME = "eds_pathogenic_candidates.csv"   # or collagen_pathogenic_candidates.csv

# If SOURCE_MODE == "panel_parquet"
SOURCE_PANEL_FILENAME = "eds_panel_variants.parquet"       # or collagen_panel_variants.parquet

# Optional broad pre-filter before window construction
# Example: "Ehlers-Danlos", "aneurysm", "cardiomyopathy"
DISEASE_KEYWORD = "Ehlers-Danlos"

# Anchor selection
#   Leave [] to auto-select anchors from the filtered source
#   Or specify exact rsIDs (without or with "rs")
ANCHOR_RSID_WHITELIST: List[str] = []

MAX_ANCHORS_TO_SCAN = 30

# Window size (base pairs on each side of anchor)
WINDOW_BP = 50_000

# Only keep rows with significance bucket at or above this level.
#   pathogenic=5, likely_pathogenic=4, conflicting_pathogenic=3, risk_factor=2, vus_or_uncertain=1
MIN_SEVERITY_SCORE = 3

# Neighborhood sizing
MIN_ROWS_PER_WINDOW = 2
MAX_VARIANTS_PER_WINDOW = 8
MAX_STATES_WARNING = 2 ** 12

# Deduplicate heavily overlapping neighborhoods by requiring anchor uniqueness
# on (chrom, window_start, window_end)
DROP_DUPLICATE_WINDOWS = True

# Extra ranking preferences
REQUIRE_NONEMPTY_RSID = True
SORT_BY = ["severity_score", "gene_symbol", "disease_text_len", "rsid"]
SORT_ASC = [False, True, False, True]

# Output location
OUT_DIR = DATA_PROCESSED / "type3_position_window"
PER_WINDOW_DIR = OUT_DIR / "per_window"


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


def safe_window_id(chrom: str, start: int, end: int, anchor_rsid: str) -> str:
    chrom = re.sub(r"[^A-Za-z0-9_]+", "_", str(chrom))
    rs = normalize_rsid(anchor_rsid) or "no_rsid"
    return f"{chrom}_{start}_{end}_anchor_rs{rs}"


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
# Neighborhood construction
# =============================================================================

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["rsid", "gene_symbol", "clinvar_significance", "clinvar_diseases", "chrom", "pos"]:
        if col not in df.columns:
            df[col] = ""

    df["significance_bucket"] = df["clinvar_significance"].apply(normalize_significance)
    df["severity_score"] = df["clinvar_significance"].apply(severity_score)
    df["disease_text_len"] = df["clinvar_diseases"].fillna("").astype(str).str.len()
    df["rsid"] = df["rsid"].apply(normalize_rsid)
    df["chrom"] = df["chrom"].astype(str)

    # pos must be numeric
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    df = df[df["pos"].notna()].copy()
    df["pos"] = df["pos"].astype(int)

    if REQUIRE_NONEMPTY_RSID:
        df = df[df["rsid"].fillna("").astype(str).str.strip() != ""]

    if DISEASE_KEYWORD:
        mask = df["clinvar_diseases"].fillna("").astype(str).str.contains(
            re.escape(DISEASE_KEYWORD), case=False, na=False
        )
        df = df[mask].copy()

    df = df[df["severity_score"] >= MIN_SEVERITY_SCORE].copy()

    return df


def choose_anchor_df(df: pd.DataFrame) -> pd.DataFrame:
    adf = df.copy()

    if ANCHOR_RSID_WHITELIST:
        wanted = {normalize_rsid(x) for x in ANCHOR_RSID_WHITELIST if normalize_rsid(x)}
        adf = adf[adf["rsid"].isin(wanted)].copy()

    adf = adf.sort_values(SORT_BY, ascending=SORT_ASC)
    adf = adf.drop_duplicates(subset=["rsid", "chrom", "pos", "ref", "alt"], keep="first")

    if not ANCHOR_RSID_WHITELIST:
        adf = adf.head(MAX_ANCHORS_TO_SCAN)

    if DROP_DUPLICATE_WINDOWS:
        adf["window_start"] = (adf["pos"] - WINDOW_BP).clip(lower=1)
        adf["window_end"] = adf["pos"] + WINDOW_BP
        adf = adf.drop_duplicates(subset=["chrom", "window_start", "window_end"], keep="first")

    return adf.reset_index(drop=True)


def build_window_neighborhoods(df: pd.DataFrame, anchors: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows = []
    window_to_df: Dict[str, pd.DataFrame] = {}

    for _, a in anchors.iterrows():
        chrom = a["chrom"]
        pos = int(a["pos"])
        start = max(1, pos - WINDOW_BP)
        end = pos + WINDOW_BP
        anchor_rsid = a["rsid"]

        wdf = df[(df["chrom"] == chrom) & (df["pos"] >= start) & (df["pos"] <= end)].copy()

        wdf = wdf.sort_values(SORT_BY, ascending=SORT_ASC)
        wdf = wdf.drop_duplicates(subset=["rsid", "chrom", "pos", "ref", "alt"], keep="first")
        wdf = wdf.head(MAX_VARIANTS_PER_WINDOW).reset_index(drop=True)

        if len(wdf) < MIN_ROWS_PER_WINDOW:
            continue

        window_id = safe_window_id(chrom, start, end, anchor_rsid)
        top_genes = "|".join(wdf["gene_symbol"].fillna("").astype(str).value_counts().index.tolist()[:10])
        top_rsids = "|".join(wdf["rsid"].astype(str).tolist()[:10])

        rows.append({
            "window_id": window_id,
            "chrom": chrom,
            "anchor_pos": pos,
            "anchor_rsid": anchor_rsid,
            "window_start": start,
            "window_end": end,
            "count_rows_selected": int(len(wdf)),
            "estimated_states": int(2 ** len(wdf)),
            "top_genes": top_genes,
            "top_rsids": top_rsids,
        })
        window_to_df[window_id] = wdf

    neighborhood_df = pd.DataFrame(rows).sort_values(
        ["count_rows_selected", "chrom", "anchor_pos"], ascending=[False, True, True]
    ).reset_index(drop=True)

    return neighborhood_df, window_to_df


# =============================================================================
# Reporting
# =============================================================================

def write_report(summary_df: pd.DataFrame, source_df: pd.DataFrame, anchors_df: pd.DataFrame) -> None:
    lines = []
    lines.append("QuantumGenome — Phase 2 Type 3 Position-Window Discovery Engine")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"SOURCE_MODE             : {SOURCE_MODE}")
    lines.append(f"SOURCE_REPORT_FILENAME  : {SOURCE_REPORT_FILENAME}")
    lines.append(f"SOURCE_PANEL_FILENAME   : {SOURCE_PANEL_FILENAME}")
    lines.append(f"DISEASE_KEYWORD         : {DISEASE_KEYWORD}")
    lines.append(f"MIN_SEVERITY_SCORE      : {MIN_SEVERITY_SCORE}")
    lines.append(f"WINDOW_BP               : {WINDOW_BP}")
    lines.append(f"MAX_ANCHORS_TO_SCAN     : {MAX_ANCHORS_TO_SCAN}")
    lines.append(f"MAX_VARIANTS_PER_WINDOW : {MAX_VARIANTS_PER_WINDOW}")
    lines.append("")
    lines.append(f"Filtered source rows    : {len(source_df):,}")
    lines.append(f"Anchors used            : {len(anchors_df):,}")
    lines.append(f"Window neighborhoods    : {len(summary_df):,}")
    lines.append("")

    if len(summary_df) > 0:
        lines.append("Top window neighborhoods by minimum energy")
        lines.append("-" * 78)
        for _, row in summary_df.head(25).iterrows():
            lines.append(
                f"{row['window_id']:<52} "
                f"selected={int(row['count_rows_selected']):>2}  "
                f"states={int(row['estimated_states']):>7,}  "
                f"min_energy={row['min_energy']:>8.3f}"
            )
        lines.append("")

    lines.append("Interpretation:")
    lines.append("- Each anchor variant defines a genomic +/- WINDOW_BP neighborhood.")
    lines.append("- The engine selects up to MAX_VARIANTS_PER_WINDOW variants per window.")
    lines.append("- It then enumerates ALL 2^n states exactly on the Jetson.")
    lines.append("- Lower energy = more disease-loaded combination under the current local model.")
    lines.append("- This is reusable for any disease or panel by changing SOURCE_MODE and DISEASE_KEYWORD.")
    lines.append("")

    (OUT_DIR / "type3_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_WINDOW_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_source_df()
    df = prepare_df(raw_df)

    if df.empty:
        raise ValueError("No rows left after filtering. Loosen DISEASE_KEYWORD or MIN_SEVERITY_SCORE.")

    anchors_df = choose_anchor_df(df)
    if anchors_df.empty:
        raise ValueError("No anchors available under current settings.")

    anchors_df.to_csv(OUT_DIR / "anchors_used.csv", index=False)

    neighborhood_df, window_to_df = build_window_neighborhoods(df, anchors_df)
    if neighborhood_df.empty:
        raise ValueError("No position-window neighborhoods met the current thresholds.")

    neighborhood_df.to_csv(OUT_DIR / "selected_neighborhoods.csv", index=False)

    summary_rows = []

    for _, nrow in neighborhood_df.iterrows():
        window_id = nrow["window_id"]
        wdf = window_to_df[window_id].copy()

        wdf.to_csv(PER_WINDOW_DIR / f"{window_id}_selected_variants.csv", index=False)

        n = len(wdf)
        if 2 ** n > MAX_STATES_WARNING:
            print(f"[WARN] {window_id}: selected {n} variants => {2**n:,} states. Still scanning exactly.")

        states_df = enumerate_states(wdf)
        states_df.to_csv(PER_WINDOW_DIR / f"{window_id}_state_scan.csv", index=False)

        best = states_df.iloc[0]
        summary_rows.append({
            "window_id": window_id,
            "chrom": nrow["chrom"],
            "anchor_pos": int(nrow["anchor_pos"]),
            "anchor_rsid": nrow["anchor_rsid"],
            "window_start": int(nrow["window_start"]),
            "window_end": int(nrow["window_end"]),
            "count_rows_selected": int(nrow["count_rows_selected"]),
            "estimated_states": int(nrow["estimated_states"]),
            "min_energy": float(best["energy"]),
            "top_bitstring": best["bitstring"],
            "top_present_rsids": best["present_rsids"],
            "top_present_genes": best["present_genes"],
            "top_genes": nrow["top_genes"],
            "top_rsids": nrow["top_rsids"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["min_energy", "count_rows_selected"], ascending=[True, False]
    ).reset_index(drop=True)

    summary_df.to_csv(OUT_DIR / "type3_window_summary.csv", index=False)
    write_report(summary_df, df, anchors_df)

    print("\n" + "=" * 78)
    print("QuantumGenome — Phase 2 Type 3 Position-Window Discovery Engine")
    print("=" * 78)
    print(f"Filtered source rows  : {len(df):,}")
    print(f"Anchors used          : {len(anchors_df):,}")
    print(f"Window neighborhoods  : {len(summary_df):,}")
    print("")
    print("Top window neighborhoods by minimum energy:")
    print(summary_df.head(20).to_string(index=False))
    print("")
    print(f"Outputs written to: {OUT_DIR}")
    print("Most important files:")
    print(f"  - {OUT_DIR / 'type3_report.txt'}")
    print(f"  - {OUT_DIR / 'type3_window_summary.csv'}")
    print(f"  - {OUT_DIR / 'anchors_used.csv'}")
    print(f"  - {OUT_DIR / 'selected_neighborhoods.csv'}")
    print(f"  - {PER_WINDOW_DIR}/*_selected_variants.csv")
    print(f"  - {PER_WINDOW_DIR}/*_state_scan.csv")


if __name__ == "__main__":
    main()
