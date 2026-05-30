#!/usr/bin/env python3
"""
phase3_master_discovery_ranker.py

Phase 3 — Master discovery ranker
---------------------------------
Jetson-first, non-IBM meta-ranking engine that combines results from:

  - Phase 2 Type 1: same-gene neighborhoods
  - Phase 2 Type 2: same-disease neighborhoods
  - Phase 2 Type 3: position-window neighborhoods
  - Phase 2 Type 4: mixed biologic panels

What it does
------------
1. Loads the summary outputs from any/all Phase 2 engines that exist.
2. Normalizes them into one shared result schema.
3. Builds recurrence tables for:
   - variants (rsIDs)
   - genes
   - neighborhood fingerprints
4. Computes a reusable Phase 3 priority score using:
   - minimum energy
   - number of present variants in the top state
   - recurrence across engine types
   - gene diversity
   - variant diversity
5. Produces a master ranked list of candidate discovery neighborhoods.
6. Writes a text report and reusable CSV outputs.

Why this matters
----------------
Phase 2 gives you four different ways to discover neighborhoods.
Phase 3 tells you which findings are strongest overall.

This is the first place where you can say:
  "Which genes / rsIDs / neighborhoods keep showing up as high-interest
   across different discovery lenses?"

That is exactly the right bridge before:
  - expanding to new disease panels
  - adding gnomAD/dbSNP/dbVar later
  - eventually comparing with a quantum-style run if desired

Outputs
-------
Writes to:
  data_processed/phase3_master/

Main outputs:
  phase3_master_ranked_hits.csv
  phase3_variant_recurrence.csv
  phase3_gene_recurrence.csv
  phase3_fingerprint_recurrence.csv
  phase3_report.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import pandas as pd

from config import DATA_PROCESSED


# =============================================================================
# USER CONFIG
# =============================================================================

PHASE3_DIR = DATA_PROCESSED / "phase3_master"
PHASE3_DIR.mkdir(parents=True, exist_ok=True)

TYPE1_DIR = DATA_PROCESSED / "type1_same_gene"
TYPE2_DIR = DATA_PROCESSED / "type2_same_disease"
TYPE3_DIR = DATA_PROCESSED / "type3_position_window"
TYPE4_DIR = DATA_PROCESSED / "type4_mixed_panel"

INCLUDE_TYPE1 = True
INCLUDE_TYPE2 = True
INCLUDE_TYPE3 = True
INCLUDE_TYPE4 = True

# Optional substring filter over normalized result names / genes / top present genes.
# Example: "Ehlers", "COL3A1", "aneurysm"
FOCUS_KEYWORD = ""

# Score weights
W_ENERGY = 4.0
W_ENGINE_RECURRENCE = 2.0
W_GENE_RECURRENCE = 1.5
W_VARIANT_RECURRENCE = 1.5
W_GENE_DIVERSITY = 0.5
W_VARIANT_COUNT = 0.5

TOP_N_FOR_REPORT = 50


# =============================================================================
# Helpers
# =============================================================================

def safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def split_pipe(val) -> List[str]:
    s = safe_str(val)
    if not s:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


def normalize_rsid_token(x: str) -> str:
    s = safe_str(x)
    if not s:
        return ""
    if s.lower().startswith("rs"):
        s = s[2:]
    return s


def fingerprint_from_lists(genes: List[str], rsids: List[str]) -> str:
    g = sorted({safe_str(x) for x in genes if safe_str(x)})
    r = sorted({normalize_rsid_token(x) for x in rsids if normalize_rsid_token(x)})
    return f"GENES:{'|'.join(g)}__RSIDS:{'|'.join(r)}"


def energy_to_score(min_energy: float) -> float:
    """
    More negative energy => higher score.
    We convert by negating it.
    """
    try:
        return -float(min_energy)
    except Exception:
        return 0.0


def maybe_read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


# =============================================================================
# Loaders for each Phase 2 engine
# =============================================================================

def load_type1() -> pd.DataFrame:
    path = TYPE1_DIR / "type1_gene_summary.csv"
    df = maybe_read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "engine_type": "type1_same_gene",
        "neighborhood_id": df["gene_symbol"].astype(str),
        "display_name": df["gene_symbol"].astype(str),
        "min_energy": pd.to_numeric(df["min_energy"], errors="coerce").fillna(0.0),
        "top_bitstring": df["top_bitstring"].astype(str),
        "present_rsids": df["top_present_rsids"].fillna("").astype(str),
        "present_genes": df["top_present_genes"].fillna("").astype(str),
        "count_rows_selected": pd.to_numeric(df["count_rows_selected"], errors="coerce").fillna(0).astype(int),
        "estimated_states": pd.to_numeric(df["estimated_states"], errors="coerce").fillna(0).astype(int),
        "source_details": df["top_significance_buckets"].fillna("").astype(str),
    })
    return out


def load_type2() -> pd.DataFrame:
    path = TYPE2_DIR / "type2_disease_summary.csv"
    df = maybe_read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "engine_type": "type2_same_disease",
        "neighborhood_id": df["normalized_disease_term"].astype(str),
        "display_name": df["normalized_disease_term"].astype(str),
        "min_energy": pd.to_numeric(df["min_energy"], errors="coerce").fillna(0.0),
        "top_bitstring": df["top_bitstring"].astype(str),
        "present_rsids": df["top_present_rsids"].fillna("").astype(str),
        "present_genes": df["top_present_genes"].fillna("").astype(str),
        "count_rows_selected": pd.to_numeric(df["count_rows_selected"], errors="coerce").fillna(0).astype(int),
        "estimated_states": pd.to_numeric(df["estimated_states"], errors="coerce").fillna(0).astype(int),
        "source_details": df["top_genes"].fillna("").astype(str),
    })
    return out


def load_type3() -> pd.DataFrame:
    path = TYPE3_DIR / "type3_window_summary.csv"
    df = maybe_read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "engine_type": "type3_position_window",
        "neighborhood_id": df["window_id"].astype(str),
        "display_name": df["window_id"].astype(str),
        "min_energy": pd.to_numeric(df["min_energy"], errors="coerce").fillna(0.0),
        "top_bitstring": df["top_bitstring"].astype(str),
        "present_rsids": df["top_present_rsids"].fillna("").astype(str),
        "present_genes": df["top_present_genes"].fillna("").astype(str),
        "count_rows_selected": pd.to_numeric(df["count_rows_selected"], errors="coerce").fillna(0).astype(int),
        "estimated_states": pd.to_numeric(df["estimated_states"], errors="coerce").fillna(0).astype(int),
        "source_details": (
            "chrom=" + df["chrom"].astype(str)
            + "|anchor_rsid=" + df["anchor_rsid"].astype(str)
            + "|anchor_pos=" + df["anchor_pos"].astype(str)
        ),
    })
    return out


def load_type4() -> pd.DataFrame:
    path = TYPE4_DIR / "type4_panel_summary.csv"
    df = maybe_read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "engine_type": "type4_mixed_panel",
        "neighborhood_id": df["panel_name"].astype(str),
        "display_name": df["panel_name"].astype(str),
        "min_energy": pd.to_numeric(df["min_energy"], errors="coerce").fillna(0.0),
        "top_bitstring": df["top_bitstring"].astype(str),
        "present_rsids": df["top_present_rsids"].fillna("").astype(str),
        "present_genes": df["top_present_genes"].fillna("").astype(str),
        "count_rows_selected": pd.to_numeric(df["count_rows_selected"], errors="coerce").fillna(0).astype(int),
        "estimated_states": pd.to_numeric(df["estimated_states"], errors="coerce").fillna(0).astype(int),
        "source_details": df["genes_requested"].fillna("").astype(str),
    })
    return out


# =============================================================================
# Phase 3 ranking
# =============================================================================

def build_master_df() -> pd.DataFrame:
    frames = []
    if INCLUDE_TYPE1:
        frames.append(load_type1())
    if INCLUDE_TYPE2:
        frames.append(load_type2())
    if INCLUDE_TYPE3:
        frames.append(load_type3())
    if INCLUDE_TYPE4:
        frames.append(load_type4())

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise FileNotFoundError(
            "No Phase 2 summary files were found. Run one or more Phase 2 engines first."
        )

    df = pd.concat(frames, ignore_index=True)

    if FOCUS_KEYWORD:
        mask = (
            df["display_name"].astype(str).str.contains(FOCUS_KEYWORD, case=False, na=False)
            | df["present_genes"].astype(str).str.contains(FOCUS_KEYWORD, case=False, na=False)
            | df["source_details"].astype(str).str.contains(FOCUS_KEYWORD, case=False, na=False)
        )
        df = df[mask].copy()

    if df.empty:
        raise ValueError("No Phase 3 rows left after applying FOCUS_KEYWORD filter.")

    df["present_rsids_list"] = df["present_rsids"].apply(split_pipe)
    df["present_genes_list"] = df["present_genes"].apply(split_pipe)
    df["fingerprint"] = df.apply(
        lambda r: fingerprint_from_lists(r["present_genes_list"], r["present_rsids_list"]), axis=1
    )
    df["energy_score"] = df["min_energy"].apply(energy_to_score)
    df["variant_count"] = df["present_rsids_list"].apply(lambda x: len([v for v in x if v]))
    df["gene_diversity"] = df["present_genes_list"].apply(lambda x: len(sorted(set([g for g in x if g]))))

    return df.reset_index(drop=True)


def build_recurrence_tables(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Variant recurrence
    variant_rows = []
    for _, row in master_df.iterrows():
        for rsid in row["present_rsids_list"]:
            rsid = normalize_rsid_token(rsid)
            if not rsid:
                continue
            variant_rows.append({
                "rsid": rsid,
                "engine_type": row["engine_type"],
                "display_name": row["display_name"],
                "min_energy": row["min_energy"],
            })
    variant_df = pd.DataFrame(variant_rows)
    if len(variant_df) > 0:
        variant_recurrence = (
            variant_df.groupby("rsid")
            .agg(
                count_hits=("rsid", "size"),
                engine_types=("engine_type", lambda x: "|".join(sorted(set(map(str, x))))),
                neighborhoods=("display_name", lambda x: "|".join(sorted(set(map(str, x))))[:2000]),
                best_energy=("min_energy", "min"),
            )
            .reset_index()
            .sort_values(["count_hits", "best_energy"], ascending=[False, True])
            .reset_index(drop=True)
        )
    else:
        variant_recurrence = pd.DataFrame(columns=["rsid", "count_hits", "engine_types", "neighborhoods", "best_energy"])

    # Gene recurrence
    gene_rows = []
    for _, row in master_df.iterrows():
        for gene in row["present_genes_list"]:
            gene = safe_str(gene)
            if not gene:
                continue
            gene_rows.append({
                "gene_symbol": gene,
                "engine_type": row["engine_type"],
                "display_name": row["display_name"],
                "min_energy": row["min_energy"],
            })
    gene_df = pd.DataFrame(gene_rows)
    if len(gene_df) > 0:
        gene_recurrence = (
            gene_df.groupby("gene_symbol")
            .agg(
                count_hits=("gene_symbol", "size"),
                engine_types=("engine_type", lambda x: "|".join(sorted(set(map(str, x))))),
                neighborhoods=("display_name", lambda x: "|".join(sorted(set(map(str, x))))[:2000]),
                best_energy=("min_energy", "min"),
            )
            .reset_index()
            .sort_values(["count_hits", "best_energy"], ascending=[False, True])
            .reset_index(drop=True)
        )
    else:
        gene_recurrence = pd.DataFrame(columns=["gene_symbol", "count_hits", "engine_types", "neighborhoods", "best_energy"])

    # Fingerprint recurrence
    fp_recurrence = (
        master_df.groupby("fingerprint")
        .agg(
            count_hits=("fingerprint", "size"),
            engine_types=("engine_type", lambda x: "|".join(sorted(set(map(str, x))))),
            neighborhoods=("display_name", lambda x: "|".join(sorted(set(map(str, x))))[:2000]),
            best_energy=("min_energy", "min"),
            genes=("present_genes", lambda x: "|".join(sorted(set(map(str, x))))[:2000]),
            rsids=("present_rsids", lambda x: "|".join(sorted(set(map(str, x))))[:2000]),
        )
        .reset_index()
        .sort_values(["count_hits", "best_energy"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return variant_recurrence, gene_recurrence, fp_recurrence


def apply_phase3_priority_score(
    master_df: pd.DataFrame,
    variant_recurrence: pd.DataFrame,
    gene_recurrence: pd.DataFrame,
    fp_recurrence: pd.DataFrame,
) -> pd.DataFrame:
    master_df = master_df.copy()

    # Map recurrence
    variant_hit_map = dict(zip(
        variant_recurrence["rsid"],
        variant_recurrence["count_hits"]
    ))
    gene_hit_map = dict(zip(
        gene_recurrence["gene_symbol"],
        gene_recurrence["count_hits"]
    ))
    fp_hit_map = dict(zip(
        fp_recurrence["fingerprint"],
        fp_recurrence["count_hits"]
    ))

    def avg_variant_recurrence(rsids: List[str]) -> float:
        vals = [variant_hit_map.get(normalize_rsid_token(r), 0) for r in rsids if normalize_rsid_token(r)]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def avg_gene_recurrence(genes: List[str]) -> float:
        vals = [gene_hit_map.get(safe_str(g), 0) for g in genes if safe_str(g)]
        return float(sum(vals) / len(vals)) if vals else 0.0

    master_df["fingerprint_recurrence"] = master_df["fingerprint"].map(fp_hit_map).fillna(0).astype(float)
    master_df["avg_variant_recurrence"] = master_df["present_rsids_list"].apply(avg_variant_recurrence)
    master_df["avg_gene_recurrence"] = master_df["present_genes_list"].apply(avg_gene_recurrence)

    master_df["phase3_priority_score"] = (
        W_ENERGY * master_df["energy_score"]
        + W_ENGINE_RECURRENCE * master_df["fingerprint_recurrence"]
        + W_GENE_RECURRENCE * master_df["avg_gene_recurrence"]
        + W_VARIANT_RECURRENCE * master_df["avg_variant_recurrence"]
        + W_GENE_DIVERSITY * master_df["gene_diversity"]
        + W_VARIANT_COUNT * master_df["variant_count"]
    )

    master_df = master_df.sort_values(
        ["phase3_priority_score", "min_energy", "variant_count"],
        ascending=[False, True, False]
    ).reset_index(drop=True)

    return master_df


def write_report(master_ranked_df: pd.DataFrame, variant_recurrence: pd.DataFrame, gene_recurrence: pd.DataFrame) -> None:
    lines = []
    lines.append("QuantumGenome — Phase 3 Master Discovery Ranker")
    lines.append("=" * 78)
    lines.append("")
    lines.append("What this report is:")
    lines.append("- A cross-engine ranking of the strongest neighborhoods discovered in Phase 2.")
    lines.append("- It combines same-gene, same-disease, position-window, and mixed-panel results.")
    lines.append("- Higher phase3_priority_score means the hit is stronger under the current local framework.")
    lines.append("")
    lines.append(f"Total ranked hits   : {len(master_ranked_df):,}")
    lines.append(f"Unique variant hits : {len(variant_recurrence):,}")
    lines.append(f"Unique gene hits    : {len(gene_recurrence):,}")
    lines.append("")

    lines.append("Top master-ranked hits")
    lines.append("-" * 78)
    for _, row in master_ranked_df.head(TOP_N_FOR_REPORT).iterrows():
        lines.append(
            f"{row['engine_type']:<22} "
            f"{row['display_name']:<34} "
            f"score={row['phase3_priority_score']:>8.3f}  "
            f"energy={row['min_energy']:>8.3f}  "
            f"genes={row['present_genes']}"
        )
    lines.append("")

    lines.append("Top recurrent genes")
    lines.append("-" * 78)
    for _, row in gene_recurrence.head(25).iterrows():
        lines.append(
            f"{row['gene_symbol']:<18} hits={int(row['count_hits']):>4}  "
            f"best_energy={row['best_energy']:>8.3f}"
        )
    lines.append("")

    lines.append("Top recurrent rsIDs")
    lines.append("-" * 78)
    for _, row in variant_recurrence.head(25).iterrows():
        lines.append(
            f"rs{row['rsid']:<16} hits={int(row['count_hits']):>4}  "
            f"best_energy={row['best_energy']:>8.3f}"
        )
    lines.append("")
    lines.append("How to use this:")
    lines.append("1. Start with phase3_master_ranked_hits.csv")
    lines.append("2. Inspect the top hits and note which engine(s) found them")
    lines.append("3. Check whether the same genes / rsIDs recur across engines")
    lines.append("4. Promote the strongest recurring neighborhoods into deeper local review")
    lines.append("5. Later, only if desired, compare a tiny subset to the quantum-style runner")
    lines.append("")

    (PHASE3_DIR / "phase3_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    master_df = build_master_df()
    variant_recurrence, gene_recurrence, fp_recurrence = build_recurrence_tables(master_df)
    master_ranked_df = apply_phase3_priority_score(master_df, variant_recurrence, gene_recurrence, fp_recurrence)

    # Save
    master_ranked_df.to_csv(PHASE3_DIR / "phase3_master_ranked_hits.csv", index=False)
    variant_recurrence.to_csv(PHASE3_DIR / "phase3_variant_recurrence.csv", index=False)
    gene_recurrence.to_csv(PHASE3_DIR / "phase3_gene_recurrence.csv", index=False)
    fp_recurrence.to_csv(PHASE3_DIR / "phase3_fingerprint_recurrence.csv", index=False)

    write_report(master_ranked_df, variant_recurrence, gene_recurrence)

    print("\n" + "=" * 78)
    print("QuantumGenome — Phase 3 Master Discovery Ranker")
    print("=" * 78)
    print(f"Ranked hits          : {len(master_ranked_df):,}")
    print(f"Unique recurrent rsIDs: {len(variant_recurrence):,}")
    print(f"Unique recurrent genes: {len(gene_recurrence):,}")
    print("")
    print("Top master-ranked hits:")
    cols = [
        "engine_type", "display_name", "phase3_priority_score",
        "min_energy", "present_genes", "present_rsids"
    ]
    print(master_ranked_df[cols].head(20).to_string(index=False))
    print("")
    print(f"Outputs written to: {PHASE3_DIR}")
    print("Most important files:")
    print(f"  - {PHASE3_DIR / 'phase3_report.txt'}")
    print(f"  - {PHASE3_DIR / 'phase3_master_ranked_hits.csv'}")
    print(f"  - {PHASE3_DIR / 'phase3_variant_recurrence.csv'}")
    print(f"  - {PHASE3_DIR / 'phase3_gene_recurrence.csv'}")
    print(f"  - {PHASE3_DIR / 'phase3_fingerprint_recurrence.csv'}")


if __name__ == "__main__":
    main()
