#!/usr/bin/env python3
"""
phase4_novel_candidate_miner.py

Phase 4 — Novel Candidate Miner
--------------------------------
Jetson-first, non-IBM candidate nomination engine.

What it is designed to discover
-------------------------------
Phase 4 is designed to discover weakly labeled or unlabeled variants that
repeatedly sit in the shadow of strong known disease signals.

In plain words:
- Phase 3 found the strongest known neighborhoods.
- Phase 4 asks: what other suspicious variants keep showing up near those hits?

It looks for candidate variants that are:
- in the same gene as strong Phase 3 hits
- in the same genomic window as strong Phase 3 hits
- in the same disease-defined neighborhood as strong Phase 3 hits
- in the same mixed biologic panel as strong Phase 3 hits
- but are NOT already the obvious known pathogenic answers

What it uses
------------
Required:
- data_processed/phase3_master/phase3_master_ranked_hits.csv
- data_processed/phase3_master/phase3_gene_recurrence.csv
- data_processed/phase3_master/phase3_variant_recurrence.csv
- data_processed/variant_kg.parquet

Optional:
- Phase 2 outputs already inform Phase 3, so they do not have to be read again here.

Outputs
-------
Writes to:
  data_processed/phase4_novel/

Files:
  phase4_novel_variant_candidates.csv
  phase4_novel_neighborhood_candidates.csv
  phase4_anchor_contexts.csv
  phase4_report.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, VARIANT_KG_PATH


# =============================================================================
# USER CONFIG
# =============================================================================

OUT_DIR = DATA_PROCESSED / "phase4_novel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE3_DIR = DATA_PROCESSED / "phase3_master"

# How many top Phase 3 hits should act as anchors?
TOP_PHASE3_HITS = 100

# Optional broad filter if you want to keep this phase disease-focused.
# Example: "Ehlers-Danlos", "aneurysm", "cardiomyopathy"
DISEASE_KEYWORD = ""

# Expansion controls
INCLUDE_SAME_GENE = True
INCLUDE_SAME_WINDOW = True
INCLUDE_SAME_DISEASE = True
INCLUDE_SAME_PANEL = True

WINDOW_BP = 50_000

# Candidate labels to promote as novel suspects.
# These are normalized significance buckets.
CANDIDATE_LABELS = [
    "vus_or_uncertain",
    "conflicting_pathogenic",
    "risk_factor",
    "association",
    "other_or_mixed",
]

# Exclude already obvious known answers from candidate outputs
EXCLUDE_ALREADY_PATHOGENIC = True

# Require rsid on candidate variants
REQUIRE_NONEMPTY_RSID = True

# Scoring weights
W_SAME_GENE = 3.0
W_SAME_WINDOW = 3.0
W_SAME_DISEASE = 2.5
W_SAME_PANEL = 2.0
W_ANCHOR_ENGINE_DIVERSITY = 2.0
W_PHASE3_GENE_RECURRENCE = 1.0
W_PHASE3_VARIANT_RECURRENCE = 0.5
W_LABEL_BASE = 1.0

TOP_N_FOR_REPORT = 50


# =============================================================================
# Helpers
# =============================================================================

def safe_str(x) -> str:
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        if x.size == 1:
            return safe_str(x.item())
        return "|".join(sorted({safe_str(v) for v in x.tolist()}))
    if isinstance(x, (list, tuple, set)):
        if len(x) == 0:
            return ""
        if len(x) == 1:
            return safe_str(next(iter(x)))
        return "|".join(sorted({safe_str(v) for v in x}))
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


def split_diseases(val) -> List[str]:
    s = safe_str(val)
    if not s:
        return []
    raw = s.replace("|", ",")
    return [x.strip() for x in raw.split(",") if x.strip()]


def normalize_rsid(val) -> str:
    s = safe_str(val)
    if not s:
        return ""
    return re.sub(r"^rs", "", s, flags=re.I)


def normalize_significance(sig: str) -> str:
    s = safe_str(sig).lower()
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


def normalize_disease_term(term: str) -> str:
    t = safe_str(term)
    if not t:
        return ""
    t = t.replace(" ", "_")
    t = re.sub(r"[\/\\:;,\(\)\[\]\{\}]", "_", t)
    t = re.sub(r"__+", "_", t)
    return t.strip("_")


def candidate_label_base(sig_bucket: str) -> float:
    mapping = {
        "conflicting_pathogenic": 2.5,
        "vus_or_uncertain": 2.0,
        "risk_factor": 1.5,
        "association": 1.0,
        "other_or_mixed": 0.5,
        "drug_response": 0.5,
        "protective": 0.0,
        "likely_benign": 0.0,
        "benign": 0.0,
        "pathogenic": 0.0,
        "likely_pathogenic": 0.0,
        "missing": 0.0,
        "affects": 0.0,
    }
    return mapping.get(sig_bucket, 0.0)


def make_candidate_key(row: pd.Series) -> str:
    chrom = safe_str(row.get("chrom"))
    pos = safe_str(row.get("pos"))
    ref = safe_str(row.get("ref"))
    alt = safe_str(row.get("alt"))
    rsid = normalize_rsid(row.get("rsid"))
    gene = safe_str(row.get("gene_symbol"))
    return f"{chrom}:{pos}:{ref}:{alt}:rs{rsid}:{gene}"


# =============================================================================
# Loaders
# =============================================================================

def load_phase3() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hits_path = PHASE3_DIR / "phase3_master_ranked_hits.csv"
    genes_path = PHASE3_DIR / "phase3_gene_recurrence.csv"
    vars_path = PHASE3_DIR / "phase3_variant_recurrence.csv"

    if not hits_path.exists():
        raise FileNotFoundError(f"Missing required Phase 3 file: {hits_path}")
    if not genes_path.exists():
        raise FileNotFoundError(f"Missing required Phase 3 file: {genes_path}")
    if not vars_path.exists():
        raise FileNotFoundError(f"Missing required Phase 3 file: {vars_path}")

    hits = pd.read_csv(hits_path)
    gene_recur = pd.read_csv(genes_path)
    var_recur = pd.read_csv(vars_path)

    if DISEASE_KEYWORD:
        mask = (
            hits["display_name"].astype(str).str.contains(DISEASE_KEYWORD, case=False, na=False)
            | hits["present_genes"].astype(str).str.contains(DISEASE_KEYWORD, case=False, na=False)
            | hits["source_details"].astype(str).str.contains(DISEASE_KEYWORD, case=False, na=False)
        )
        hits = hits[mask].copy()

    if hits.empty:
        raise ValueError("No Phase 3 hits available after optional DISEASE_KEYWORD filtering.")

    hits = hits.head(TOP_PHASE3_HITS).reset_index(drop=True)
    return hits, gene_recur, var_recur


def load_variant_kg() -> pd.DataFrame:
    if not VARIANT_KG_PATH.exists():
        raise FileNotFoundError(f"variant_kg not found: {VARIANT_KG_PATH}")

    cols = [
        "chrom", "pos", "ref", "alt", "rsid",
        "gene_symbol", "clinvar_significance", "clinvar_diseases"
    ]
    df = pd.read_parquet(VARIANT_KG_PATH, columns=cols)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_str)

    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    df = df[df["pos"].notna()].copy()
    df["pos"] = df["pos"].astype(int)

    df["normalized_rsid"] = df["rsid"].apply(normalize_rsid)
    df["significance_bucket"] = df["clinvar_significance"].apply(normalize_significance)

    if REQUIRE_NONEMPTY_RSID:
        df = df[df["normalized_rsid"] != ""].copy()

    if DISEASE_KEYWORD:
        mask = df["clinvar_diseases"].astype(str).str.contains(DISEASE_KEYWORD, case=False, na=False)
        # Keep both disease-focused rows and weakly labeled rows elsewhere in the same genes/windows later.
        # So do NOT hard-filter the master KG here.
        # pass

    return df.reset_index(drop=True)


# =============================================================================
# Anchor expansion
# =============================================================================

def build_anchor_maps(hits: pd.DataFrame) -> pd.DataFrame:
    hits = hits.copy()

    hits["anchor_hit_id"] = hits.apply(
        lambda r: f"{safe_str(r['engine_type'])}::{safe_str(r['display_name'])}",
        axis=1
    )
    hits["anchor_genes_list"] = hits["present_genes"].apply(split_pipe)
    hits["anchor_rsids_list"] = hits["present_rsids"].apply(lambda x: [normalize_rsid(v) for v in split_pipe(x)])

    # type2 uses display_name as disease term
    def disease_terms_for_hit(row) -> List[str]:
        if safe_str(row["engine_type"]) == "type2_same_disease":
            term = normalize_disease_term(row["display_name"])
            return [term] if term else []
        return []

    hits["anchor_disease_terms"] = hits.apply(disease_terms_for_hit, axis=1)

    # type4 can use genes_requested from source_details
    def panel_genes_for_hit(row) -> List[str]:
        if safe_str(row["engine_type"]) == "type4_mixed_panel":
            return [safe_str(g) for g in split_pipe(row.get("source_details", "")) if safe_str(g)]
        return []

    hits["anchor_panel_genes"] = hits.apply(panel_genes_for_hit, axis=1)
    return hits


def collect_candidate_contexts(
    hits: pd.DataFrame,
    kg: pd.DataFrame,
    gene_recur: pd.DataFrame,
    var_recur: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # recurrence maps
    gene_hit_map = dict(zip(gene_recur["gene_symbol"].astype(str), gene_recur["count_hits"]))
    var_hit_map = dict(zip(var_recur["rsid"].astype(str).apply(normalize_rsid), var_recur["count_hits"]))

    # universes
    candidate_universe = kg[kg["significance_bucket"].isin(CANDIDATE_LABELS)].copy()

    if EXCLUDE_ALREADY_PATHOGENIC:
        candidate_universe = candidate_universe[
            ~candidate_universe["significance_bucket"].isin(["pathogenic", "likely_pathogenic"])
        ].copy()

    # anchor lookup by rsid
    anchor_lookup = (
        kg[kg["normalized_rsid"] != ""]
        .drop_duplicates(subset=["normalized_rsid", "chrom", "pos"], keep="first")
        .copy()
    )

    context_rows = []

    for _, hit in hits.iterrows():
        anchor_hit_id = safe_str(hit["anchor_hit_id"])
        engine_type = safe_str(hit["engine_type"])
        display_name = safe_str(hit["display_name"])
        anchor_genes = [g for g in hit["anchor_genes_list"] if g]
        anchor_rsids = [r for r in hit["anchor_rsids_list"] if r]
        anchor_disease_terms = [d for d in hit["anchor_disease_terms"] if d]
        anchor_panel_genes = [g for g in hit["anchor_panel_genes"] if g]

        # A) same gene
        if INCLUDE_SAME_GENE and anchor_genes:
            sg = candidate_universe[candidate_universe["gene_symbol"].isin(anchor_genes)].copy()
            for _, row in sg.iterrows():
                rsid = row["normalized_rsid"]
                if rsid in anchor_rsids:
                    continue
                context_rows.append({
                    "anchor_hit_id": anchor_hit_id,
                    "anchor_engine_type": engine_type,
                    "anchor_display_name": display_name,
                    "context_type": "same_gene",
                    "anchor_genes": "|".join(sorted(set(anchor_genes))),
                    "anchor_rsids": "|".join(sorted(set(anchor_rsids))),
                    "candidate_key": make_candidate_key(row),
                    "candidate_rsid": rsid,
                    "candidate_gene": safe_str(row["gene_symbol"]),
                    "candidate_chrom": safe_str(row["chrom"]),
                    "candidate_pos": int(row["pos"]),
                    "candidate_ref": safe_str(row["ref"]),
                    "candidate_alt": safe_str(row["alt"]),
                    "candidate_significance": safe_str(row["clinvar_significance"]),
                    "candidate_significance_bucket": safe_str(row["significance_bucket"]),
                    "candidate_diseases": safe_str(row["clinvar_diseases"]),
                })

        # B) same window
        if INCLUDE_SAME_WINDOW and anchor_rsids:
            anchor_coords = anchor_lookup[anchor_lookup["normalized_rsid"].isin(anchor_rsids)].copy()
            for _, arow in anchor_coords.iterrows():
                chrom = safe_str(arow["chrom"])
                pos = int(arow["pos"])
                start = max(1, pos - WINDOW_BP)
                end = pos + WINDOW_BP

                sw = candidate_universe[
                    (candidate_universe["chrom"] == chrom)
                    & (candidate_universe["pos"] >= start)
                    & (candidate_universe["pos"] <= end)
                ].copy()

                for _, row in sw.iterrows():
                    rsid = row["normalized_rsid"]
                    if rsid in anchor_rsids:
                        continue
                    context_rows.append({
                        "anchor_hit_id": anchor_hit_id,
                        "anchor_engine_type": engine_type,
                        "anchor_display_name": display_name,
                        "context_type": "same_window",
                        "anchor_genes": "|".join(sorted(set(anchor_genes))),
                        "anchor_rsids": "|".join(sorted(set(anchor_rsids))),
                        "candidate_key": make_candidate_key(row),
                        "candidate_rsid": rsid,
                        "candidate_gene": safe_str(row["gene_symbol"]),
                        "candidate_chrom": safe_str(row["chrom"]),
                        "candidate_pos": int(row["pos"]),
                        "candidate_ref": safe_str(row["ref"]),
                        "candidate_alt": safe_str(row["alt"]),
                        "candidate_significance": safe_str(row["clinvar_significance"]),
                        "candidate_significance_bucket": safe_str(row["significance_bucket"]),
                        "candidate_diseases": safe_str(row["clinvar_diseases"]),
                    })

        # C) same disease
        if INCLUDE_SAME_DISEASE and anchor_disease_terms:
            for term in anchor_disease_terms:
                term_space = term.replace("_", " ")
                sd = candidate_universe[
                    candidate_universe["clinvar_diseases"].astype(str).str.contains(term_space, case=False, na=False)
                    | candidate_universe["clinvar_diseases"].astype(str).str.contains(term, case=False, na=False)
                ].copy()

                for _, row in sd.iterrows():
                    rsid = row["normalized_rsid"]
                    if rsid in anchor_rsids:
                        continue
                    context_rows.append({
                        "anchor_hit_id": anchor_hit_id,
                        "anchor_engine_type": engine_type,
                        "anchor_display_name": display_name,
                        "context_type": "same_disease",
                        "anchor_genes": "|".join(sorted(set(anchor_genes))),
                        "anchor_rsids": "|".join(sorted(set(anchor_rsids))),
                        "candidate_key": make_candidate_key(row),
                        "candidate_rsid": rsid,
                        "candidate_gene": safe_str(row["gene_symbol"]),
                        "candidate_chrom": safe_str(row["chrom"]),
                        "candidate_pos": int(row["pos"]),
                        "candidate_ref": safe_str(row["ref"]),
                        "candidate_alt": safe_str(row["alt"]),
                        "candidate_significance": safe_str(row["clinvar_significance"]),
                        "candidate_significance_bucket": safe_str(row["significance_bucket"]),
                        "candidate_diseases": safe_str(row["clinvar_diseases"]),
                    })

        # D) same panel
        if INCLUDE_SAME_PANEL and anchor_panel_genes:
            sp = candidate_universe[candidate_universe["gene_symbol"].isin(anchor_panel_genes)].copy()
            for _, row in sp.iterrows():
                rsid = row["normalized_rsid"]
                if rsid in anchor_rsids:
                    continue
                context_rows.append({
                    "anchor_hit_id": anchor_hit_id,
                    "anchor_engine_type": engine_type,
                    "anchor_display_name": display_name,
                    "context_type": "same_panel",
                    "anchor_genes": "|".join(sorted(set(anchor_panel_genes))),
                    "anchor_rsids": "|".join(sorted(set(anchor_rsids))),
                    "candidate_key": make_candidate_key(row),
                    "candidate_rsid": rsid,
                    "candidate_gene": safe_str(row["gene_symbol"]),
                    "candidate_chrom": safe_str(row["chrom"]),
                    "candidate_pos": int(row["pos"]),
                    "candidate_ref": safe_str(row["ref"]),
                    "candidate_alt": safe_str(row["alt"]),
                    "candidate_significance": safe_str(row["clinvar_significance"]),
                    "candidate_significance_bucket": safe_str(row["significance_bucket"]),
                    "candidate_diseases": safe_str(row["clinvar_diseases"]),
                })

    anchor_contexts = pd.DataFrame(context_rows)
    if anchor_contexts.empty:
        raise ValueError("Phase 4 found no candidate contexts. Try loosening settings or increasing TOP_PHASE3_HITS.")

    # Deduplicate exact repeated context rows
    anchor_contexts = anchor_contexts.drop_duplicates(
        subset=["anchor_hit_id", "context_type", "candidate_key"],
        keep="first"
    ).reset_index(drop=True)

    # Aggregate candidate variants
    agg_rows = []
    for candidate_key, g in anchor_contexts.groupby("candidate_key", sort=False):
        first = g.iloc[0]
        context_counts = g["context_type"].value_counts().to_dict()
        anchor_hit_count = g["anchor_hit_id"].nunique()
        anchor_engine_diversity = g["anchor_engine_type"].nunique()

        same_gene_hits = int(context_counts.get("same_gene", 0))
        same_window_hits = int(context_counts.get("same_window", 0))
        same_disease_hits = int(context_counts.get("same_disease", 0))
        same_panel_hits = int(context_counts.get("same_panel", 0))

        cand_gene = safe_str(first["candidate_gene"])
        cand_rsid = normalize_rsid(first["candidate_rsid"])
        cand_bucket = safe_str(first["candidate_significance_bucket"])

        gene_rec = float(gene_hit_map.get(cand_gene, 0))
        var_rec = float(var_hit_map.get(cand_rsid, 0))
        label_base = candidate_label_base(cand_bucket)

        candidate_score = (
            W_SAME_GENE * same_gene_hits
            + W_SAME_WINDOW * same_window_hits
            + W_SAME_DISEASE * same_disease_hits
            + W_SAME_PANEL * same_panel_hits
            + W_ANCHOR_ENGINE_DIVERSITY * anchor_engine_diversity
            + W_PHASE3_GENE_RECURRENCE * gene_rec
            + W_PHASE3_VARIANT_RECURRENCE * var_rec
            + W_LABEL_BASE * label_base
        )

        why_parts = []
        if same_gene_hits:
            why_parts.append(f"same_gene={same_gene_hits}")
        if same_window_hits:
            why_parts.append(f"same_window={same_window_hits}")
        if same_disease_hits:
            why_parts.append(f"same_disease={same_disease_hits}")
        if same_panel_hits:
            why_parts.append(f"same_panel={same_panel_hits}")
        why_parts.append(f"anchor_engine_diversity={anchor_engine_diversity}")
        why_parts.append(f"gene_recurrence={gene_rec:g}")
        why_parts.append(f"variant_recurrence={var_rec:g}")
        why_parts.append(f"label_base={label_base:g}")

        agg_rows.append({
            "candidate_key": candidate_key,
            "rsid": cand_rsid,
            "gene_symbol": cand_gene,
            "chrom": safe_str(first["candidate_chrom"]),
            "pos": int(first["candidate_pos"]),
            "ref": safe_str(first["candidate_ref"]),
            "alt": safe_str(first["candidate_alt"]),
            "clinvar_significance": safe_str(first["candidate_significance"]),
            "significance_bucket": cand_bucket,
            "clinvar_diseases": safe_str(first["candidate_diseases"]),
            "candidate_score": float(candidate_score),
            "anchor_hit_count": int(anchor_hit_count),
            "anchor_engine_diversity": int(anchor_engine_diversity),
            "same_gene_anchor_hits": same_gene_hits,
            "same_window_anchor_hits": same_window_hits,
            "same_disease_anchor_hits": same_disease_hits,
            "same_panel_anchor_hits": same_panel_hits,
            "phase3_gene_recurrence": gene_rec,
            "phase3_variant_recurrence": var_rec,
            "label_base": label_base,
            "anchor_hit_ids": "|".join(sorted(set(g["anchor_hit_id"].astype(str))))[:4000],
            "anchor_display_names": "|".join(sorted(set(g["anchor_display_name"].astype(str))))[:4000],
            "anchor_genes": "|".join(sorted(set(g["anchor_genes"].astype(str))))[:4000],
            "why_flagged": "; ".join(why_parts),
        })

    novel_variants = pd.DataFrame(agg_rows).sort_values(
        ["candidate_score", "anchor_hit_count", "anchor_engine_diversity"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    # Gene/neighborhood candidates
    n_rows = []
    for gene_symbol, g in novel_variants.groupby("gene_symbol", sort=False):
        n_rows.append({
            "neighborhood_gene": safe_str(gene_symbol),
            "candidate_count": int(len(g)),
            "max_candidate_score": float(g["candidate_score"].max()),
            "mean_candidate_score": float(g["candidate_score"].mean()),
            "top_candidate_rsids": "|".join(["rs" + r for r in g["rsid"].astype(str).head(20).tolist() if r])[:2000],
            "top_significance_buckets": "|".join(g["significance_bucket"].value_counts().index.tolist()[:10]),
            "anchor_display_names": "|".join(sorted(set("|".join(g["anchor_display_names"].astype(str)).split("|"))))[:4000],
            "why_interesting": (
                f"candidate_count={len(g)}; "
                f"max_candidate_score={g['candidate_score'].max():.3f}; "
                f"mean_candidate_score={g['candidate_score'].mean():.3f}"
            ),
        })

    novel_neighborhoods = pd.DataFrame(n_rows).sort_values(
        ["max_candidate_score", "candidate_count", "mean_candidate_score"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return novel_variants, novel_neighborhoods, anchor_contexts


# =============================================================================
# Reporting
# =============================================================================

def write_report(hits: pd.DataFrame, novel_variants: pd.DataFrame, novel_neighborhoods: pd.DataFrame) -> None:
    lines = []
    lines.append("QuantumGenome — Phase 4 Novel Candidate Miner")
    lines.append("=" * 78)
    lines.append("")
    lines.append("What this phase is designed to discover:")
    lines.append("- Weakly labeled or unclear variants that repeatedly appear near strong Phase 3 anchors.")
    lines.append("- These are NOT treated as proven discoveries.")
    lines.append("- They are promoted as novel suspects for deeper review.")
    lines.append("")
    lines.append(f"Top Phase 3 anchors used      : {len(hits):,}")
    lines.append(f"Novel variant candidates      : {len(novel_variants):,}")
    lines.append(f"Novel neighborhood candidates : {len(novel_neighborhoods):,}")
    lines.append("")

    lines.append("Top novel variant candidates")
    lines.append("-" * 78)
    for _, row in novel_variants.head(TOP_N_FOR_REPORT).iterrows():
        lines.append(
            f"rs{row['rsid']:<16} "
            f"{row['gene_symbol']:<14} "
            f"score={row['candidate_score']:>8.3f}  "
            f"bucket={row['significance_bucket']:<22} "
            f"anchors={row['anchor_hit_count']:>3}  "
            f"engines={row['anchor_engine_diversity']:>2}"
        )
    lines.append("")

    lines.append("Top novel neighborhood genes")
    lines.append("-" * 78)
    for _, row in novel_neighborhoods.head(25).iterrows():
        lines.append(
            f"{row['neighborhood_gene']:<18} "
            f"candidates={int(row['candidate_count']):>4}  "
            f"max_score={row['max_candidate_score']:>8.3f}  "
            f"mean_score={row['mean_candidate_score']:>8.3f}"
        )
    lines.append("")
    lines.append("How to use this:")
    lines.append("1. Start with phase4_novel_variant_candidates.csv")
    lines.append("2. Look for variants with high candidate_score that are not already clearly pathogenic")
    lines.append("3. Check why_flagged and the anchor contexts")
    lines.append("4. Promote the strongest genes/windows/panels into deeper local review")
    lines.append("5. Later, enrich with dbSNP/gnomAD/dbVar/sample VCFs for broader novelty")
    lines.append("")

    (OUT_DIR / "phase4_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    hits, gene_recur, var_recur = load_phase3()
    hits = build_anchor_maps(hits)
    kg = load_variant_kg()

    novel_variants, novel_neighborhoods, anchor_contexts = collect_candidate_contexts(
        hits=hits,
        kg=kg,
        gene_recur=gene_recur,
        var_recur=var_recur,
    )

    # Save
    novel_variants.to_csv(OUT_DIR / "phase4_novel_variant_candidates.csv", index=False)
    novel_neighborhoods.to_csv(OUT_DIR / "phase4_novel_neighborhood_candidates.csv", index=False)
    anchor_contexts.to_csv(OUT_DIR / "phase4_anchor_contexts.csv", index=False)

    write_report(hits, novel_variants, novel_neighborhoods)

    print("\n" + "=" * 78)
    print("QuantumGenome — Phase 4 Novel Candidate Miner")
    print("=" * 78)
    print(f"Top Phase 3 anchors used      : {len(hits):,}")
    print(f"Novel variant candidates      : {len(novel_variants):,}")
    print(f"Novel neighborhood candidates : {len(novel_neighborhoods):,}")
    print("")
    print("Top novel variant candidates:")
    cols = [
        "rsid", "gene_symbol", "candidate_score", "significance_bucket",
        "anchor_hit_count", "anchor_engine_diversity", "why_flagged"
    ]
    print(novel_variants[cols].head(20).to_string(index=False))
    print("")
    print(f"Outputs written to: {OUT_DIR}")
    print("Most important files:")
    print(f"  - {OUT_DIR / 'phase4_report.txt'}")
    print(f"  - {OUT_DIR / 'phase4_novel_variant_candidates.csv'}")
    print(f"  - {OUT_DIR / 'phase4_novel_neighborhood_candidates.csv'}")
    print(f"  - {OUT_DIR / 'phase4_anchor_contexts.csv'}")


if __name__ == "__main__":
    main()
