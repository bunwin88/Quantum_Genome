#!/usr/bin/env python3
"""
phase5_epistasis_hypothesis_builder.py

Phase 5 — Epistasis Hypothesis Builder
--------------------------------------
Jetson-first, non-IBM hypothesis engine that turns Phase 3 + Phase 4 outputs into:

- variant-variant interaction hypotheses
- gene-gene interaction hypotheses
- anchor-candidate interaction hypotheses
- one master ranked hypothesis table

Important:
These are HYPOTHESES, not proof. This phase is designed to nominate
interaction candidates worth deeper review.

It is designed to discover:
- which candidate variants from Phase 4 are most plausibly interacting with strong Phase 3 anchors
- which genes keep pairing together across strong anchor/candidate contexts
- which anchor neighborhoods repeatedly attract the same weakly labeled variants
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import math
import re

import pandas as pd

from config import DATA_PROCESSED


# =============================================================================
# USER CONFIG
# =============================================================================

OUT_DIR = DATA_PROCESSED / "phase5_epistasis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE3_DIR = DATA_PROCESSED / "phase3_master"
PHASE4_DIR = DATA_PROCESSED / "phase4_novel"

FOCUS_KEYWORD = ""

TOP_N_VARIANT_VARIANT = 500
TOP_N_GENE_GENE = 500
TOP_N_ANCHOR_CANDIDATE = 1000

# scoring weights
W_PHASE4_CANDIDATE = 4.0
W_ANCHOR_PRIORITY = 3.0
W_CONTEXT_COUNT = 2.0
W_CONTEXT_DIVERSITY = 2.0
W_ENGINE_DIVERSITY = 2.0
W_GENE_RECURRENCE = 1.5
W_VARIANT_RECURRENCE = 1.0

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


def normalize_rsid(val) -> str:
    s = safe_str(val)
    if not s:
        return ""
    return re.sub(r"^rs", "", s, flags=re.I)


def normalize_pair(a: str, b: str) -> Tuple[str, str]:
    a = safe_str(a)
    b = safe_str(b)
    return tuple(sorted([a, b]))


def normalize_display_anchor(engine_type: str, display_name: str) -> str:
    return f"{safe_str(engine_type)}::{safe_str(display_name)}"


def keyword_mask(df: pd.DataFrame) -> pd.Series:
    if not FOCUS_KEYWORD:
        return pd.Series([True] * len(df), index=df.index)
    cols = [c for c in df.columns if df[c].dtype == object]
    mask = pd.Series([False] * len(df), index=df.index)
    for c in cols:
        mask = mask | df[c].astype(str).str.contains(FOCUS_KEYWORD, case=False, na=False)
    return mask


# =============================================================================
# Loaders
# =============================================================================

def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def load_inputs():
    phase3_hits = load_required_csv(PHASE3_DIR / "phase3_master_ranked_hits.csv")
    phase3_gene_recur = load_required_csv(PHASE3_DIR / "phase3_gene_recurrence.csv")
    phase3_var_recur = load_required_csv(PHASE3_DIR / "phase3_variant_recurrence.csv")

    phase4_variants = load_required_csv(PHASE4_DIR / "phase4_novel_variant_candidates.csv")
    phase4_contexts = load_required_csv(PHASE4_DIR / "phase4_anchor_contexts.csv")

    phase3_hits = phase3_hits[keyword_mask(phase3_hits)].copy()
    phase4_variants = phase4_variants[keyword_mask(phase4_variants)].copy()
    phase4_contexts = phase4_contexts[keyword_mask(phase4_contexts)].copy()

    if phase3_hits.empty:
        raise ValueError("No Phase 3 hits available after optional FOCUS_KEYWORD filtering.")
    if phase4_variants.empty:
        raise ValueError("No Phase 4 novel variants available after optional FOCUS_KEYWORD filtering.")
    if phase4_contexts.empty:
        raise ValueError("No Phase 4 anchor contexts available after optional FOCUS_KEYWORD filtering.")

    return phase3_hits, phase3_gene_recur, phase3_var_recur, phase4_variants, phase4_contexts


# =============================================================================
# Builders
# =============================================================================

def build_maps(phase3_hits: pd.DataFrame, phase3_gene_recur: pd.DataFrame, phase3_var_recur: pd.DataFrame):
    hit_map = {}
    max_phase3 = float(phase3_hits["phase3_priority_score"].max()) if "phase3_priority_score" in phase3_hits.columns else 1.0
    if max_phase3 <= 0:
        max_phase3 = 1.0

    for _, row in phase3_hits.iterrows():
        key = normalize_display_anchor(row["engine_type"], row["display_name"])
        hit_map[key] = {
            "phase3_priority_score": float(row.get("phase3_priority_score", 0.0)),
            "phase3_priority_norm": float(row.get("phase3_priority_score", 0.0)) / max_phase3,
            "engine_type": safe_str(row["engine_type"]),
            "display_name": safe_str(row["display_name"]),
            "present_genes": split_pipe(row.get("present_genes", "")),
            "present_rsids": [normalize_rsid(v) for v in split_pipe(row.get("present_rsids", ""))],
        }

    gene_map = {
        safe_str(r["gene_symbol"]): float(r["count_hits"])
        for _, r in phase3_gene_recur.iterrows()
    }
    var_map = {
        normalize_rsid(r["rsid"]): float(r["count_hits"])
        for _, r in phase3_var_recur.iterrows()
    }
    return hit_map, gene_map, var_map


def build_anchor_candidate_hypotheses(
    phase4_variants: pd.DataFrame,
    phase4_contexts: pd.DataFrame,
    hit_map: Dict,
    gene_map: Dict,
    var_map: Dict,
) -> pd.DataFrame:
    variant_score_map = {
        safe_str(r["candidate_key"]): float(r["candidate_score"])
        for _, r in phase4_variants.iterrows()
    }

    rows = []
    grouped = phase4_contexts.groupby(["candidate_key", "anchor_hit_id"], sort=False)

    for (candidate_key, anchor_hit_id), g in grouped:
        first = g.iloc[0]
        hit = hit_map.get(safe_str(anchor_hit_id), None)
        if not hit:
            continue

        candidate_rsid = normalize_rsid(first["candidate_rsid"])
        candidate_gene = safe_str(first["candidate_gene"])

        context_count = int(len(g))
        context_diversity = int(g["context_type"].nunique())
        engine_diversity = int(g["anchor_engine_type"].nunique())

        phase4_candidate_score = float(variant_score_map.get(candidate_key, 0.0))
        anchor_priority_norm = float(hit["phase3_priority_norm"])
        gene_recur = float(gene_map.get(candidate_gene, 0.0))
        var_recur = float(var_map.get(candidate_rsid, 0.0))

        hypothesis_score = (
            W_PHASE4_CANDIDATE * phase4_candidate_score
            + W_ANCHOR_PRIORITY * anchor_priority_norm
            + W_CONTEXT_COUNT * context_count
            + W_CONTEXT_DIVERSITY * context_diversity
            + W_ENGINE_DIVERSITY * engine_diversity
            + W_GENE_RECURRENCE * gene_recur
            + W_VARIANT_RECURRENCE * var_recur
        )

        anchor_genes = sorted(set(split_pipe(first["anchor_genes"])))
        anchor_rsids = sorted(set(normalize_rsid(x) for x in split_pipe(first["anchor_rsids"]) if normalize_rsid(x)))

        rows.append({
            "candidate_key": safe_str(candidate_key),
            "anchor_hit_id": safe_str(anchor_hit_id),
            "anchor_engine_type": hit["engine_type"],
            "anchor_display_name": hit["display_name"],
            "candidate_rsid": candidate_rsid,
            "candidate_gene": candidate_gene,
            "candidate_significance_bucket": safe_str(first["candidate_significance_bucket"]),
            "candidate_diseases": safe_str(first["candidate_diseases"]),
            "anchor_genes": "|".join(anchor_genes),
            "anchor_rsids": "|".join(anchor_rsids),
            "context_types": "|".join(sorted(set(g["context_type"].astype(str)))),
            "context_count": context_count,
            "context_diversity": context_diversity,
            "anchor_engine_diversity": engine_diversity,
            "phase4_candidate_score": phase4_candidate_score,
            "anchor_phase3_priority_norm": anchor_priority_norm,
            "phase3_gene_recurrence": gene_recur,
            "phase3_variant_recurrence": var_recur,
            "hypothesis_score": hypothesis_score,
            "why_hypothesis": (
                f"candidate_score={phase4_candidate_score:.3f}; "
                f"context_count={context_count}; "
                f"context_diversity={context_diversity}; "
                f"anchor_priority_norm={anchor_priority_norm:.3f}; "
                f"gene_recurrence={gene_recur:g}; "
                f"variant_recurrence={var_recur:g}"
            ),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No anchor-candidate hypotheses could be built from Phase 4 contexts.")

    df = df.sort_values(
        ["hypothesis_score", "context_count", "context_diversity"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return df.head(TOP_N_ANCHOR_CANDIDATE).copy()


def build_variant_variant_hypotheses(anchor_candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in anchor_candidate_df.iterrows():
        candidate_rsid = normalize_rsid(row["candidate_rsid"])
        candidate_gene = safe_str(row["candidate_gene"])
        candidate_bucket = safe_str(row["candidate_significance_bucket"])

        for anchor_rsid in split_pipe(row["anchor_rsids"]):
            anchor_rsid = normalize_rsid(anchor_rsid)
            if not anchor_rsid or anchor_rsid == candidate_rsid:
                continue

            rows.append({
                "variant_a_rsid": anchor_rsid,
                "variant_b_rsid": candidate_rsid,
                "anchor_display_name": safe_str(row["anchor_display_name"]),
                "anchor_engine_type": safe_str(row["anchor_engine_type"]),
                "candidate_gene": candidate_gene,
                "candidate_significance_bucket": candidate_bucket,
                "context_types": safe_str(row["context_types"]),
                "anchor_candidate_hypothesis_score": float(row["hypothesis_score"]),
            })

    vv = pd.DataFrame(rows)
    if vv.empty:
        return pd.DataFrame(columns=[
            "variant_a_rsid", "variant_b_rsid", "count_hits", "engine_types", "anchor_display_names",
            "candidate_genes", "candidate_buckets", "context_types", "max_hypothesis_score", "mean_hypothesis_score"
        ])

    out = (
        vv.groupby(["variant_a_rsid", "variant_b_rsid"])
        .agg(
            count_hits=("variant_b_rsid", "size"),
            engine_types=("anchor_engine_type", lambda x: "|".join(sorted(set(map(str, x))))),
            anchor_display_names=("anchor_display_name", lambda x: "|".join(sorted(set(map(str, x))))[:2000]),
            candidate_genes=("candidate_gene", lambda x: "|".join(sorted(set(map(str, x))))[:1000]),
            candidate_buckets=("candidate_significance_bucket", lambda x: "|".join(sorted(set(map(str, x))))),
            context_types=("context_types", lambda x: "|".join(sorted(set(map(str, x))))),
            max_hypothesis_score=("anchor_candidate_hypothesis_score", "max"),
            mean_hypothesis_score=("anchor_candidate_hypothesis_score", "mean"),
        )
        .reset_index()
        .sort_values(["max_hypothesis_score", "count_hits", "mean_hypothesis_score"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    return out.head(TOP_N_VARIANT_VARIANT).copy()


def build_gene_gene_hypotheses(anchor_candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in anchor_candidate_df.iterrows():
        candidate_gene = safe_str(row["candidate_gene"])
        if not candidate_gene:
            continue

        for anchor_gene in split_pipe(row["anchor_genes"]):
            anchor_gene = safe_str(anchor_gene)
            if not anchor_gene or anchor_gene == candidate_gene:
                continue

            g1, g2 = normalize_pair(anchor_gene, candidate_gene)

            rows.append({
                "gene_a": g1,
                "gene_b": g2,
                "anchor_display_name": safe_str(row["anchor_display_name"]),
                "anchor_engine_type": safe_str(row["anchor_engine_type"]),
                "candidate_significance_bucket": safe_str(row["candidate_significance_bucket"]),
                "context_types": safe_str(row["context_types"]),
                "anchor_candidate_hypothesis_score": float(row["hypothesis_score"]),
            })

    gg = pd.DataFrame(rows)
    if gg.empty:
        return pd.DataFrame(columns=[
            "gene_a", "gene_b", "count_hits", "engine_types", "anchor_display_names",
            "candidate_buckets", "context_types", "max_hypothesis_score", "mean_hypothesis_score"
        ])

    out = (
        gg.groupby(["gene_a", "gene_b"])
        .agg(
            count_hits=("gene_b", "size"),
            engine_types=("anchor_engine_type", lambda x: "|".join(sorted(set(map(str, x))))),
            anchor_display_names=("anchor_display_name", lambda x: "|".join(sorted(set(map(str, x))))[:2000]),
            candidate_buckets=("candidate_significance_bucket", lambda x: "|".join(sorted(set(map(str, x))))),
            context_types=("context_types", lambda x: "|".join(sorted(set(map(str, x))))),
            max_hypothesis_score=("anchor_candidate_hypothesis_score", "max"),
            mean_hypothesis_score=("anchor_candidate_hypothesis_score", "mean"),
        )
        .reset_index()
        .sort_values(["max_hypothesis_score", "count_hits", "mean_hypothesis_score"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    return out.head(TOP_N_GENE_GENE).copy()


def build_master_table(anchor_candidate_df: pd.DataFrame, vv_df: pd.DataFrame, gg_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in anchor_candidate_df.head(TOP_N_ANCHOR_CANDIDATE).iterrows():
        rows.append({
            "hypothesis_type": "anchor_candidate",
            "display_name": f"{safe_str(r['anchor_display_name'])} :: rs{safe_str(r['candidate_rsid'])}",
            "score": float(r["hypothesis_score"]),
            "details": f"candidate_gene={safe_str(r['candidate_gene'])}; contexts={safe_str(r['context_types'])}",
        })

    for _, r in vv_df.head(TOP_N_VARIANT_VARIANT).iterrows():
        rows.append({
            "hypothesis_type": "variant_variant",
            "display_name": f"rs{safe_str(r['variant_a_rsid'])} + rs{safe_str(r['variant_b_rsid'])}",
            "score": float(r["max_hypothesis_score"]),
            "details": f"count_hits={int(r['count_hits'])}; candidate_genes={safe_str(r['candidate_genes'])}",
        })

    for _, r in gg_df.head(TOP_N_GENE_GENE).iterrows():
        rows.append({
            "hypothesis_type": "gene_gene",
            "display_name": f"{safe_str(r['gene_a'])} + {safe_str(r['gene_b'])}",
            "score": float(r["max_hypothesis_score"]),
            "details": f"count_hits={int(r['count_hits'])}; contexts={safe_str(r['context_types'])}",
        })

    master = pd.DataFrame(rows)
    if master.empty:
        return pd.DataFrame(columns=["hypothesis_type", "display_name", "score", "details"])

    master = master.sort_values(["score", "hypothesis_type"], ascending=[False, True]).reset_index(drop=True)
    return master


# =============================================================================
# Reporting
# =============================================================================

def write_report(anchor_candidate_df: pd.DataFrame, vv_df: pd.DataFrame, gg_df: pd.DataFrame, master_df: pd.DataFrame) -> None:
    lines = []
    lines.append("QuantumGenome — Phase 5 Epistasis Hypothesis Builder")
    lines.append("=" * 78)
    lines.append("")
    lines.append("What this phase is designed to discover:")
    lines.append("- variant pairs, gene pairs, and anchor-candidate combinations that look like plausible EDS-relevant interactions")
    lines.append("- these are hypothesis-ranked interaction candidates, not proof")
    lines.append("")
    lines.append(f"Anchor-candidate hypotheses : {len(anchor_candidate_df):,}")
    lines.append(f"Variant-variant hypotheses  : {len(vv_df):,}")
    lines.append(f"Gene-gene hypotheses        : {len(gg_df):,}")
    lines.append(f"Master ranked entries       : {len(master_df):,}")
    lines.append("")

    lines.append("Top anchor-candidate interaction hypotheses")
    lines.append("-" * 78)
    for _, r in anchor_candidate_df.head(TOP_N_FOR_REPORT).iterrows():
        lines.append(
            f"{r['anchor_display_name']:<34} "
            f"candidate=rs{safe_str(r['candidate_rsid']):<16} "
            f"gene={safe_str(r['candidate_gene']):<12} "
            f"score={float(r['hypothesis_score']):>8.3f}"
        )
    lines.append("")

    lines.append("Top variant-variant hypotheses")
    lines.append("-" * 78)
    for _, r in vv_df.head(25).iterrows():
        lines.append(
            f"rs{safe_str(r['variant_a_rsid']):<14} + rs{safe_str(r['variant_b_rsid']):<14} "
            f"score={float(r['max_hypothesis_score']):>8.3f}  hits={int(r['count_hits']):>3}"
        )
    lines.append("")

    lines.append("Top gene-gene hypotheses")
    lines.append("-" * 78)
    for _, r in gg_df.head(25).iterrows():
        lines.append(
            f"{safe_str(r['gene_a']):<14} + {safe_str(r['gene_b']):<14} "
            f"score={float(r['max_hypothesis_score']):>8.3f}  hits={int(r['count_hits']):>3}"
        )
    lines.append("")
    lines.append("How to use this:")
    lines.append("1. Start with phase5_master_hypotheses.csv")
    lines.append("2. Separate same-gene vs cross-gene signals")
    lines.append("3. Prioritize candidates involving strong Phase 3 genes and weakly labeled Phase 4 variants")
    lines.append("4. Promote the strongest hypotheses into deeper review / enrichment / genotype testing later")
    lines.append("")

    (OUT_DIR / "phase5_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    phase3_hits, phase3_gene_recur, phase3_var_recur, phase4_variants, phase4_contexts = load_inputs()
    hit_map, gene_map, var_map = build_maps(phase3_hits, phase3_gene_recur, phase3_var_recur)

    anchor_candidate_df = build_anchor_candidate_hypotheses(
        phase4_variants=phase4_variants,
        phase4_contexts=phase4_contexts,
        hit_map=hit_map,
        gene_map=gene_map,
        var_map=var_map,
    )
    vv_df = build_variant_variant_hypotheses(anchor_candidate_df)
    gg_df = build_gene_gene_hypotheses(anchor_candidate_df)
    master_df = build_master_table(anchor_candidate_df, vv_df, gg_df)

    anchor_candidate_df.to_csv(OUT_DIR / "phase5_anchor_candidate_hypotheses.csv", index=False)
    vv_df.to_csv(OUT_DIR / "phase5_variant_variant_hypotheses.csv", index=False)
    gg_df.to_csv(OUT_DIR / "phase5_gene_gene_hypotheses.csv", index=False)
    master_df.to_csv(OUT_DIR / "phase5_master_hypotheses.csv", index=False)

    write_report(anchor_candidate_df, vv_df, gg_df, master_df)

    print("\n" + "=" * 78)
    print("QuantumGenome — Phase 5 Epistasis Hypothesis Builder")
    print("=" * 78)
    print(f"Anchor-candidate hypotheses : {len(anchor_candidate_df):,}")
    print(f"Variant-variant hypotheses  : {len(vv_df):,}")
    print(f"Gene-gene hypotheses        : {len(gg_df):,}")
    print("")
    print("Top master hypotheses:")
    print(master_df.head(20).to_string(index=False))
    print("")
    print(f"Outputs written to: {OUT_DIR}")
    print("Most important files:")
    print(f"  - {OUT_DIR / 'phase5_report.txt'}")
    print(f"  - {OUT_DIR / 'phase5_master_hypotheses.csv'}")
    print(f"  - {OUT_DIR / 'phase5_anchor_candidate_hypotheses.csv'}")
    print(f"  - {OUT_DIR / 'phase5_variant_variant_hypotheses.csv'}")
    print(f"  - {OUT_DIR / 'phase5_gene_gene_hypotheses.csv'}")


if __name__ == "__main__":
    main()
