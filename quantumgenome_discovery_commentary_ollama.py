#!/usr/bin/env python3
"""
quantumgenome_discovery_commentary_ollama.py

One-script discovery commentary runner for Jetson using local Ollama.

Purpose
-------
This script reads the current discovery outputs and asks a local Ollama model
for cautious scientific commentary.

Recommended model right now:
- qwen2.5:3b

What it reads
-------------
If present:
- data_processed/phase3_master/phase3_master_ranked_hits.csv
- data_processed/phase4_novel/phase4_novel_variant_candidates.csv
- data_processed/phase5_epistasis/phase5_master_hypotheses.csv
- data_processed/phase5_epistasis/phase5_gene_gene_hypotheses.csv
- data_processed/phase5_epistasis/phase5_variant_variant_hypotheses.csv

What it writes
--------------
- data_processed/discovery_commentary/discovery_commentary_prompt.txt
- data_processed/discovery_commentary/discovery_commentary_raw.txt
- data_processed/discovery_commentary/discovery_commentary_report.md

This script keeps everything local on the Jetson and uses Ollama at:
  http://127.0.0.1:11434
"""

from __future__ import annotations

from pathlib import Path
import json
import requests
import pandas as pd

from config import DATA_PROCESSED


OUT_DIR = DATA_PROCESSED / "discovery_commentary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

TOP_N_PHASE3 = 20
TOP_N_PHASE4 = 20
TOP_N_PHASE5 = 20
TIMEOUT_SECS = 300


def maybe_read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def dataframe_block(title: str, df: pd.DataFrame | None, cols=None, n=20) -> str:
    lines = [f"## {title}"]
    if df is None or df.empty:
        lines.append("No data available.")
        lines.append("")
        return "\n".join(lines)

    if cols:
        use_cols = [c for c in cols if c in df.columns]
        show = df[use_cols].head(n).copy()
    else:
        show = df.head(n).copy()

    lines.append(show.to_csv(index=False))
    lines.append("")
    return "\n".join(lines)


def build_prompt() -> str:
    p3 = maybe_read_csv(DATA_PROCESSED / "phase3_master" / "phase3_master_ranked_hits.csv")
    p4 = maybe_read_csv(DATA_PROCESSED / "phase4_novel" / "phase4_novel_variant_candidates.csv")
    p5m = maybe_read_csv(DATA_PROCESSED / "phase5_epistasis" / "phase5_master_hypotheses.csv")
    p5gg = maybe_read_csv(DATA_PROCESSED / "phase5_epistasis" / "phase5_gene_gene_hypotheses.csv")
    p5vv = maybe_read_csv(DATA_PROCESSED / "phase5_epistasis" / "phase5_variant_variant_hypotheses.csv")

    parts = []
    parts.append(
        "You are a cautious local scientific commentary assistant running on Jetson.\n"
        "You are analyzing EDS/connective-tissue discovery outputs from a local pipeline.\n"
        "Important rules:\n"
        "- Treat all findings as hypotheses, not proof.\n"
        "- Do not claim clinical validity.\n"
        "- Emphasize which candidates are worth deeper review and why.\n"
        "- Highlight recurring genes, recurrent weak-label variants, and plausible interaction themes.\n"
        "- Write clearly for a smart non-specialist, but keep technical detail.\n"
        "- Organize the answer into: 1) strongest findings, 2) likely novel candidates, "
        "3) strongest epistasis hypotheses, 4) what this means for EDS/collagen biology, "
        "5) next best validation steps.\n"
    )

    parts.append(dataframe_block(
        "Phase 3 top ranked hits", p3,
        cols=["engine_type", "display_name", "phase3_priority_score", "min_energy", "present_genes", "present_rsids"],
        n=TOP_N_PHASE3
    ))
    parts.append(dataframe_block(
        "Phase 4 top novel variant candidates", p4,
        cols=["rsid", "gene_symbol", "candidate_score", "significance_bucket", "anchor_hit_count",
              "anchor_engine_diversity", "why_flagged"],
        n=TOP_N_PHASE4
    ))
    parts.append(dataframe_block(
        "Phase 5 top master hypotheses", p5m,
        cols=["hypothesis_type", "display_name", "score", "details"],
        n=TOP_N_PHASE5
    ))
    parts.append(dataframe_block(
        "Phase 5 top gene-gene hypotheses", p5gg,
        cols=["gene_a", "gene_b", "count_hits", "max_hypothesis_score", "mean_hypothesis_score", "context_types"],
        n=TOP_N_PHASE5
    ))
    parts.append(dataframe_block(
        "Phase 5 top variant-variant hypotheses", p5vv,
        cols=["variant_a_rsid", "variant_b_rsid", "count_hits", "max_hypothesis_score", "mean_hypothesis_score", "candidate_genes"],
        n=TOP_N_PHASE5
    ))

    parts.append(
        "Now provide a careful commentary report. Be explicit about uncertainty. "
        "Do not overstate causality. Identify the most actionable next validation steps."
    )
    return "\n".join(parts)


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECS)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def fallback_commentary() -> str:
    return (
        "# Discovery Commentary (Fallback)\n\n"
        "Ollama did not return a response. This usually means the local model was not running "
        "or the request timed out.\n\n"
        "Recommended checks:\n"
        "1. Run `ollama ps`\n"
        "2. Run `ollama run qwen2.5:3b`\n"
        "3. Re-run this script\n"
    )


def main():
    prompt = build_prompt()
    (OUT_DIR / "discovery_commentary_prompt.txt").write_text(prompt, encoding="utf-8")

    try:
        commentary = call_ollama(prompt)
        if not commentary.strip():
            commentary = fallback_commentary()
    except Exception as e:
        commentary = fallback_commentary() + f"\n\nError detail:\n\n{e}\n"

    (OUT_DIR / "discovery_commentary_raw.txt").write_text(commentary, encoding="utf-8")

    report_md = "# QuantumGenome Discovery Commentary\n\n" + commentary
    (OUT_DIR / "discovery_commentary_report.md").write_text(report_md, encoding="utf-8")

    print("=" * 78)
    print("QuantumGenome — Discovery Commentary via Ollama")
    print("=" * 78)
    print(f"Model: {MODEL_NAME}")
    print(f"Output dir: {OUT_DIR}")
    print()
    print("Most important files:")
    print(f"  - {OUT_DIR / 'discovery_commentary_report.md'}")
    print(f"  - {OUT_DIR / 'discovery_commentary_raw.txt'}")
    print(f"  - {OUT_DIR / 'discovery_commentary_prompt.txt'}")


if __name__ == "__main__":
    main()
