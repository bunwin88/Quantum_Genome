#!/usr/bin/env python3
"""
phase1_panel_inspector.py

Combines Phase 1 A/B/C into one local Jetson script.

What it does:
1. Reads EDS pathogenic candidates CSV
2. Reads collagen pathogenic candidates CSV
3. Reads the EDS panel parquet
4. Prints and saves:
   - top genes in EDS pathogenic candidates
   - top genes in collagen pathogenic candidates
   - top disease strings in the EDS panel
   - quick panel size summary

Outputs:
  ~/Quantum_Genome/data_processed/reports/phase1_panel_inspector_report.txt
  ~/Quantum_Genome/data_processed/reports/phase1_eds_top_genes.csv
  ~/Quantum_Genome/data_processed/reports/phase1_collagen_top_genes.csv
  ~/Quantum_Genome/data_processed/reports/phase1_eds_top_disease_strings.csv
"""

from pathlib import Path
from collections import Counter
import pandas as pd


def normalize_text(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    if hasattr(x, "tolist"):
        try:
            x = x.tolist()
        except Exception:
            pass
    if isinstance(x, (list, tuple, set)):
        return "|".join(str(v).strip() for v in x if str(v).strip())
    return str(x).strip()


def load_paths() -> dict:
    root = Path.home() / "Quantum_Genome"
    reports_dir = root / "data_processed" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "reports_dir": reports_dir,
        "eds_candidates": reports_dir / "eds_pathogenic_candidates.csv",
        "collagen_candidates": reports_dir / "collagen_pathogenic_candidates.csv",
        "eds_panel": root / "data_processed" / "panels" / "eds_panel_variants.parquet",
    }


def top_gene_counts(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    tmp = df.copy()
    tmp["gene_symbol"] = tmp["gene_symbol"].apply(normalize_text)
    out = (
        tmp["gene_symbol"]
        .replace("", "NA")
        .value_counts()
        .head(n)
        .rename_axis("gene_symbol")
        .reset_index(name="count_rows")
    )
    return out


def top_disease_terms(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    terms = []
    for val in df["clinvar_diseases"].fillna(""):
        txt = normalize_text(val).replace("|", ",")
        parts = [x.strip() for x in txt.split(",") if x.strip()]
        for part in parts:
            if part.lower() not in {"not_provided", "not specified", "na", "none"}:
                terms.append(part)
    c = Counter(terms)
    out = pd.DataFrame(c.most_common(n), columns=["disease_term", "count_rows"])
    return out


def main():
    paths = load_paths()

    missing = [p for k, p in paths.items() if k not in {"root", "reports_dir"} and not p.exists()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print(f"  - {p}")
        print("Run summarize_variant_buckets.py first so the candidate CSVs exist.")
        return

    eds_candidates = pd.read_csv(paths["eds_candidates"])
    collagen_candidates = pd.read_csv(paths["collagen_candidates"])
    eds_panel = pd.read_parquet(paths["eds_panel"])

    eds_gene_counts = top_gene_counts(eds_candidates, n=50)
    collagen_gene_counts = top_gene_counts(collagen_candidates, n=50)
    eds_disease_counts = top_disease_terms(eds_panel, n=50)

    eds_gene_counts.to_csv(paths["reports_dir"] / "phase1_eds_top_genes.csv", index=False)
    collagen_gene_counts.to_csv(paths["reports_dir"] / "phase1_collagen_top_genes.csv", index=False)
    eds_disease_counts.to_csv(paths["reports_dir"] / "phase1_eds_top_disease_strings.csv", index=False)

    report_lines = []
    report_lines.append("QuantumGenome — Phase 1 Panel Inspector")
    report_lines.append("=" * 72)
    report_lines.append("")
    report_lines.append("Panel sizes")
    report_lines.append("-" * 72)
    report_lines.append(f"EDS pathogenic candidates      : {len(eds_candidates):,}")
    report_lines.append(f"Collagen pathogenic candidates : {len(collagen_candidates):,}")
    report_lines.append(f"EDS panel total rows           : {len(eds_panel):,}")
    report_lines.append("")
    report_lines.append("Top genes in EDS pathogenic candidates")
    report_lines.append("-" * 72)
    for _, row in eds_gene_counts.iterrows():
        report_lines.append(f"{row['gene_symbol']:<24} {int(row['count_rows']):>10,}")
    report_lines.append("")
    report_lines.append("Top genes in collagen pathogenic candidates")
    report_lines.append("-" * 72)
    for _, row in collagen_gene_counts.iterrows():
        report_lines.append(f"{row['gene_symbol']:<24} {int(row['count_rows']):>10,}")
    report_lines.append("")
    report_lines.append("Top disease strings in the EDS panel")
    report_lines.append("-" * 72)
    for _, row in eds_disease_counts.iterrows():
        report_lines.append(f"{row['disease_term']:<56} {int(row['count_rows']):>8,}")
    report_lines.append("")
    report_lines.append("What to use this for")
    report_lines.append("-" * 72)
    report_lines.append("1. Identify the strongest genes to use as Type 1 same-gene neighborhoods.")
    report_lines.append("2. Identify dominant disease phrases to use as Type 2 same-disease neighborhoods.")
    report_lines.append("3. Use the gene list + disease phrases to decide which local discovery engine to run first.")

    report_path = paths["reports_dir"] / "phase1_panel_inspector_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 72)
    print("QuantumGenome — Phase 1 Panel Inspector")
    print("=" * 72)
    print(f"EDS pathogenic candidates      : {len(eds_candidates):,}")
    print(f"Collagen pathogenic candidates : {len(collagen_candidates):,}")
    print(f"EDS panel total rows           : {len(eds_panel):,}")
    print("")
    print("Top genes in EDS pathogenic candidates:")
    print(eds_gene_counts.head(20).to_string(index=False))
    print("")
    print("Top genes in collagen pathogenic candidates:")
    print(collagen_gene_counts.head(20).to_string(index=False))
    print("")
    print("Top disease strings in the EDS panel:")
    print(eds_disease_counts.head(20).to_string(index=False))
    print("")
    print(f"Saved report: {report_path}")
    print(f"Saved CSVs in: {paths['reports_dir']}")


if __name__ == "__main__":
    main()
