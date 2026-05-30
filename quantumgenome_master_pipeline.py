#!/usr/bin/env python3
"""
quantumgenome_master_pipeline.py

One-script orchestrator for the QuantumGenome Jetson-first workflow.

Purpose
-------
This script gives you ONE place to run setup and the full phase chain in order.

It can run:
- setup
- download
- ClinVar parse / variant KG build
- panel filtering
- summary scripts
- Phase 1 inspector
- Phase 2 Type 1-4 engines
- Phase 3 master ranker
- Phase 4 novel candidate miner
- Phase 5 epistasis hypothesis builder

How to use
----------
1. Put this script in ~/Quantum_Genome/
2. Make sure the other scripts are also in ~/Quantum_Genome/
3. Activate your venv
4. Run: python3 quantumgenome_master_pipeline.py
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "pipeline_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PY = sys.executable

# -----------------------------------------------------------------------------
# Toggle steps here
# -----------------------------------------------------------------------------
RUN_SETUP = True
RUN_DOWNLOAD = False              # usually False after first successful download
RUN_BUILD_VARIANT_KG = False      # usually False after first successful build
RUN_FILTER_PANELS = False         # usually False after panels already exist
RUN_BUCKET_SUMMARY = False
RUN_PHASE1_INSPECTOR = False
RUN_PHASE2_TYPE1 = False
RUN_PHASE2_TYPE2 = False
RUN_PHASE2_TYPE3 = False
RUN_PHASE2_TYPE4 = False
RUN_PHASE3 = False
RUN_PHASE4 = False
RUN_PHASE5 = False

# Set True to re-run even if output sentinel exists
FORCE_RERUN = False


STEPS = [
    {
        "name": "setup_quantum_genome",
        "enabled_var": "RUN_SETUP",
        "script": "setup_quantum_genome.py",
        "sentinel": ROOT / "data_processed",
    },
    {
        "name": "etl_download",
        "enabled_var": "RUN_DOWNLOAD",
        "script": "etl_download.py",
        "sentinel": ROOT / "data_raw" / "clinvar" / "clinvar_GRCh38.vcf.gz",
    },
    {
        "name": "etl_build_variant_table",
        "enabled_var": "RUN_BUILD_VARIANT_KG",
        "script": "etl_build_variant_table.py",
        "sentinel": ROOT / "data_processed" / "variant_kg.parquet",
    },
    {
        "name": "filter_panels",
        "enabled_var": "RUN_FILTER_PANELS",
        "script": "filter_panels.py",
        "sentinel": ROOT / "data_processed" / "panels" / "eds_panel_variants.parquet",
    },
    {
        "name": "summarize_variant_buckets",
        "enabled_var": "RUN_BUCKET_SUMMARY",
        "script": "summarize_variant_buckets.py",
        "sentinel": ROOT / "data_processed" / "reports" / "bucket_summary_report.txt",
    },
    {
        "name": "phase1_panel_inspector",
        "enabled_var": "RUN_PHASE1_INSPECTOR",
        "script": "phase1_panel_inspector.py",
        "sentinel": ROOT / "data_processed" / "reports" / "phase1_panel_inspector_report.txt",
    },
    {
        "name": "phase2_type1_same_gene_engine",
        "enabled_var": "RUN_PHASE2_TYPE1",
        "script": "phase2_type1_same_gene_engine.py",
        "sentinel": ROOT / "data_processed" / "type1_same_gene" / "type1_gene_summary.csv",
    },
    {
        "name": "phase2_type2_same_disease_engine",
        "enabled_var": "RUN_PHASE2_TYPE2",
        "script": "phase2_type2_same_disease_engine.py",
        "sentinel": ROOT / "data_processed" / "type2_same_disease" / "type2_disease_summary.csv",
    },
    {
        "name": "phase2_type3_position_window_engine",
        "enabled_var": "RUN_PHASE2_TYPE3",
        "script": "phase2_type3_position_window_engine.py",
        "sentinel": ROOT / "data_processed" / "type3_position_window" / "type3_window_summary.csv",
    },
    {
        "name": "phase2_type4_mixed_biologic_panel_engine",
        "enabled_var": "RUN_PHASE2_TYPE4",
        "script": "phase2_type4_mixed_biologic_panel_engine.py",
        "sentinel": ROOT / "data_processed" / "type4_mixed_panel" / "type4_panel_summary.csv",
    },
    {
        "name": "phase3_master_discovery_ranker",
        "enabled_var": "RUN_PHASE3",
        "script": "phase3_master_discovery_ranker.py",
        "sentinel": ROOT / "data_processed" / "phase3_master" / "phase3_master_ranked_hits.csv",
    },
    {
        "name": "phase4_novel_candidate_miner",
        "enabled_var": "RUN_PHASE4",
        "script": "phase4_novel_candidate_miner.py",
        "sentinel": ROOT / "data_processed" / "phase4_novel" / "phase4_novel_variant_candidates.csv",
    },
    {
        "name": "phase5_epistasis_hypothesis_builder",
        "enabled_var": "RUN_PHASE5",
        "script": "phase5_epistasis_hypothesis_builder.py",
        "sentinel": ROOT / "data_processed" / "phase5_epistasis" / "phase5_master_hypotheses.csv",
    },
]


def should_run(step: dict) -> bool:
    enabled = globals()[step["enabled_var"]]
    if not enabled:
        return False
    if FORCE_RERUN:
        return True
    return not Path(step["sentinel"]).exists()


def run_step(step: dict) -> int:
    script_path = ROOT / step["script"]
    if not script_path.exists():
        print(f"[ERROR] Missing script: {script_path}")
        return 1

    log_path = LOG_DIR / f"{step['name']}.log"
    print("=" * 78)
    print(f"RUNNING: {step['name']}")
    print(f"SCRIPT : {script_path}")
    print(f"LOG    : {log_path}")
    print("=" * 78)

    start = time.time()
    proc = subprocess.run(
        [PY, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    log_text = []
    log_text.append(f"STEP: {step['name']}")
    log_text.append(f"SCRIPT: {script_path}")
    log_text.append(f"RETURN_CODE: {proc.returncode}")
    log_text.append(f"ELAPSED_SECONDS: {elapsed:.2f}")
    log_text.append("")
    log_text.append("STDOUT")
    log_text.append("-" * 40)
    log_text.append(proc.stdout or "")
    log_text.append("")
    log_text.append("STDERR")
    log_text.append("-" * 40)
    log_text.append(proc.stderr or "")
    log_path.write_text("\n".join(log_text), encoding="utf-8")

    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)

    print(f"[DONE] {step['name']} return_code={proc.returncode} elapsed={elapsed:.2f}s")
    print()
    return proc.returncode


def main():
    print("=" * 78)
    print("QuantumGenome — Master Pipeline Orchestrator")
    print("=" * 78)
    print(f"Project root: {ROOT}")
    print(f"Python: {PY}")
    print(f"Force rerun: {FORCE_RERUN}")
    print()

    selected = [s for s in STEPS if globals()[s["enabled_var"]]]
    if not selected:
        print("No steps enabled. Edit the toggles at the top of the script.")
        return

    for step in selected:
        if should_run(step):
            rc = run_step(step)
            if rc != 0:
                print(f"[STOP] Pipeline stopped at step: {step['name']}")
                sys.exit(rc)
        else:
            print(f"[SKIP] {step['name']} — sentinel exists and FORCE_RERUN=False")

    print("=" * 78)
    print("Pipeline complete.")
    print(f"Logs written to: {LOG_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
