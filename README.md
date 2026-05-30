# QuantumGenome EDS Discovery Engine

QuantumGenome EDS is a real-data genomics discovery project that converts variant data into an Ising-style interaction model and uses QAOA-based search to identify low-energy, disease-loaded multi-variant patterns.

The project began as a quantum-assisted Ehlers-Danlos / collagen discovery pipeline built from ClinVar and later expanded into a Jetson-first local discovery framework with same-gene, same-disease, position-window, mixed-biologic, novel-candidate, and epistasis-hypothesis phases.

## Real Results Achieved

This project has documented end-to-end validation on real data.

- Parsed 4,129,628 ClinVar variants and built disease-specific EDS and collagen panels.
- Validated h[i] scoring from ClinVar significance and J[i,j] scoring from shared disease annotations.
- In a PLOD1 baseline run, the lowest-energy bitstring was 0001 at -5.0, correctly isolating the disease-driving variant.
- In a B3GALT6 EDS cluster run, four pathogenic / likely pathogenic variants sharing spondylodysplastic EDS annotation produced pairwise J[i,j] = +1.2 and a lowest-energy bitstring of 1101 at -4.0.
- In a 1000 Genomes chr22 pilot, the full real-data loop was validated: genotype matrix -> h/J -> classical energy spectrum -> QAOA simulator -> measured bitstrings.

## Why This Matters

Most genomics pipelines score variants one at a time. This project is designed to explore multi-variant interaction structure, epistasis hypotheses, and disease-loaded neighborhoods that may not be visible through single-variant analysis alone.

## Current State

- Real-data quantum / Ising pipeline validated
- Jetson-first local discovery stack built through novel-candidate mining and epistasis hypothesis generation
- Ready for broader evidence enrichment and future genotype-matrix validation

## Repository Contents

- `config_template.py` — public-safe config template
- `setup_quantum_genome.py` — project folder bootstrap
- `etl_download.py` — download raw ClinVar / GENCODE inputs
- `etl_build_variant_table.py` — build variant knowledge graph
- `filter_panels.py` — build EDS / collagen panels
- `quantum_model_builder.py` — Ising + QAOA model builder
- `jetson_check.py` — Jetson diagnostics
- `summarize_variant_buckets.py` — panel and bucket summarizer
- `phase1_panel_inspector.py` — first review layer
- `phase2_type1_same_gene_engine.py` — same-gene discovery
- `phase2_type2_same_disease_engine.py` — same-disease discovery
- `phase2_type3_position_window_engine.py` — position-window discovery
- `phase2_type4_mixed_biologic_panel_engine.py` — mixed biologic panel discovery
- `phase3_master_discovery_ranker.py` — cross-engine ranking
- `phase4_novel_candidate_miner.py` — novel suspect mining
- `phase5_epistasis_hypothesis_builder.py` — interaction / epistasis hypothesis builder
- `quantumgenome_master_pipeline.py` — orchestration script
- `quantumgenome_discovery_commentary_ollama.py` — local Ollama commentary script
- `docs/` — methods, findings, status, GitHub / LinkedIn wording

## Core Workflow

1. Download and parse ClinVar or real genotype data
2. Build focused disease or population variant universes
3. Derive h single-variant fields and J pairwise interaction terms
4. Build an Ising Hamiltonian
5. Run QAOA or exact local scans
6. Rank low-energy disease-loaded configurations
7. Promote novel candidates and epistasis hypotheses for deeper review

## Public / Private Boundary

This public repository is a sanitized project showcase. It does not include private credentials, internal-only operating notes, or confidential trade-secret material.
