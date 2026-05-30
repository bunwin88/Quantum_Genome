# QuantumGenome EDS Discovery Engine

QuantumGenome EDS is a real-data genomics discovery project that converts variant data into an Ising-style interaction model and uses QAOA-based search to identify low-energy, disease-loaded multi-variant patterns.

The project began as a quantum-assisted Ehlers-Danlos / collagen discovery pipeline built from ClinVar and later expanded into a Jetson-first local discovery framework with same-gene, same-disease, position-window, mixed-biologic, novel-candidate, and epistasis-hypothesis phases.

## Real Results Achieved

This project has documented end-to-end validation on real data.

- Parsed 4,129,628 ClinVar variants and built disease-specific EDS and collagen panels.
- Validated h[i] scoring from ClinVar significance and J[i,j] scoring from shared disease annotations.
- In a PLOD1 baseline run, the lowest-energy bitstring was `0001` at `-5.0`, correctly isolating the disease-driving variant.
- In a B3GALT6 EDS cluster run, four pathogenic / likely pathogenic variants sharing spondylodysplastic EDS annotation produced pairwise `J[i,j] = +1.2` and a lowest-energy bitstring of `1101` at `-4.0`.
- In a 1000 Genomes chr22 pilot, the full real-data loop was validated: genotype matrix -> h/J -> classical energy spectrum -> QAOA simulator -> measured bitstrings.

## What Is In This Public Repo

This public repo includes runnable Python scripts with **placeholder IBM settings** so another user can:
1. copy `config_template.py` to `config.py`
2. enter their own IBM credentials and backend
3. run the ETL / panel / QAOA workflow themselves

## IBM Quantum Support In This Repo

The file `quantum_model_builder.py` includes:
- local Aer simulator mode
- IBM Quantum submission mode through `QiskitRuntimeService`
- placeholder settings in `config_template.py`
- `USE_IBM = False` by default for safe first runs

To use IBM Quantum:
1. copy `config_template.py` to `config.py`
2. fill in your own `IBM_TOKEN`, `IBM_INSTANCE`, and `IBM_BACKEND`
3. set `USE_IBM = True` in `quantum_model_builder.py`
4. run `python3 quantum_model_builder.py`

## Public / Private Boundary

This repository is publicly visible for portfolio, review, and discussion purposes only.
All rights are reserved by the author unless explicit written permission is granted.
No open-source license is provided for this repository.
