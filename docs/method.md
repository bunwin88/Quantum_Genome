# Method

## QuantumGenome EDS branch

1. Download ClinVar and GENCODE
2. Parse ClinVar into a Variant Knowledge Graph
3. Build EDS and collagen panels
4. Derive h[i] from ClinVar significance
5. Derive J[i,j] from shared disease annotations
6. Build an Ising Hamiltonian
7. Run QAOA on simulator first
8. Prepare selected experiments for IBM Quantum submission

## QuantumDiscovery chr22 branch

1. Download real chr22 data from 1000 Genomes
2. Build genotype matrices across real people
3. Derive h/J from observed variation patterns
4. Compute exact classical energy landscapes
5. Run QAOA on selected variant subsets
6. Compare sampled bitstrings to classical low-energy states

## Jetson-first expansion

Later phases extended the project into same-gene, same-disease, position-window, mixed-biologic, novel-candidate, and epistasis-hypothesis engines to rank recurrent discovery signals locally before any future hardware comparison.
