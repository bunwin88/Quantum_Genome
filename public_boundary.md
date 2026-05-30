I’ve been building a genomics discovery project that sits at the intersection of quantum optimization, statistical genetics, and local AI-assisted analysis.

QuantumGenome EDS started as a real-data pipeline that converts variant data into an Ising-style interaction model and uses QAOA-based search to identify low-energy, disease-loaded multi-variant patterns in EDS / collagen biology.

Documented results to date include:
- parsing 4.1M+ ClinVar variants into a working variant knowledge graph
- building EDS and collagen-focused variant universes
- validating h / J disease-load scoring from real clinical annotations
- recovering a disease-driving PLOD1 pattern with lowest-energy bitstring 0001 at -5.0
- identifying a tightly interacting B3GALT6 spondylodysplastic EDS cluster with pairwise J = +1.2 and lowest-energy bitstring 1101 at -4.0
- validating a real chr22 / 1000 Genomes QAOA pilot from genotype matrix -> h/J -> classical spectrum -> measured bitstrings

The project has also expanded into a Jetson-first local discovery stack for same-gene, same-disease, position-window, mixed-biologic, novel-candidate, and epistasis-hypothesis ranking.

[PASTE YOUR GITHUB REPO LINK]
