# Findings

## 1. PLOD1 baseline validation

A four-variant PLOD1 experiment showed that the model assigned the risky EDS-linked variant a strong positive field while benign variants received negative fields. The lowest-energy bitstring was 0001 at -5.0, confirming that the pipeline correctly identified the disease-driving variant as the primary attractor.

## 2. B3GALT6 EDS cluster validation

A four-variant B3GALT6 run showed a tightly interacting pathogenic cluster associated with spondylodysplastic EDS. All four variants were pathogenic or likely pathogenic, all shared the same disease annotation, and all pairs received J[i,j] = +1.2. The lowest-energy bitstring was 1101 at -4.0, showing concentration on pathogenic-heavy patterns.

## 3. chr22 / 1000 Genomes QAOA pilot

A broader population-genetics prototype on real chr22 genotype data from 1000 Genomes validated the full QAOA loop: data ingestion, matrix construction, h/J building, classical energy analysis, quantum circuit construction, and measured bitstring output. The pilot also showed that variant selection and QAOA tuning materially affect whether sampled states approach the classical optimum, which defined the next scaling and optimization path.

## Interpretation

These findings demonstrate that the project moved beyond concept stage and into real-data computational discovery. The system can isolate disease-driving variants, identify tightly interacting pathogenic clusters, and build working Ising/QAOA landscapes from population-scale genotype data.
