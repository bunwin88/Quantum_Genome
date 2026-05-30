from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# IMPORTANT:
# 1. Copy config_template.py -> config.py
# 2. Fill in your own IBM_TOKEN / IBM_INSTANCE / IBM_BACKEND
# 3. Keep config.py out of version control
from config import PANELS_DIR, IBM_TOKEN, IBM_INSTANCE, IBM_BACKEND

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
PANEL_NAME = "eds_panel_variants.parquet"      # or "collagen_panel_variants.parquet"
DISEASE_FILTER = "Ehlers-Danlos"
MAX_VARIANTS = 4
QAOA_P = 2
SHOTS = 2000
USE_IBM = False

# Optional manual override, e.g. ["3760089", "3250476"]
RSID_WHITELIST: List[str] = []

# -----------------------------------------------------------------------------
# Load and select variants
# -----------------------------------------------------------------------------
def load_panel(panel_name: str) -> pd.DataFrame:
    path = PANELS_DIR / panel_name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run filter_panels.py first.")
    return pd.read_parquet(path)

def _rsid_to_str(x) -> str:
    if isinstance(x, list) and x:
        return str(x[0])
    return str(x)

def select_variants(
    panel_df: pd.DataFrame,
    max_variants: int = 4,
    disease_substring: str | None = None,
    rsid_whitelist: List[str] | None = None,
) -> pd.DataFrame:
    df = panel_df.dropna(subset=["rsid", "clinvar_significance"]).copy()
    if df.empty:
        raise ValueError("No variants with ClinVar significance found in this panel.")

    if rsid_whitelist:
        df["rsid_str"] = df["rsid"].apply(_rsid_to_str)
        df = df[df["rsid_str"].isin(rsid_whitelist)]
        if df.empty:
            raise ValueError(f"No rows found for rsIDs: {rsid_whitelist}")
        print(f"[Select] Using {len(df.head(max_variants))} variants from rsID whitelist.")
        return df.head(max_variants).reset_index(drop=True)

    if disease_substring:
        mask = df["clinvar_diseases"].fillna("").str.contains(disease_substring, case=False, na=False)
        filtered = df[mask]
        if not filtered.empty:
            df = filtered
        else:
            print(f"[Select] WARNING: No variants matching '{disease_substring}'. Using unfiltered panel.")

    df = df.head(max_variants)
    print(f"[Select] Using {len(df)} real variants for quantum model.")
    return df.reset_index(drop=True)

# -----------------------------------------------------------------------------
# Build h and J from ClinVar annotations
# -----------------------------------------------------------------------------
def _score_significance(sig: str) -> float:
    if not isinstance(sig, str):
        return 0.0
    s = sig.lower()
    if "pathogenic" in s and "benign" not in s:
        return +1.5 if "likely" in s else +2.0
    if "conflicting" in s and "pathogenic" in s:
        return +1.5
    if "risk factor" in s:
        return +1.0
    if "benign" in s and "pathogenic" not in s:
        return -0.5 if "likely" in s else -1.0
    if "uncertain" in s or "vus" in s:
        return 0.0
    return 0.0

def _split_diseases(val) -> set[str]:
    if not isinstance(val, str) or not val:
        return set()
    raw = str(val).replace("|", ",")
    terms = {x.strip() for x in raw.split(",") if x.strip()}
    return {t for t in terms if "not_provided" not in t.lower() and len(t) > 2}

def build_h(subset_df: pd.DataFrame) -> Dict[int, float]:
    h: Dict[int, float] = {}
    for i, row in subset_df.iterrows():
        sig = row.get("clinvar_significance")
        score = _score_significance(sig)
        h[i] = score
        rsid = _rsid_to_str(row["rsid"])
        gene = row.get("gene_symbol", "?")
        print(f"  [h] Variant {i}  rsid={rsid}  gene={gene}  sig='{sig}'  -> h={score:+.3f}")
    return h

def build_J(subset_df: pd.DataFrame, h: Dict[int, float]) -> Dict[Tuple[int, int], float]:
    J: Dict[Tuple[int, int], float] = {}
    n = len(subset_df)
    dis = [_split_diseases(subset_df.loc[i, "clinvar_diseases"]) for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            shared = dis[i] & dis[j]
            val = 0.0
            if shared:
                val += 0.8
            if h.get(i, 0) > 0.5 and h.get(j, 0) > 0.5:
                val += 0.4
            if abs(val) > 0.0:
                J[(i, j)] = val
                print(f"  [J] Variants ({i},{j})  shared_diseases={bool(shared)}  -> J={val:+.3f}  shared={list(shared)[:3]}")

    if not J:
        print("  [J] No pairwise interactions found in this subset.")
    return J

# -----------------------------------------------------------------------------
# Hamiltonian + QAOA
# -----------------------------------------------------------------------------
def build_ising_hamiltonian(
    num_qubits: int,
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
):
    paulis: List[str] = []
    coeffs: List[float] = []

    for i, h_i in h.items():
        if abs(h_i) < 1e-9:
            continue
        label = ["I"] * num_qubits
        label[num_qubits - 1 - i] = "Z"
        paulis.append("".join(label))
        coeffs.append(h_i)

    for (i, j), J_ij in J.items():
        if abs(J_ij) < 1e-9:
            continue
        label = ["I"] * num_qubits
        label[num_qubits - 1 - i] = "Z"
        label[num_qubits - 1 - j] = "Z"
        paulis.append("".join(label))
        coeffs.append(J_ij)

    return SparsePauliOp(paulis, np.array(coeffs, dtype=float))

def build_qaoa_circuit(
    num_qubits: int,
    p: int,
    gammas: List[float],
    betas: List[float],
    H: SparsePauliOp,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]

        for label, coeff in zip(H.paulis.to_labels(), H.coeffs):
            c = float(np.real(coeff))
            if abs(c) == 0:
                continue

            z_qubits = [
                num_qubits - 1 - idx
                for idx, pauli in enumerate(label[::-1])
                if pauli == "Z"
            ]

            if len(z_qubits) == 1:
                qc.rz(2 * gamma * c, z_qubits[0])
            elif len(z_qubits) >= 2:
                first = z_qubits[0]
                for q in z_qubits[1:]:
                    qc.cx(first, q)
                qc.rz(2 * gamma * c, first)
                for q in reversed(z_qubits[1:]):
                    qc.cx(first, q)

        for q in range(num_qubits):
            qc.rx(2 * beta, q)

    qc.measure_all()
    return qc

# -----------------------------------------------------------------------------
# Execution backends
# -----------------------------------------------------------------------------
def run_on_simulator(qc: QuantumCircuit, shots: int = 2000) -> Dict[str, int]:
    backend = AerSimulator()
    transpiled = transpile(qc, backend)
    result = backend.run(transpiled, shots=shots).result()
    return result.get_counts()

def run_on_ibm(qc: QuantumCircuit, shots: int = 1000) -> Dict[str, int]:
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        token=IBM_TOKEN,
        instance=IBM_INSTANCE,
    )
    backend = service.backend(IBM_BACKEND)
    sampler = Sampler(backend)
    job = sampler.run([qc], shots=shots)
    pub_result = job.result()[0]
    counts = pub_result.join_data().get_counts()
    return dict(counts)

# -----------------------------------------------------------------------------
# Energy helper
# -----------------------------------------------------------------------------
def energy_of_bitstring(
    bitstring: str,
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
) -> float:
    spins = np.array([+1 if b == "0" else -1 for b in bitstring], dtype=float)
    E = 0.0
    for i, h_i in h.items():
        E += h_i * spins[i]
    for (i, j), J_ij in J.items():
        E += J_ij * spins[i] * spins[j]
    return float(E)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    panel_df = load_panel(PANEL_NAME)

    subset_df = select_variants(
        panel_df,
        max_variants=MAX_VARIANTS,
        disease_substring=DISEASE_FILTER,
        rsid_whitelist=RSID_WHITELIST if RSID_WHITELIST else None,
    )

    print("\n[Subset variants]")
    print(subset_df[["rsid", "gene_symbol", "clinvar_significance", "clinvar_diseases"]])

    num_snps = len(subset_df)

    h = build_h(subset_df)
    J = build_J(subset_df, h)

    H = build_ising_hamiltonian(num_snps, h, J)
    gammas = [0.8] * QAOA_P
    betas = [0.7] * QAOA_P
    qc = build_qaoa_circuit(num_snps, QAOA_P, gammas, betas, H)

    if USE_IBM:
        print(f"\n[QAOA] Running on IBM backend {IBM_BACKEND}...")
        counts = run_on_ibm(qc, shots=SHOTS)
    else:
        print("\n[QAOA] Running on local simulator first...")
        counts = run_on_simulator(qc, shots=SHOTS)

    print("\n[Top bitstrings]")
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    for bitstring, c in items[:15]:
        E = energy_of_bitstring(bitstring, h, J)
        print(f"{bitstring}  count={c}  energy={E:+.3f}")

if __name__ == "__main__":
    main()
