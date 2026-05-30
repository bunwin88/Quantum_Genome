from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from config_template import PANELS_DIR

PANEL_NAME = "eds_panel_variants.parquet"
DISEASE_FILTER = "Ehlers-Danlos"
MAX_VARIANTS = 4
QAOA_P = 2
SHOTS = 2000

def _score_significance(sig: str) -> float:
    if not isinstance(sig, str):
        return 0.0
    s = sig.lower()
    if "pathogenic" in s and "benign" not in s:
        return 1.5 if "likely" in s else 2.0
    if "conflicting" in s and "pathogenic" in s:
        return 1.5
    if "risk factor" in s:
        return 1.0
    if "benign" in s and "pathogenic" not in s:
        return -0.5 if "likely" in s else -1.0
    return 0.0

def _split_diseases(val) -> set[str]:
    if not isinstance(val, str) or not val:
        return set()
    raw = str(val).replace("|", ",")
    return {x.strip() for x in raw.split(",") if x.strip() and "not_provided" not in x.lower()}

def build_h(df: pd.DataFrame) -> Dict[int, float]:
    return {i: _score_significance(row.get("clinvar_significance")) for i, row in df.iterrows()}

def build_J(df: pd.DataFrame, h: Dict[int, float]) -> Dict[Tuple[int, int], float]:
    J = {}
    n = len(df)
    dis = [_split_diseases(df.loc[i, "clinvar_diseases"]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            val = 0.0
            if dis[i] & dis[j]:
                val += 0.8
            if h.get(i, 0) > 0.5 and h.get(j, 0) > 0.5:
                val += 0.4
            if val:
                J[(i, j)] = val
    return J

def build_ising_hamiltonian(num_qubits: int, h: Dict[int, float], J: Dict[Tuple[int, int], float]) -> SparsePauliOp:
    paulis, coeffs = [], []
    for i, h_i in h.items():
        if abs(h_i) < 1e-9:
            continue
        label = ["I"] * num_qubits
        label[num_qubits - 1 - i] = "Z"
        paulis.append("".join(label))
        coeffs.append(h_i)
    for (i, j), J_ij in J.items():
        label = ["I"] * num_qubits
        label[num_qubits - 1 - i] = "Z"
        label[num_qubits - 1 - j] = "Z"
        paulis.append("".join(label))
        coeffs.append(J_ij)
    return SparsePauliOp(paulis, np.array(coeffs, dtype=float))

def build_qaoa_circuit(num_qubits: int, p: int, H: SparsePauliOp) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    gammas = [0.8] * p
    betas = [0.7] * p
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        for label, coeff in zip(H.paulis.to_labels(), H.coeffs):
            c = float(np.real(coeff))
            z_qubits = [num_qubits - 1 - idx for idx, pauli in enumerate(label[::-1]) if pauli == "Z"]
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

def main():
    df = pd.read_parquet(PANELS_DIR / PANEL_NAME)
    df = df[df["clinvar_diseases"].fillna("").str.contains(DISEASE_FILTER, case=False, na=False)].head(MAX_VARIANTS).reset_index(drop=True)
    h = build_h(df)
    J = build_J(df, h)
    H = build_ising_hamiltonian(len(df), h, J)
    qc = build_qaoa_circuit(len(df), QAOA_P, H)
    backend = AerSimulator()
    counts = backend.run(transpile(qc, backend), shots=SHOTS).result().get_counts()
    print(df[["rsid","gene_symbol","clinvar_significance","clinvar_diseases"]].to_string(index=False))
    print(counts)

if __name__ == "__main__":
    main()
