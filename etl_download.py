from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"

CLINVAR_DIR = DATA_RAW / "clinvar"
GENCODE_DIR = DATA_RAW / "gencode"
PHARMGKB_DIR = DATA_RAW / "pharmgkb"

VARIANT_KG_PATH = DATA_PROCESSED / "variant_kg.parquet"
PANELS_DIR = DATA_PROCESSED / "panels"
QUANTUM_OUT = PROJECT_ROOT / "quantum_outputs"

# Copy this file to config.py locally, then fill in your own IBM values.
# Do NOT commit real credentials.

IBM_TOKEN = "YOUR_REAL_IBM_API_KEY"
IBM_INSTANCE = "YOUR_OPEN_INSTANCE_CRN"
IBM_BACKEND = "ibm_torino"  # or ibm_fez / ibm_marrakesh

COLLAGEN_GENES = [
    "COL1A1", "COL1A2",
    "COL3A1",
    "COL5A1", "COL5A2",
    "COL11A1", "COL11A2",
    "TNXB",
    "PLOD1",
    "FKBP14",
    "B3GALT6",
    "B4GALT7",
    "SLC39A13",
    "ADAMTS2",
]

for _p in [DATA_RAW, DATA_PROCESSED, CLINVAR_DIR, GENCODE_DIR, PHARMGKB_DIR, PANELS_DIR, QUANTUM_OUT]:
    _p.mkdir(parents=True, exist_ok=True)
