from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    dirs = [
        root / "data_raw" / "clinvar",
        root / "data_raw" / "gencode",
        root / "data_raw" / "pharmgkb",
        root / "data_processed" / "panels",
        root / "quantum_outputs" / "logs",
        root / "quantum_outputs" / "qpu_results",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        keeper = d / ".gitkeep"
        if not keeper.exists():
            keeper.write_text("")
    print("Setup complete.")

if __name__ == "__main__":
    main()
