import pandas as pd
from config_template import VARIANT_KG_PATH, PANELS_DIR, COLLAGEN_GENES

def main():
    df = pd.read_parquet(VARIANT_KG_PATH)
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    collagen = df[df["gene_symbol"].isin(COLLAGEN_GENES)].copy()
    collagen.to_parquet(PANELS_DIR / "collagen_panel_variants.parquet", index=False)
    eds = df[df["clinvar_diseases"].fillna("").str.contains("Ehlers-Danlos", case=False, na=False)].copy()
    eds.to_parquet(PANELS_DIR / "eds_panel_variants.parquet", index=False)
    print(f"Collagen panel: {len(collagen):,}")
    print(f"EDS panel: {len(eds):,}")

if __name__ == "__main__":
    main()
