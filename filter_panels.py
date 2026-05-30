from pathlib import Path
from typing import List
import pandas as pd
import vcfpy
from config_template import CLINVAR_DIR, PHARMGKB_DIR, VARIANT_KG_PATH, DATA_PROCESSED

def parse_clinvar_vcf(vcf_path: Path) -> pd.DataFrame:
    records: List[dict] = []
    reader = vcfpy.Reader.from_path(str(vcf_path))
    for rec in reader:
        info = rec.INFO
        geneinfo = info.get("GENEINFO")
        if isinstance(geneinfo, list):
            geneinfo = ",".join(str(x) for x in geneinfo)
        gene_symbol = str(geneinfo).split("|")[0].split(":")[0] if geneinfo else None
        clnsig = info.get("CLNSIG")
        if isinstance(clnsig, list):
            clnsig = ",".join(str(x) for x in clnsig)
        clndn = info.get("CLNDN")
        if isinstance(clndn, list):
            clndn = ",".join(str(x) for x in clndn)
        rsid = rec.ID if rec.ID else None
        alts = rec.ALT or [None]
        for alt in alts:
            alt_val = getattr(alt, "value", str(alt)) if alt is not None else None
            records.append({
                "chrom": rec.CHROM, "pos": rec.POS, "ref": rec.REF, "alt": alt_val,
                "rsid": rsid, "gene_symbol": gene_symbol,
                "clinvar_significance": clnsig, "clinvar_diseases": clndn,
            })
    return pd.DataFrame.from_records(records)

def main():
    clinvar_vcf = CLINVAR_DIR / "clinvar_GRCh38.vcf.gz"
    if not clinvar_vcf.exists():
        raise FileNotFoundError("Run etl_download.py first.")
    df = parse_clinvar_vcf(clinvar_vcf)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(VARIANT_KG_PATH, index=False)
    print(f"Saved {len(df):,} rows to {VARIANT_KG_PATH}")

if __name__ == "__main__":
    main()
