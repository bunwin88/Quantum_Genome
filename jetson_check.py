import requests
from pathlib import Path
from config_template import CLINVAR_DIR, GENCODE_DIR, PHARMGKB_DIR

def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"[SKIP] {dest.name}")
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"[DONE] {dest}")

def main():
    download_file("https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz", CLINVAR_DIR / "clinvar_GRCh38.vcf.gz")
    download_file("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.basic.annotation.gtf.gz", GENCODE_DIR / "gencode.v49.basic.annotation.gtf.gz")

if __name__ == "__main__":
    main()
