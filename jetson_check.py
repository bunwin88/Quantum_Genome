import sys, os, subprocess, importlib, platform, shutil

def run(cmd, default="unknown"):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or default
    except Exception:
        return default

def check_import(module):
    try:
        mod = importlib.import_module(module)
        return True, str(getattr(mod, "__version__", "installed"))
    except ImportError as e:
        return False, str(e)

def main():
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"JetPack: {run('dpkg -l | grep -i jetpack | awk \'{print $3}\' | head -1')}")
    print(f"CUDA: {run('nvcc --version 2>/dev/null | grep release | awk \'{print $6}\' | tr -d \,')}")
    for pkg in ["numpy", "scipy", "pandas", "pyarrow", "requests", "vcfpy", "qiskit", "qiskit_aer", "qiskit_ibm_runtime"]:
        ok, ver = check_import(pkg)
        print(f"{pkg}: {'OK ' + ver if ok else 'MISSING'}")

if __name__ == "__main__":
    main()
