"""
Caperucita Pipeline -- Pythia 2.8B (sin Docker)
Equivalente a run_caperucita.sh para Windows sin WSL.

Uso:
    python run_caperucita.py
"""
import subprocess
import sys
import os

def run(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable] + cmd, check=True)
    return result

def main():
    # Asegurar que trabajamos desde el directorio correcto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 60)
    print("  CAPERUCITA PIPELINE -- Pythia 2.8B")
    print("=" * 60)

    # Verificar GPU
    print("\nVerificando GPU...")
    import torch
    assert torch.cuda.is_available(), "ERROR: CUDA no disponible!"
    print(f"  GPU: {torch.cuda.get_device_properties(0).name}")

    # Verificar prerequisitos
    print("\nVerificando prerequisitos...")
    assert os.path.exists("experiment_data.json"), \
        "ERROR: experiment_data.json no encontrado. Ejecuta data_gen.py primero."
    assert os.path.isdir("cavs"), \
        "ERROR: directorio cavs/ no encontrado."
    print("  Prerequisites OK")

    # [1/3] Generar datos del concepto wolf
    run(["gen_wolf_data.py"], "[1/3] Generating wolf concept data...")

    # [2/3] Extraer CAVs (32 capas)
    run(["extract_wolf_cavs.py"], "[2/3] Extracting wolf CAVs (all 32 layers)...")

    # [3/3] Ejecutar caperucita test
    run(["-m", "test.caperucita_test"], "[3/3] Running caperucita test...")

    print("\n" + "=" * 60)
    print("  DONE -- Results in results/caperucita_test_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
