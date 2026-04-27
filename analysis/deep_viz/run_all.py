"""Run all V1-V7 figures in order."""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
scripts = [
    "concept_graph.py",
    "cav_validity.py",
    "dose_response_heatmap.py",
    "activation_pca.py",
    "trajectory.py",
    "selectivity_matrix.py",
    "aphasia_progression.py",
]
for s in scripts:
    print(f"\n=== {s} ===")
    subprocess.check_call([sys.executable, os.path.join(HERE, s)])
