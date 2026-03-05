#!/bin/bash
# =============================================================
# Full Experiment Pipeline — Pythia 2.8B
# =============================================================
# Run inside Docker:
#   docker run --gpus all --shm-size=8g -it tesis-pythia28b bash run_pipeline.sh
#
# Requirements:
#   - NVIDIA RTX 5060 Ti (16GB VRAM) or compatible GPU
#   - CUDA 12.8+ driver on host (Blackwell architecture)
#   - ~16GB system RAM
#
# Steps:
#   1. Generate data              (data_gen.py)
#   2. Validate data              (validation_tests.py --no-model)
#   3. Extract CAVs               (cav_extraction.py)
#   4. Statistical CAV validation (cav_statistical_tests.py --from-saved)
#   5. Run factorial experiment   (experiment_runner.py)
#   6. Run multi-layer experiment (experiment_multilayer.py)
#   7. Full validation            (validation_tests.py)
#   8. Qualitative aphasia test   (test/aphasia_test.py)
#   9. Wolf concept data gen      (gen_wolf_data.py)
#  10. Wolf CAV extraction        (extract_wolf_cavs.py)
#  11. Caperucita Roja test       (test/caperucita_test.py)
# =============================================================

set -e  # Stop on first error

echo "============================================================"
echo "Pythia 2.8B — Semantic Aphasia Experiment"
echo "============================================================"

# Verify GPU availability
python -c "
import torch
assert torch.cuda.is_available(), 'ERROR: CUDA not available! This experiment requires a GPU.'
props = torch.cuda.get_device_properties(0)
print(f'  GPU: {props.name}')
print(f'  VRAM: {props.total_memory / 1e9:.1f} GB')
print(f'  CUDA: {torch.version.cuda}')
print(f'  Compute Capability: {props.major}.{props.minor}')
if props.total_memory < 8e9:
    print('  WARNING: Less than 8GB VRAM detected. float16 mode requires ~6GB.')
print(f'  Pythia 2.8B float16 will use ~5.6GB of {props.total_memory / 1e9:.0f}GB VRAM')
"

echo ""
echo "============================================================"
echo "STEP 1/11: Generate Experiment Data"
echo "============================================================"
python data_gen.py

echo ""
echo "============================================================"
echo "STEP 2/11: Validate Data (no model needed)"
echo "============================================================"
python validation_tests.py --no-model

echo ""
echo "============================================================"
echo "STEP 3/11: Extract CAVs (dual method, multiple layers)"
echo "============================================================"
python cav_extraction.py

echo ""
echo "============================================================"
echo "STEP 4/11: Statistical CAV Validation (from saved activations)"
echo "============================================================"
python cav_statistical_tests.py --from-saved

echo ""
echo "============================================================"
echo "STEP 5/11: Run Full Factorial Experiment (alphas 1.5-20)"
echo "============================================================"
python experiment_runner.py

echo ""
echo "============================================================"
echo "STEP 6/11: Run Multi-Layer Experiment (best approach)"
echo "============================================================"
python experiment_multilayer.py

echo ""
echo "============================================================"
echo "STEP 7/11: Full Validation Suite"
echo "============================================================"
python validation_tests.py

echo ""
echo "============================================================"
echo "STEP 8/11: Qualitative Aphasia Test (text generation)"
echo "============================================================"
cd test && python aphasia_test.py && cd ..

echo ""
echo "============================================================"
echo "STEP 9/11: Generate Wolf Concept Data"
echo "============================================================"
python gen_wolf_data.py

echo ""
echo "============================================================"
echo "STEP 10/11: Extract Wolf CAVs"
echo "============================================================"
python extract_wolf_cavs.py

echo ""
echo "============================================================"
echo "STEP 11/11: Caperucita Roja Test (Wolf Ablation Narrative)"
echo "============================================================"
cd test && python caperucita_test.py && cd ..

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE — Pythia 2.8B"
echo "============================================================"
echo "Results:"
echo "  - experiment_data.json              (training data + benchmarks)"
echo "  - cavs/                             (CAV tensors + extraction report)"
echo "  - cavs/statistical_validation.json  (TCAV, selectivity, bootstrap)"
echo "  - results/results_factorial.csv     (factorial experiment, 144 conditions)"
echo "  - results/results_specificity.csv   (specificity test)"
echo "  - results/results_baseline.json     (baseline scores)"
echo "  - results_multilayer/               (multi-layer SVM results)"
echo "  - results/validation_report.json    (all 20 tests)"
echo "  - results/aphasia_test_results.json (qualitative text generation)"
echo "  - cavs/wolf_extraction_report.json  (wolf CAV quality metrics)"
echo "  - results/caperucita_test_results.json (wolf ablation narrative test)"
