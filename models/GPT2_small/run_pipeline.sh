#!/bin/bash
# =============================================================
# Full Experiment Pipeline — GPT-2 Small
# =============================================================
# Run inside Docker:
#   docker run --gpus all -it tesis-gpt2 bash run_pipeline.sh
#
# Requirements:
#   - NVIDIA GPU with CUDA 12.1+ driver
#   - ~4GB VRAM (GPT-2-small is 125M params)
#   - ~8GB system RAM
#
# Steps:
#   1. Generate data              (data_gen.py)
#   2. Validate data              (validation_tests.py --no-model)
#   3. Extract CAVs               (cav_extraction.py)
#   4. Statistical CAV validation (cav_statistical_tests.py --from-saved)
#   5. Run factorial experiment   (experiment_runner.py)
#   6. Run multi-layer experiment (experiment_multilayer.py)
#   7. Full validation            (validation_tests.py)
# =============================================================

set -e  # Stop on first error

echo "============================================================"
echo "GPT-2 Small — Semantic Aphasia Experiment"
echo "============================================================"

# Verify GPU availability
python -c "
import torch
assert torch.cuda.is_available(), 'ERROR: CUDA not available! This experiment requires a GPU.'
props = torch.cuda.get_device_properties(0)
print(f'  GPU: {props.name}')
print(f'  VRAM: {props.total_memory / 1e9:.1f} GB')
print(f'  CUDA: {torch.version.cuda}')
print(f'  GPT-2 Small will use ~0.5GB of {props.total_memory / 1e9:.0f}GB VRAM')
"

echo ""
echo "============================================================"
echo "STEP 1/7: Generate Experiment Data"
echo "============================================================"
python data_gen.py

echo ""
echo "============================================================"
echo "STEP 2/7: Validate Data (no model needed)"
echo "============================================================"
python validation_tests.py --no-model

echo ""
echo "============================================================"
echo "STEP 3/7: Extract CAVs (dual method, multiple layers)"
echo "============================================================"
python cav_extraction.py

echo ""
echo "============================================================"
echo "STEP 4/7: Statistical CAV Validation (from saved activations)"
echo "============================================================"
python cav_statistical_tests.py --from-saved

echo ""
echo "============================================================"
echo "STEP 5/7: Run Full Factorial Experiment"
echo "============================================================"
python experiment_runner.py

echo ""
echo "============================================================"
echo "STEP 6/7: Run Multi-Layer Experiment (best approach)"
echo "============================================================"
python experiment_multilayer.py

echo ""
echo "============================================================"
echo "STEP 7/7: Full Validation Suite"
echo "============================================================"
python validation_tests.py

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE — GPT-2 Small"
echo "============================================================"
echo "Results:"
echo "  - experiment_data.json              (training data + benchmarks)"
echo "  - cavs/                             (CAV tensors + extraction report)"
echo "  - cavs/statistical_validation.json  (TCAV, selectivity, bootstrap)"
echo "  - results/results_factorial.csv     (factorial experiment)"
echo "  - results/results_specificity.csv   (specificity test)"
echo "  - results/results_baseline.json     (baseline scores)"
echo "  - results_multilayer/               (multi-layer SVM results)"
echo "  - results/validation_report.json    (validation suite)"
