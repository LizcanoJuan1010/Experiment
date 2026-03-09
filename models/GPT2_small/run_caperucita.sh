#!/bin/bash
# =============================================================
# Caperucita Test — Wolf Concept Ablation in Little Red Riding Hood
# =============================================================
# Run inside Docker:
#   docker run --gpus all -it tesis-gpt2 bash run_caperucita.sh
#
# Prerequisites:
#   - Run run_pipeline.sh first (generates experiment_data.json + CAVs)
#
# Output:
#   - results/caperucita_test_results.json
# =============================================================

set -e

echo "============================================================"
echo "Caperucita Test — Wolf Concept Ablation Pipeline"
echo "============================================================"

# Verify GPU
python -c "
import torch
assert torch.cuda.is_available(), 'ERROR: CUDA not available!'
print(f'  GPU: {torch.cuda.get_device_properties(0).name}')
"

# Verify prerequisites
python -c "
import os
assert os.path.exists('experiment_data.json'), \
    'ERROR: experiment_data.json not found. Run run_pipeline.sh first.'
assert os.path.isdir('cavs'), \
    'ERROR: cavs/ directory not found. Run run_pipeline.sh first.'
print('  Prerequisites OK')
"

echo ""
echo "============================================================"
echo "STEP 1/3: Generate wolf concept data"
echo "============================================================"
python gen_wolf_data.py

echo ""
echo "============================================================"
echo "STEP 2/3: Extract wolf CAVs (all 12 layers)"
echo "============================================================"
python extract_wolf_cavs.py

echo ""
echo "============================================================"
echo "STEP 3/3: Run Caperucita test"
echo "============================================================"
python -m test.caperucita_test

echo ""
echo "============================================================"
echo "CAPERUCITA TEST COMPLETE"
echo "============================================================"
echo "Results: results/caperucita_test_results.json"
