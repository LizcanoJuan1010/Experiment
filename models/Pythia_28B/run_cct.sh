#!/bin/bash
# =============================================================
# CCT Pipeline — Full pipeline to run CCT test (Pythia 2.8B)
# =============================================================
# Run inside Docker:
#   docker run --gpus all --shm-size=8g -it tesis-pythia28b bash run_cct.sh
#
# This script runs the complete pipeline:
#   1. Generate training data for 30 concepts (data_gen.py)
#   2. Extract CAVs: 30 concepts × 2 methods × 3 layers (cav_extraction.py)
#   3. Run CCT test with factorial ablation (test/cct_test.py)
#
# Requirements:
#   - NVIDIA GPU with >= 16GB VRAM (RTX 5060 Ti or better)
#   - BEA association benchmark in Benchmark_construction/output/
#   - concepts/*.json for all 30 concepts
#
# Output:
#   - experiment_data.json              (training data)
#   - cavs/                             (CAV tensors)
#   - results/cct_test_results.json     (CCT factorial results)
# =============================================================

set -e

echo "============================================================"
echo "Pythia 2.8B — CCT Pipeline (30 concepts)"
echo "============================================================"

# Verify GPU
python -c "
import torch
assert torch.cuda.is_available(), 'ERROR: CUDA not available!'
props = torch.cuda.get_device_properties(0)
print(f'  GPU: {props.name}')
print(f'  VRAM: {props.total_memory / 1e9:.1f} GB')
print(f'  CUDA: {torch.version.cuda}')
print(f'  Pythia 2.8B float16 will use ~5.6GB of {props.total_memory / 1e9:.0f}GB VRAM')
"

# Verify benchmark exists
python -c "
import os
assert os.path.exists('Benchmark_construction/output/bea_association_benchmark.json'), \
    'ERROR: BEA association benchmark not found!'
import json
with open('Benchmark_construction/output/bea_association_benchmark.json') as f:
    d = json.load(f)
n_concepts = len(d['items'])
n_items = sum(len(v) for v in d['items'].values())
print(f'  Benchmark: {n_concepts} concepts, {n_items} items')
assert n_concepts >= 20, f'ERROR: Only {n_concepts} concepts in benchmark (need 30)'
print('  Prerequisites OK')
"

echo ""
echo "============================================================"
echo "STEP 1/3: Generate Training Data (30 concepts)"
echo "============================================================"
python data_gen.py

echo ""
echo "============================================================"
echo "STEP 2/3: Extract CAVs (30 concepts × 2 methods × 3 layers)"
echo "============================================================"
python cav_extraction.py

echo ""
echo "============================================================"
echo "STEP 3/3: Run CCT Test (Factorial Ablation)"
echo "============================================================"
python -m test.cct_test

echo ""
echo "============================================================"
echo "CCT PIPELINE COMPLETE — Pythia 2.8B"
echo "============================================================"
echo "Results:"
echo "  - experiment_data.json              (training data for 30 concepts)"
echo "  - cavs/                             (CAV tensors + extraction report)"
echo "  - results/cct_test_results.json     (CCT factorial results)"
