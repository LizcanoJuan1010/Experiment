#!/bin/bash
# =============================================================
# CCT Test — Camel and Cactus Test (Semantic Association)
# =============================================================
# Run inside Docker:
#   docker run --gpus all -it tesis-gpt2 bash run_cct.sh
#
# Prerequisites:
#   - Run run_pipeline.sh first (generates CAVs + benchmark data)
#   - BEA association benchmark in Benchmark_construction/output/
#
# Output:
#   - results/cct_test_results.json
# =============================================================

set -e

echo "============================================================"
echo "CCT Test — Semantic Association with Ablation"
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
assert os.path.exists('Benchmark_construction/output/bea_association_benchmark.json'), \
    'ERROR: BEA benchmark not found. Run the benchmark builder first.'
assert os.path.isdir('cavs'), \
    'ERROR: cavs/ directory not found. Run run_pipeline.sh first.'
print('  Prerequisites OK')
"

echo ""
echo "Running CCT test..."
python -m test.cct_test

echo ""
echo "============================================================"
echo "CCT TEST COMPLETE"
echo "============================================================"
echo "Results: results/cct_test_results.json"
