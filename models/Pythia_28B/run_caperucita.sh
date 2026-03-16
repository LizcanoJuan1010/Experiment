#!/bin/bash
set -e

echo "============================================================"
echo "  CAPERUCITA PIPELINE -- Pythia 2.8B (sin Docker)"
echo "============================================================"

# Verify GPU
python -c "
import torch
assert torch.cuda.is_available(), 'ERROR: CUDA no disponible!'
print(f'  GPU: {torch.cuda.get_device_properties(0).name}')
"

# Verify prerequisites
python -c "
import os
assert os.path.exists('experiment_data.json'), \
    'ERROR: experiment_data.json no encontrado. Ejecuta data_gen.py primero.'
assert os.path.isdir('cavs'), \
    'ERROR: directorio cavs/ no encontrado.'
print('  Prerequisites OK')
"

echo ""
echo "[1/3] Generating wolf concept data..."
python gen_wolf_data.py

echo ""
echo "[2/3] Extracting wolf CAVs (all 32 layers)..."
python extract_wolf_cavs.py

echo ""
echo "[3/3] Running caperucita test..."
python -m test.caperucita_test

echo ""
echo "============================================================"
echo "  DONE -- Results in results/caperucita_test_results.json"
echo "============================================================"
