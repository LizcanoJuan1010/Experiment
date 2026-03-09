#!/bin/bash
set -e

echo "=============================================="
echo "  CAPERUCITA PIPELINE"
echo "=============================================="

echo ""
echo "[1/3] Generating wolf concept data..."
python gen_wolf_data.py

echo ""
echo "[2/3] Extracting wolf CAVs (all 32 layers)..."
python extract_wolf_cavs.py

echo ""
echo "[3/3] Running caperucita test..."
python test/caperucita_test.py

echo ""
echo "=============================================="
echo "  DONE -- Results in results/caperucita_test_results.json"
echo "=============================================="
