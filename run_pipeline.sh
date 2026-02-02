#!/bin/bash
# =============================================================
# Full Experiment Pipeline
# =============================================================
# Run inside Docker:
#   docker run --gpus all -it tesis bash run_pipeline.sh
#
# Steps:
#   1. Generate data          (data_gen.py)
#   2. Validate data          (validation_tests.py --no-model)
#   3. Extract CAVs           (cav_extraction.py)
#   4. Run experiment         (experiment_runner.py)
#   5. Full validation        (validation_tests.py)
# =============================================================

set -e  # Stop on first error

echo "============================================================"
echo "STEP 1/5: Generate Experiment Data"
echo "============================================================"
python data_gen.py

echo ""
echo "============================================================"
echo "STEP 2/5: Validate Data (no model needed)"
echo "============================================================"
python validation_tests.py --no-model

echo ""
echo "============================================================"
echo "STEP 3/5: Extract CAVs (dual method, multiple layers)"
echo "============================================================"
python cav_extraction.py

echo ""
echo "============================================================"
echo "STEP 4/5: Run Full Factorial Experiment"
echo "============================================================"
python experiment_runner.py

echo ""
echo "============================================================"
echo "STEP 5/5: Full Validation Suite"
echo "============================================================"
python validation_tests.py

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Results:"
echo "  - experiment_data.json       (training data + benchmarks)"
echo "  - cavs/                      (CAV tensors + extraction report)"
echo "  - results/results_factorial.csv    (factorial experiment)"
echo "  - results/results_specificity.csv  (specificity test)"
echo "  - results/results_baseline.json    (baseline scores)"
echo "  - validation_report.json     (all 12 tests)"
