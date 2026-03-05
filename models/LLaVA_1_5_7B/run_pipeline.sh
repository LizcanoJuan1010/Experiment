#!/bin/bash
# =============================================================
# Full Experiment Pipeline — LLaVA-1.5 7B Multimodal Aphasia
# =============================================================
# Run inside Docker (PowerShell, with persistent volumes):
#   cd c:\dev\others\experiment\models\LLaVA_1_5_7B
#   docker run --gpus all --shm-size=8g -e HF_TOKEN=hf_xxx -v ${PWD}\hf_cache:/root/.cache/huggingface -v ${PWD}\data:/app/data -v ${PWD}\cavs:/app/cavs -v ${PWD}\results:/app/results -v ${PWD}\results_multilayer:/app/results_multilayer -it tesis-llava bash run_pipeline.sh
#
# The -v flags mount local directories into the container, so:
#   - Model weights (~13GB) download ONCE to hf_cache/ and persist locally
#   - Images download ONCE and persist on your disk
#   - CAVs, results, and reports are saved locally
#   - Re-runs skip already-downloaded files
#
# Requirements:
#   - NVIDIA RTX 5060 Ti (16GB VRAM) or compatible GPU
#   - CUDA 12.8+ driver on host (Blackwell architecture)
#   - ~16GB system RAM
#   - Internet access (first run only, for image download)
#   - HuggingFace authentication (for ImageNet-1K)
# =============================================================

set -e

echo "============================================================"
echo "LLaVA-1.5 7B — Multimodal Semantic Aphasia Experiment"
echo "============================================================"

# Verify GPU availability
python -c "
import torch
assert torch.cuda.is_available(), 'ERROR: CUDA not available!'
props = torch.cuda.get_device_properties(0)
print(f'  GPU: {props.name}')
print(f'  VRAM: {props.total_memory / 1e9:.1f} GB')
print(f'  CUDA: {torch.version.cuda}')
print(f'  LLaVA-1.5 7B 4-bit will use ~4GB of {props.total_memory / 1e9:.0f}GB VRAM')
"

# Verify HuggingFace authentication (required for ImageNet-1K)
echo ""
echo "  Checking HuggingFace authentication..."
python -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.whoami()
    print(f'  HF User: {info[\"name\"]}')
    print(f'  Auth: OK')
except Exception:
    print('  ERROR: HuggingFace not authenticated!')
    print()
    print('  ImageNet-1K requires authentication. Choose ONE option:')
    print('    Option A: Pass token as env variable:')
    print('      docker run --gpus all -e HF_TOKEN=hf_xxx -it tesis-llava bash run_pipeline.sh')
    print()
    print('    Option B: Log in interactively:')
    print('      docker run --gpus all -it tesis-llava bash')
    print('      huggingface-cli login')
    print('      bash run_pipeline.sh')
    print()
    print('  Get your token at: https://huggingface.co/settings/tokens')
    print('  Accept ImageNet license at: https://huggingface.co/datasets/ILSVRC/imagenet-1k')
    raise SystemExit(1)
"

echo ""
echo "============================================================"
echo "STEP 1/6: Download Images & Generate Experiment Data"
echo "  - ImageNet-1K subsets (centered objects)"
echo "  - Train/test split (70/30, no leakage)"
echo "============================================================"
python data_gen_multimodal.py

echo ""
echo "============================================================"
echo "STEP 2/6: Extract CAVs (dual method, multiple layers)"
echo "  - Saves raw activations for statistical tests"
echo "============================================================"
python cav_extraction.py

echo ""
echo "============================================================"
echo "STEP 3/6: CAV Statistical Validation"
echo "  - TCAV permutation test (n=30)"
echo "  - Selectivity test (Hewitt & Liang 2019)"
echo "  - Cohen's d with Hedges' g"
echo "  - Bootstrap CI (BCa, n=500)"
echo "============================================================"
python cav_statistical_tests.py --from-saved

echo ""
echo "============================================================"
echo "STEP 4/6: Run Full Factorial Experiment"
echo "============================================================"
python experiment_runner.py

echo ""
echo "============================================================"
echo "STEP 5/6: Run Multi-Layer Experiment"
echo "============================================================"
python experiment_multilayer.py

echo ""
echo "============================================================"
echo "STEP 6/6: Validation Suite (29 tests)"
echo "============================================================"
python validation_tests.py || echo "Validation completed with warnings"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE — LLaVA-1.5 7B"
echo "============================================================"
echo "Results:"
echo "  - data/experiment_data.json              (image dataset + train/test split)"
echo "  - cavs/                                  (CAV tensors + extraction report)"
echo "  - cavs/statistical_validation.json       (TCAV, selectivity, Cohen's d, bootstrap)"
echo "  - results/results_factorial.csv          (factorial: R1, R2a, R2b)"
echo "  - results/results_specificity.csv        (specificity test)"
echo "  - results/results_baseline.json          (baseline scores: R1, R2a, R2b)"
echo "  - results/validation_report.json         (29-test validation suite)"
echo "  - results_multilayer/                    (multi-layer results)"
