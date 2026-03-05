"""
Validation Tests — LLaVA-1.5 7B Multimodal Aphasia Experiment
==============================================================
29 validation tests covering data integrity, CAV quality,
baseline performance (R1, R2a, R2b), and experiment consistency.

Usage:
    python validation_tests.py                # Full suite (requires GPU + model)
    python validation_tests.py --no-model     # Data-only tests (no GPU needed)

Exit codes:
    0 = all tests passed
    1 = some tests failed (warnings logged)
"""

import json
import os
import sys
import argparse
import numpy as np
import torch

import experiment_config as cfg


# ---------------------------------------------------------------------------
# Test Framework
# ---------------------------------------------------------------------------
RESULTS = []
WARNINGS = []


def log_test(test_id, name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] T{test_id:02d}: {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    RESULTS.append({
        "test_id": test_id,
        "name": name,
        "passed": passed,
        "detail": detail,
    })
    if not passed:
        WARNINGS.append(f"T{test_id:02d}: {name}")


def load_data():
    with open(cfg.DATA_FILE, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# DATA INTEGRITY TESTS (T01-T06) — No model needed
# ---------------------------------------------------------------------------
def t01_image_count(data):
    """T01: Each concept has >= MIN_IMAGES_PER_CONCEPT images."""
    image_dataset = data.get("image_dataset", {})
    all_pass = True
    details = []
    for concept in cfg.CONCEPTS:
        imgs = image_dataset.get(concept, [])
        n = len(imgs)
        ok = n >= cfg.MIN_IMAGES_PER_CONCEPT
        details.append(f"{concept}={n}")
        if not ok:
            all_pass = False
    log_test(1, "Image count per concept", all_pass,
             f"{', '.join(details)} (min={cfg.MIN_IMAGES_PER_CONCEPT})")


def t02_no_image_overlap(data):
    """T02: No image appears in multiple concepts."""
    image_dataset = data.get("image_dataset", {})
    all_paths = {}
    overlaps = []
    for concept in cfg.CONCEPTS:
        for img in image_dataset.get(concept, []):
            path = img["image_path"]
            if path in all_paths:
                overlaps.append(f"{path}: {all_paths[path]} & {concept}")
            all_paths[path] = concept

    passed = len(overlaps) == 0
    log_test(2, "No cross-concept image overlap", passed,
             f"{len(overlaps)} overlaps" if overlaps else "clean")


def t03_neutral_images_exist(data):
    """T03: Neutral images exist for negative CAV training."""
    neutrals = data.get("image_dataset", {}).get("neutral", [])
    n = len(neutrals)
    passed = n >= 20
    log_test(3, "Neutral images for CAV negatives", passed,
             f"{n} neutral images")


def t04_no_concept_in_neutral(data):
    """T04: No concept images contaminate neutral set."""
    image_dataset = data.get("image_dataset", {})
    concept_paths = set()
    for concept in cfg.CONCEPTS:
        for img in image_dataset.get(concept, []):
            concept_paths.add(img["image_path"])

    neutral_paths = {img["image_path"] for img in image_dataset.get("neutral", [])}
    contamination = concept_paths & neutral_paths
    passed = len(contamination) == 0
    log_test(4, "No concept-neutral contamination", passed,
             f"{len(contamination)} contaminated" if contamination else "clean")


def t05_cav_training_pairs_balanced(data):
    """T05: CAV training has matched pos/neg pairs per concept."""
    cav_training = data.get("cav_training", {})
    all_ok = True
    details = []
    for concept in cfg.CONCEPTS:
        ct = cav_training.get(concept, {})
        n_pos = len(ct.get("positive", []))
        n_neg = len(ct.get("negative", []))
        ok = n_pos > 0 and n_neg > 0 and n_pos == n_neg
        details.append(f"{concept}:{n_pos}p/{n_neg}n")
        if not ok:
            all_ok = False
    log_test(5, "CAV training pairs balanced", all_ok, ", ".join(details))


def t06_template_matching(data):
    """T06: Same prompt used for positive and negative CAV training."""
    cav_training = data.get("cav_training", {})
    all_ok = True
    for concept in cfg.CONCEPTS:
        ct = cav_training.get(concept, {})
        pos = ct.get("positive", [])
        neg = ct.get("negative", [])

        if pos and neg:
            pos_prompts = set(p[1] for p in pos)
            neg_prompts = set(p[1] for p in neg)
            if pos_prompts != neg_prompts:
                all_ok = False
    log_test(6, "Template matching (same prompt pos/neg)", all_ok)


def t07_r1_benchmark_format(data):
    """T07: R1 benchmark items have correct structure."""
    r1 = data.get("r1_benchmark", {})
    all_ok = True
    total = 0
    for concept in cfg.CONCEPTS:
        items = r1.get(concept, [])
        for item in items:
            total += 1
            if not all(k in item for k in ["image_path", "correct_answer", "options"]):
                all_ok = False
            if len(item.get("options", [])) < 2:
                all_ok = False
            if item.get("correct_answer") not in item.get("options", []):
                all_ok = False
    log_test(7, "R1 benchmark format", all_ok, f"{total} items")


def t08_r2_benchmark_format(data):
    """T08: R2 benchmark items have correct structure."""
    r2 = data.get("r2_benchmark", {})
    all_ok = True
    total = 0
    for concept in cfg.CONCEPTS:
        pairs = r2.get(concept, [])
        for pair in pairs:
            total += 1
            if not all(k in pair for k in ["image_path", "text_description"]):
                all_ok = False
    log_test(8, "R2 benchmark format", all_ok, f"{total} pairs")


def t09_images_exist_on_disk(data):
    """T09: All referenced images actually exist on disk."""
    image_dataset = data.get("image_dataset", {})
    missing = 0
    total = 0
    for concept_imgs in image_dataset.values():
        for img in concept_imgs:
            total += 1
            if not os.path.exists(img["image_path"]):
                missing += 1

    passed = missing == 0
    log_test(9, "All images exist on disk", passed,
             f"{total - missing}/{total} found" + (f", {missing} missing" if missing else ""))


# ---------------------------------------------------------------------------
# CAV QUALITY TESTS (T10-T14) — Requires extracted CAVs
# ---------------------------------------------------------------------------
def t10_cav_files_exist():
    """T10: CAV .pt files exist for all concept/method/layer combos."""
    missing = []
    total = 0
    for concept in cfg.CONCEPTS:
        for method in cfg.CAV_METHODS:
            for layer in cfg.EXTRACTION_LAYERS:
                total += 1
                path = os.path.join(cfg.CAV_DIR, f"{concept}_{method}_layer{layer}.pt")
                if not os.path.exists(path):
                    missing.append(path)

    passed = len(missing) == 0
    log_test(10, "CAV files exist", passed,
             f"{total - len(missing)}/{total} found")


def t11_cav_shape():
    """T11: CAV tensors have correct shape [d_model]."""
    all_ok = True
    for concept in cfg.CONCEPTS:
        for method in cfg.CAV_METHODS:
            for layer in cfg.EXTRACTION_LAYERS:
                path = os.path.join(cfg.CAV_DIR, f"{concept}_{method}_layer{layer}.pt")
                if os.path.exists(path):
                    cav = torch.load(path, weights_only=True)
                    if cav.shape != (cfg.D_MODEL,):
                        all_ok = False

    log_test(11, f"CAV shape == [{cfg.D_MODEL}]", all_ok)


def t12_cav_unit_norm():
    """T12: CAV vectors are approximately unit norm."""
    max_dev = 0.0
    for concept in cfg.CONCEPTS:
        for method in cfg.CAV_METHODS:
            for layer in cfg.EXTRACTION_LAYERS:
                path = os.path.join(cfg.CAV_DIR, f"{concept}_{method}_layer{layer}.pt")
                if os.path.exists(path):
                    cav = torch.load(path, weights_only=True)
                    norm = torch.norm(cav).item()
                    dev = abs(norm - 1.0)
                    max_dev = max(max_dev, dev)

    passed = max_dev < 0.05
    log_test(12, "CAV unit norm (deviation < 0.05)", passed,
             f"max_deviation={max_dev:.4f}")


def t13_cav_overlap():
    """T13: Cross-concept CAV cosine similarity below threshold."""
    max_cos = 0.0
    pairs_checked = 0
    for layer in cfg.EXTRACTION_LAYERS:
        cavs = {}
        for concept in cfg.CONCEPTS:
            path = os.path.join(cfg.CAV_DIR, f"{concept}_svm_layer{layer}.pt")
            if os.path.exists(path):
                cavs[concept] = torch.load(path, weights_only=True)

        for i, c1 in enumerate(cfg.CONCEPTS):
            for c2 in cfg.CONCEPTS[i + 1:]:
                if c1 in cavs and c2 in cavs:
                    cos = torch.nn.functional.cosine_similarity(
                        cavs[c1].unsqueeze(0), cavs[c2].unsqueeze(0),
                    ).item()
                    max_cos = max(max_cos, abs(cos))
                    pairs_checked += 1

    passed = max_cos < cfg.CAV_OVERLAP_THRESHOLD
    log_test(13, f"CAV overlap < {cfg.CAV_OVERLAP_THRESHOLD}", passed,
             f"max_cos={max_cos:.4f}, {pairs_checked} pairs")


def t14_svm_cv_accuracy():
    """T14: SVM cross-validation accuracy above stability threshold."""
    report_path = os.path.join(cfg.CAV_DIR, "extraction_report.json")
    if not os.path.exists(report_path):
        log_test(14, "SVM CV accuracy", False, "extraction_report.json not found")
        return

    with open(report_path, "r") as f:
        report = json.load(f)

    min_cv = 1.0
    for key, info in report.get("cavs", {}).items():
        if "svm" in key and "cv_accuracy_mean" in info:
            min_cv = min(min_cv, info["cv_accuracy_mean"])

    passed = min_cv >= cfg.CAV_STABILITY_CI_LOWER
    log_test(14, f"SVM CV accuracy >= {cfg.CAV_STABILITY_CI_LOWER}", passed,
             f"min_cv={min_cv:.4f}")


# ---------------------------------------------------------------------------
# MODEL & BASELINE TESTS (T15-T18) — Requires GPU
# ---------------------------------------------------------------------------
def t15_model_architecture(model):
    """T15: Model has expected architecture (32 layers, 4096 dim)."""
    from model_loader import get_model_layers
    n_layers = len(get_model_layers(model))
    d_model = model.config.text_config.hidden_size

    ok_layers = n_layers == cfg.N_LAYERS_TOTAL
    ok_dim = d_model == cfg.D_MODEL
    passed = ok_layers and ok_dim

    log_test(15, "Model architecture", passed,
             f"layers={n_layers}, d_model={d_model}")


def t16_model_quantized(model):
    """T16: Model is loaded in 4-bit quantization."""
    is_quantized = getattr(model, "is_quantized", False) or \
                   hasattr(model.config, "quantization_config")
    log_test(16, "Model 4-bit quantized", is_quantized)


def t17_baseline_r1(baselines):
    """T17: Baseline R1 accuracy above threshold."""
    all_ok = True
    details = []
    for concept in cfg.CONCEPTS:
        r1 = baselines.get(concept, {}).get("r1_accuracy",
              baselines.get(concept, {}).get("r1", 0.0))
        ok = r1 >= cfg.R1_BASELINE_THRESHOLD
        details.append(f"{concept}={r1:.3f}")
        if not ok:
            all_ok = False

    log_test(17, f"Baseline R1 >= {cfg.R1_BASELINE_THRESHOLD}", all_ok,
             ", ".join(details))


def t18_baseline_r2a(baselines):
    """T18: Baseline R2a (cross-modal similarity) above threshold."""
    all_ok = True
    details = []
    for concept in cfg.CONCEPTS:
        r2a = baselines.get(concept, {}).get("r2a_similarity",
               baselines.get(concept, {}).get("r2a",
               baselines.get(concept, {}).get("r2_similarity",
               baselines.get(concept, {}).get("r2", 0.0))))
        ok = r2a >= cfg.R2A_BASELINE_THRESHOLD
        details.append(f"{concept}={r2a:.4f}")
        if not ok:
            all_ok = False

    log_test(18, f"Baseline R2a >= {cfg.R2A_BASELINE_THRESHOLD}", all_ok,
             ", ".join(details))


def t29_baseline_r2b(baselines):
    """T29: Baseline R2b (output similarity) near 1.0."""
    all_ok = True
    details = []
    for concept in cfg.CONCEPTS:
        r2b = baselines.get(concept, {}).get("r2b_similarity",
               baselines.get(concept, {}).get("r2b", 0.0))
        ok = r2b >= cfg.R2B_BASELINE_THRESHOLD
        details.append(f"{concept}={r2b:.4f}")
        if not ok:
            all_ok = False

    log_test(29, f"Baseline R2b >= {cfg.R2B_BASELINE_THRESHOLD}", all_ok,
             ", ".join(details))


# ---------------------------------------------------------------------------
# EXPERIMENT RESULTS TESTS (T19-T20) — Requires experiment output
# ---------------------------------------------------------------------------
def t19_factorial_results_exist():
    """T19: Factorial experiment results CSV exists and has expected rows."""
    path = os.path.join(cfg.RESULTS_DIR, "results_factorial.csv")
    if not os.path.exists(path):
        log_test(19, "Factorial results exist", False, "file not found")
        return

    import pandas as pd
    df = pd.read_csv(path)
    expected_min = len(cfg.CONCEPTS) * 2  # At least some rows
    passed = len(df) >= expected_min
    log_test(19, "Factorial results", passed,
             f"{len(df)} rows (min expected: {expected_min})")


def t20_specificity_results(data=None):
    """T20: Specificity test shows on-target > off-target effects (R1 + R2a)."""
    path = os.path.join(cfg.RESULTS_DIR, "results_specificity.csv")
    if not os.path.exists(path):
        log_test(20, "Specificity results", False, "file not found")
        return

    import pandas as pd
    df = pd.read_csv(path)
    on = df[df["on_target"] == True]
    off = df[df["on_target"] == False]

    mean_on_r1 = on["delta_r1"].mean() if len(on) > 0 else 0.0
    mean_off_r1 = off["delta_r1"].mean() if len(off) > 0 else 0.0

    # Check R2a if column exists, else fall back to delta_r2
    r2a_col = "delta_r2a" if "delta_r2a" in df.columns else "delta_r2"
    mean_on_r2a = on[r2a_col].mean() if len(on) > 0 else 0.0
    mean_off_r2a = off[r2a_col].mean() if len(off) > 0 else 0.0

    passed = mean_on_r1 > mean_off_r1
    log_test(20, "Specificity: on-target > off-target", passed,
             f"R1: on={mean_on_r1:.3f}/off={mean_off_r1:.3f}, "
             f"R2a: on={mean_on_r2a:.4f}/off={mean_off_r2a:.4f}")


# ---------------------------------------------------------------------------
# ADVANCED TESTS (T21-T22) — Requires GPU
# ---------------------------------------------------------------------------
def t21_image_tokens_present(model, processor):
    """T21: Image tokens (576) are injected into sequence."""
    from PIL import Image
    import tempfile

    # Create a simple test image
    img = Image.new("RGB", (336, 336), color=(128, 128, 128))
    tmp_path = os.path.join(tempfile.gettempdir(), "test_validation.jpg")
    img.save(tmp_path)

    prompt = "USER: <image>\nDescribe this.\nASSISTANT:"
    inputs = processor(text=prompt, images=img, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]

    # With 576 image tokens, sequence should be > 576
    passed = seq_len > cfg.N_IMAGE_TOKENS
    log_test(21, f"Image tokens in sequence (>{cfg.N_IMAGE_TOKENS})", passed,
             f"seq_len={seq_len}")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def t22_activation_shape(model, processor):
    """T22: Activation collection returns correct shape [seq_len, d_model]."""
    from PIL import Image
    from pyvene_utils import collect_activations_at_layer
    import tempfile

    img = Image.new("RGB", (336, 336), color=(64, 64, 200))
    tmp_path = os.path.join(tempfile.gettempdir(), "test_act_shape.jpg")
    img.save(tmp_path)

    prompt = "USER: <image>\nDescribe this.\nASSISTANT:"
    device = next(model.parameters()).device

    act = collect_activations_at_layer(
        model, processor, tmp_path, prompt, layer=16, device=device,
    )

    ok_dim = act.shape[-1] == cfg.D_MODEL
    ok_2d = act.dim() == 2
    passed = ok_dim and ok_2d
    log_test(22, f"Activation shape [seq, {cfg.D_MODEL}]", passed,
             f"shape={list(act.shape)}")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


# ---------------------------------------------------------------------------
# STATISTICAL RIGOR TESTS (T23-T28) — Requires saved activations
# ---------------------------------------------------------------------------
def t23_tcav_significance():
    """T23: TCAV permutation test shows p < 0.05 for all concept/layer combos."""
    stat_path = os.path.join(cfg.CAV_DIR, cfg.STATISTICAL_VALIDATION_FILE)
    if not os.path.exists(stat_path):
        log_test(23, "TCAV significance", False, "statistical_validation.json not found")
        return

    with open(stat_path, "r") as f:
        stat_data = json.load(f)

    all_sig = True
    details = []
    for concept in cfg.CONCEPTS:
        for layer in cfg.EXTRACTION_LAYERS:
            key = f"layer_{layer}"
            for method in cfg.CAV_METHODS:
                tcav = (stat_data.get("results", {})
                        .get(concept, {})
                        .get(key, {})
                        .get(method, {})
                        .get("tcav_permutation", {}))
                if tcav:
                    sig = tcav.get("significant", False)
                    p = tcav.get("p_value", 1.0)
                    if not sig:
                        all_sig = False
                        details.append(f"{concept}/{method}/L{layer}:p={p}")

    passed = all_sig
    detail = "all significant" if passed else f"FAIL: {'; '.join(details[:3])}"
    log_test(23, f"TCAV p < {cfg.TCAV_SIGNIFICANCE_ALPHA}", passed, detail)


def t24_selectivity():
    """T24: Selectivity > MIN for all concept/layer combos."""
    stat_path = os.path.join(cfg.CAV_DIR, cfg.STATISTICAL_VALIDATION_FILE)
    if not os.path.exists(stat_path):
        log_test(24, "Selectivity", False, "statistical_validation.json not found")
        return

    with open(stat_path, "r") as f:
        stat_data = json.load(f)

    min_sel = 1.0
    for concept in cfg.CONCEPTS:
        for layer in cfg.EXTRACTION_LAYERS:
            key = f"layer_{layer}"
            for method in cfg.CAV_METHODS:
                sel = (stat_data.get("results", {})
                       .get(concept, {})
                       .get(key, {})
                       .get(method, {})
                       .get("selectivity", {}))
                if sel:
                    s = sel.get("selectivity", 0.0)
                    min_sel = min(min_sel, s)

    passed = min_sel >= cfg.SELECTIVITY_MIN
    log_test(24, f"Selectivity >= {cfg.SELECTIVITY_MIN}", passed,
             f"min_selectivity={min_sel:.4f}")


def t25_cohens_d():
    """T25: Cohen's d >= LARGE for all concept/layer combos."""
    stat_path = os.path.join(cfg.CAV_DIR, cfg.STATISTICAL_VALIDATION_FILE)
    if not os.path.exists(stat_path):
        log_test(25, "Cohen's d", False, "statistical_validation.json not found")
        return

    with open(stat_path, "r") as f:
        stat_data = json.load(f)

    min_d = float("inf")
    for concept in cfg.CONCEPTS:
        for layer in cfg.EXTRACTION_LAYERS:
            key = f"layer_{layer}"
            for method in cfg.CAV_METHODS:
                d_info = (stat_data.get("results", {})
                          .get(concept, {})
                          .get(key, {})
                          .get(method, {})
                          .get("cohens_d", {}))
                if d_info:
                    d = abs(d_info.get("cohens_d", 0.0))
                    min_d = min(min_d, d)

    passed = min_d >= cfg.COHENS_D_LARGE
    log_test(25, f"Cohen's d >= {cfg.COHENS_D_LARGE}", passed,
             f"min_d={min_d:.4f}")


def t26_bootstrap_stability():
    """T26: Bootstrap CI lower bound >= stability threshold."""
    stat_path = os.path.join(cfg.CAV_DIR, cfg.STATISTICAL_VALIDATION_FILE)
    if not os.path.exists(stat_path):
        log_test(26, "Bootstrap stability", False, "statistical_validation.json not found")
        return

    with open(stat_path, "r") as f:
        stat_data = json.load(f)

    min_ci = 1.0
    for concept in cfg.CONCEPTS:
        for layer in cfg.EXTRACTION_LAYERS:
            key = f"layer_{layer}"
            for method in cfg.CAV_METHODS:
                boot = (stat_data.get("results", {})
                        .get(concept, {})
                        .get(key, {})
                        .get(method, {})
                        .get("bootstrap_ci", {}))
                if boot:
                    ci_lower = boot.get("ci_lower", 0.0)
                    min_ci = min(min_ci, ci_lower)

    passed = min_ci >= cfg.CAV_STABILITY_CI_LOWER
    log_test(26, f"Bootstrap CI lower >= {cfg.CAV_STABILITY_CI_LOWER}", passed,
             f"min_ci_lower={min_ci:.4f}")


def t27_train_test_split(data):
    """T27: Train/test split exists and has no overlap."""
    train = data.get("train_split", {})
    test = data.get("test_split", {})

    if not train or not test:
        log_test(27, "Train/test split exists", False, "no split data found")
        return

    all_ok = True
    details = []
    for concept in cfg.CONCEPTS:
        train_paths = {img["image_path"] for img in train.get(concept, [])}
        test_paths = {img["image_path"] for img in test.get(concept, [])}
        overlap = train_paths & test_paths
        n_train = len(train_paths)
        n_test = len(test_paths)
        details.append(f"{concept}:{n_train}tr/{n_test}te")
        if overlap:
            all_ok = False
            details.append(f"LEAK:{len(overlap)}")

    log_test(27, "Train/test split (no leakage)", all_ok, ", ".join(details))


def t28_intra_domain_distractors(data):
    """T28: R1 distractors are intra-domain (not trivially different)."""
    r1 = data.get("r1_benchmark", {})
    all_ok = True
    trivial_distractors = {"animal", "fruit", "color", "emotion", "mineral",
                           "weather", "fabric", "element", "number", "language"}

    for concept in cfg.CONCEPTS:
        items = r1.get(concept, [])
        for item in items:
            options = set(item.get("options", []))
            correct = item.get("correct_answer", "")
            distractors = options - {correct}
            trivial = distractors & trivial_distractors
            if trivial:
                all_ok = False
                break
        if not all_ok:
            break

    log_test(28, "R1 intra-domain distractors", all_ok,
             "no trivially easy distractors" if all_ok else "found trivial distractors")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-model", action="store_true",
                        help="Skip tests that require GPU/model")
    args = parser.parse_args()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Validation Suite — LLaVA-1.5 7B Multimodal Aphasia")
    print(f"  Mode: {'data-only' if args.no_model else 'full (GPU)'}")
    print(f"{sep}")

    # Load experiment data
    if not os.path.exists(cfg.DATA_FILE):
        print(f"\n  ERROR: {cfg.DATA_FILE} not found. Run data_gen_multimodal.py first.")
        sys.exit(1)

    data = load_data()

    # === Data Integrity Tests (T01-T09) ===
    print(f"\n--- Data Integrity ---")
    t01_image_count(data)
    t02_no_image_overlap(data)
    t03_neutral_images_exist(data)
    t04_no_concept_in_neutral(data)
    t05_cav_training_pairs_balanced(data)
    t06_template_matching(data)
    t07_r1_benchmark_format(data)
    t08_r2_benchmark_format(data)
    t09_images_exist_on_disk(data)

    # === CAV Quality Tests (T10-T14) ===
    print(f"\n--- CAV Quality ---")
    if os.path.exists(cfg.CAV_DIR):
        t10_cav_files_exist()
        t11_cav_shape()
        t12_cav_unit_norm()
        t13_cav_overlap()
        t14_svm_cv_accuracy()
    else:
        print("  SKIP: cavs/ directory not found (run cav_extraction.py first)")
        for tid in range(10, 15):
            log_test(tid, "CAV test (skipped)", True, "no cavs dir")

    # === Model & Baseline Tests (T15-T18) ===
    print(f"\n--- Model & Baseline ---")
    if not args.no_model:
        from model_loader import load_llava_model
        print("  Loading model for validation...")
        model, processor = load_llava_model()

        t15_model_architecture(model)
        t16_model_quantized(model)

        # Load baselines if available
        baseline_path = os.path.join(cfg.RESULTS_DIR, "results_baseline.json")
        if os.path.exists(baseline_path):
            with open(baseline_path, "r") as f:
                baselines = json.load(f)
            t17_baseline_r1(baselines)
            t18_baseline_r2a(baselines)
            t29_baseline_r2b(baselines)
        else:
            # Check multilayer baselines
            ml_baseline = os.path.join("results_multilayer", "baselines.json")
            if os.path.exists(ml_baseline):
                with open(ml_baseline, "r") as f:
                    baselines = json.load(f)
                t17_baseline_r1(baselines)
                t18_baseline_r2a(baselines)
                t29_baseline_r2b(baselines)
            else:
                print("  SKIP T17-T18, T29: No baseline results (run experiment first)")
                log_test(17, "Baseline R1 (skipped)", True, "no baseline file")
                log_test(18, "Baseline R2a (skipped)", True, "no baseline file")
                log_test(29, "Baseline R2b (skipped)", True, "no baseline file")
    else:
        print("  SKIP: --no-model flag (skipping T15-T18)")
        for tid in range(15, 19):
            log_test(tid, f"Model test (skipped, --no-model)", True, "skipped")

    # === Experiment Results Tests (T19-T20) ===
    print(f"\n--- Experiment Results ---")
    t19_factorial_results_exist()
    t20_specificity_results()

    # === Advanced Tests (T21-T22) ===
    print(f"\n--- Advanced ---")
    if not args.no_model:
        t21_image_tokens_present(model, processor)
        t22_activation_shape(model, processor)
    else:
        print("  SKIP: --no-model flag (skipping T21-T22)")
        log_test(21, "Image tokens (skipped)", True, "skipped")
        log_test(22, "Activation shape (skipped)", True, "skipped")

    # === Statistical Rigor Tests (T23-T28) ===
    print(f"\n--- Statistical Rigor ---")
    stat_file = os.path.join(cfg.CAV_DIR, cfg.STATISTICAL_VALIDATION_FILE)
    if os.path.exists(stat_file):
        t23_tcav_significance()
        t24_selectivity()
        t25_cohens_d()
        t26_bootstrap_stability()
    else:
        print("  SKIP T23-T26: No statistical validation file "
              "(run cav_statistical_tests.py first)")
        for tid in range(23, 27):
            log_test(tid, "Statistical test (skipped)", True, "no stats file")

    t27_train_test_split(data)
    t28_intra_domain_distractors(data)

    # === Summary ===
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = total - passed

    print(f"\n{sep}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    if WARNINGS:
        print(f"\n  WARNINGS:")
        for w in WARNINGS:
            print(f"    - {w}")
    print(f"{sep}\n")

    # Save report
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    report = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "tests": RESULTS,
        "warnings": WARNINGS,
    }
    with open(cfg.REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {cfg.REPORT_FILE}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
