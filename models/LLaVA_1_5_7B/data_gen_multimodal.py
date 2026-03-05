"""
Multimodal Data Generator — ImageNet + Open Images
====================================================
Downloads and organizes images for concept CAV training and benchmarks.

KEY FIX: Train/test split prevents data leakage.
  - 70% of images -> CAV training (positive/negative pairs)
  - 30% of images -> R1/R2 benchmarks (NEVER seen during CAV extraction)

Image source priority (for maximum concept activation):
  1. ImageNet-1K subsets via HuggingFace — single centered objects (primary)
  2. Open Images V7 supplement (fallback if below minimum)

All images are single-object, centered — no COCO (objects not centered).

Usage:
    python data_gen_multimodal.py

Output:
    data/images/{concept_name}/*.jpg
    data/images/neutral/*.jpg
    data/experiment_data.json
"""

import json
import os
import random
from collections import defaultdict

from PIL import Image

import experiment_config as cfg

random.seed(cfg.RANDOM_SEED)


# ---------------------------------------------------------------------------
# ImageNet Download — Streaming Mode (no disk cache needed)
# ---------------------------------------------------------------------------
def download_all_imagenet_images(class_map):
    """
    Stream ImageNet-1K validation and save images for ALL needed classes
    in a single pass. Uses streaming=True to avoid downloading the full
    ~6.7GB dataset to disk (which overflows Docker container storage).

    Args:
        class_map: dict mapping class_name_substring -> {
            "output_dir": str,   — directory to save images
            "tag": str,          — filename prefix (e.g. "golden_retriever")
            "max_images": int,   — quota per class group
        }

    Returns:
        dict: {tag: [filepath, ...]} for each tag in class_map values
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "HuggingFace 'datasets' library not installed.\n"
            "  Fix: pip install datasets"
        )

    try:
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"ImageNet-1K load failed: {e}\n"
            f"\n"
            f"  ImageNet-1K requires TWO things:\n"
            f"  1. Accept the license at https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
            f"  2. Authenticate: huggingface-cli login\n"
            f"     Or pass HF_TOKEN environment variable:\n"
            f"       docker run --gpus all -e HF_TOKEN=hf_xxx ... bash run_pipeline.sh"
        )

    # Get label names from streaming dataset
    hf_label_names = dataset.features["label"].names

    # Build reverse index: label_idx -> (tag, output_dir, max_images)
    label_to_target = {}
    for wanted, info in class_map.items():
        wanted_lower = wanted.lower()
        matched = False
        for idx, hf_name in enumerate(hf_label_names):
            if wanted_lower in hf_name.lower():
                label_to_target[idx] = info
                print(f"    Matched '{wanted}' -> [{idx}] '{hf_name}'")
                matched = True
                break
        if not matched:
            print(f"    WARNING: No match for '{wanted}' in ImageNet labels")

    if not label_to_target:
        print("    ERROR: No matching ImageNet classes found!")
        return {}

    # Prepare output dirs and counters
    results = {}
    counters = {}
    for info in class_map.values():
        tag = info["tag"]
        os.makedirs(info["output_dir"], exist_ok=True)
        if tag not in results:
            results[tag] = []
            counters[tag] = 0

    # Compute total needed to know when to stop early
    tag_quotas = {}
    for info in class_map.values():
        tag = info["tag"]
        tag_quotas[tag] = info["max_images"]

    print(f"    Streaming ImageNet validation (single pass, {len(label_to_target)} classes)...")

    # Single pass through the stream
    for example in dataset:
        label_idx = example["label"]
        if label_idx not in label_to_target:
            continue

        info = label_to_target[label_idx]
        tag = info["tag"]

        # Check quota
        if counters[tag] >= tag_quotas[tag]:
            continue

        img = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        filename = f"imagenet_{tag}_{counters[tag]:04d}.jpg"
        filepath = os.path.join(info["output_dir"], filename)
        if not os.path.exists(filepath):
            img.save(filepath, quality=95)
        results[tag].append(filepath)
        counters[tag] += 1

        # Early exit when all quotas are filled
        if all(counters[t] >= tag_quotas[t] for t in tag_quotas):
            break

    for tag, paths in results.items():
        print(f"    {tag}: {len(paths)} images")

    return results


# ---------------------------------------------------------------------------
# Open Images Download (fallback)
# ---------------------------------------------------------------------------
def download_open_images(concept, labels, max_images=50, output_dir=None):
    """
    Download images from Open Images V7 using fiftyone.
    Falls back gracefully if fiftyone is not available.
    """
    if output_dir is None:
        output_dir = os.path.join(cfg.IMAGE_DIR, concept)
    os.makedirs(output_dir, exist_ok=True)

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print(f"    fiftyone not available. Skipping Open Images for {concept}.")
        return []

    downloaded = []
    for label in labels:
        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="validation",
                label_types=["detections"],
                classes=[label],
                max_samples=max_images // len(labels),
            )
            for sample in dataset:
                src = sample.filepath
                dst = os.path.join(output_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)
                downloaded.append(dst)
            dataset.delete()
        except Exception as e:
            print(f"    Open Images download failed for '{label}': {e}")

    return downloaded


# ---------------------------------------------------------------------------
# Build Complete Image Dataset
# ---------------------------------------------------------------------------
def build_concept_image_dataset():
    """
    Build the complete image dataset for all concepts.

    Uses a SINGLE streaming pass through ImageNet-1K to collect all needed
    classes (concepts + neutrals) without downloading the full dataset to disk.

    Source priority: ImageNet (streaming) > Open Images (fallback)
    All images are single-object, centered — ideal for concept activation.

    Returns:
        dict: {concept: [{"image_path": str, "source": str}, ...]}
    """
    print("\n  Building concept image dataset...")

    # Build unified class_map for single-pass streaming
    class_map = {}
    neutral_dir = os.path.join(cfg.IMAGE_DIR, "neutral")
    needed_per_neutral = max(
        10, cfg.MIN_IMAGES_PER_CONCEPT // len(cfg.NEUTRAL_IMAGENET_CLASS_NAMES)
    )

    # Add concept classes
    for concept in cfg.CONCEPTS:
        if concept in cfg.IMAGENET_CLASS_NAMES:
            concept_dir = os.path.join(cfg.IMAGE_DIR, concept)
            for class_name in cfg.IMAGENET_CLASS_NAMES[concept]:
                class_map[class_name] = {
                    "output_dir": concept_dir,
                    "tag": concept,
                    "max_images": cfg.MAX_IMAGES_PER_CONCEPT,
                }

    # Add neutral classes
    for class_name in cfg.NEUTRAL_IMAGENET_CLASS_NAMES:
        neutral_tag = "neutral_" + class_name.replace(" ", "_").lower()
        class_map[class_name] = {
            "output_dir": neutral_dir,
            "tag": neutral_tag,
            "max_images": needed_per_neutral,
        }

    # Single-pass streaming download
    print(f"\n  === IMAGENET STREAMING (all classes in one pass) ===")
    imagenet_results = download_all_imagenet_images(class_map)

    # Distribute results into concept dataset
    dataset = {}

    for concept in cfg.CONCEPTS:
        images = []
        inet_paths = imagenet_results.get(concept, [])
        for path in inet_paths:
            images.append({"image_path": path, "source": "imagenet"})

        print(f"\n  === {concept.upper()} ===")
        print(f"    ImageNet: {len(inet_paths)} images")

        # Open Images fallback if below minimum
        if len(images) < cfg.MIN_IMAGES_PER_CONCEPT:
            if concept in cfg.OPEN_IMAGES_LABELS:
                needed = cfg.MIN_IMAGES_PER_CONCEPT - len(images)
                print(f"    Need {needed} more. Trying Open Images...")
                oi_paths = download_open_images(
                    concept,
                    cfg.OPEN_IMAGES_LABELS[concept],
                    max_images=needed,
                )
                for path in oi_paths:
                    images.append({"image_path": path, "source": "open_images"})
                print(f"    Open Images: {len(oi_paths)} images")

        dataset[concept] = images
        print(f"    TOTAL {concept}: {len(images)} images")

        if len(images) < cfg.MIN_IMAGES_PER_CONCEPT:
            print(f"    WARNING: Only {len(images)} images "
                  f"(need {cfg.MIN_IMAGES_PER_CONCEPT})")

    # Collect neutral images from streaming results
    print(f"\n  === NEUTRAL ===")
    neutral_images = []
    for class_name in cfg.NEUTRAL_IMAGENET_CLASS_NAMES:
        neutral_tag = "neutral_" + class_name.replace(" ", "_").lower()
        paths = imagenet_results.get(neutral_tag, [])
        for path in paths:
            neutral_images.append({"image_path": path, "source": "imagenet"})
        print(f"    Neutral '{class_name}': {len(paths)} images")

    dataset["neutral"] = neutral_images
    print(f"    TOTAL neutral: {len(neutral_images)} images")

    return dataset


# ---------------------------------------------------------------------------
# Train / Test Split — Prevents Data Leakage
# ---------------------------------------------------------------------------
def split_train_test(image_dataset, train_ratio=None):
    """
    Split each concept's images into train (CAV training) and test (benchmarks).

    Images in the test set are NEVER used for CAV extraction, ensuring
    that R1/R2 benchmark scores are not inflated by memorization.

    Args:
        image_dataset: dict from build_concept_image_dataset()
        train_ratio: fraction for training (default from config)

    Returns:
        train_split: {concept: [images...]}
        test_split:  {concept: [images...]}
    """
    if train_ratio is None:
        train_ratio = cfg.TRAIN_SPLIT_RATIO

    train_split = {}
    test_split = {}

    for concept in cfg.CONCEPTS:
        images = image_dataset.get(concept, [])
        random.shuffle(images)

        n_train = max(1, int(len(images) * train_ratio))
        train_split[concept] = images[:n_train]
        test_split[concept] = images[n_train:]

        print(f"  {concept}: {len(train_split[concept])} train, "
              f"{len(test_split[concept])} test "
              f"(ratio={len(train_split[concept])/max(1,len(images)):.2f})")

    # Neutral images: all go to training (they're negatives for CAV, not benchmarked)
    train_split["neutral"] = image_dataset.get("neutral", [])
    test_split["neutral"] = []

    return train_split, test_split


# ---------------------------------------------------------------------------
# Generate CAV Training Pairs (from TRAIN split only)
# ---------------------------------------------------------------------------
def generate_cav_training_pairs(train_split):
    """
    Generate positive/negative pairs for CAV training.

    Uses ONLY the training split — test images are reserved for benchmarks.

    Positive: (concept_image, generic_prompt)
    Negative: (neutral_image, SAME generic_prompt)
    """
    prompt = cfg.CAV_TRAINING_PROMPT
    cav_training = {}

    for concept in cfg.CONCEPTS:
        concept_images = train_split.get(concept, [])
        neutral_images = list(train_split.get("neutral", []))

        if not concept_images or not neutral_images:
            print(f"  WARNING: Insufficient images for {concept} CAV training")
            cav_training[concept] = {"positive": [], "negative": []}
            continue

        n = min(len(concept_images), len(neutral_images))
        random.shuffle(neutral_images)

        positive = [(img["image_path"], prompt) for img in concept_images[:n]]
        negative = [(img["image_path"], prompt) for img in neutral_images[:n]]

        cav_training[concept] = {
            "positive": positive,
            "negative": negative,
        }
        print(f"  {concept}: {len(positive)} pos, {len(negative)} neg pairs (TRAIN only)")

    return cav_training


# ---------------------------------------------------------------------------
# Generate R1 Benchmark (from TEST split only)
# ---------------------------------------------------------------------------
def generate_r1_benchmark(test_split):
    """
    R1: Visual Question Answering with MCQ format.
    Uses ONLY the test split — these images were never seen during CAV training.

    Distractors are INTRA-DOMAIN (semantically close) for harder discrimination.
    """
    r1_benchmark = {}

    for concept in cfg.CONCEPTS:
        items = []
        correct_label = cfg.R1_CORRECT_LABELS[concept]
        concept_images = test_split.get(concept, [])
        distractors_pool = cfg.R1_DISTRACTORS_PER_CONCEPT[concept]

        for img_info in concept_images:
            # Select 3 random intra-domain distractors
            distractors = random.sample(distractors_pool, min(3, len(distractors_pool)))

            # Pad with generic distractors if needed
            while len(distractors) < 3:
                distractors.append("object")

            options = [correct_label] + distractors
            random.shuffle(options)

            items.append({
                "image_path": img_info["image_path"],
                "correct_answer": correct_label,
                "options": options,
            })

        r1_benchmark[concept] = items
        print(f"  R1 {concept}: {len(items)} items (TEST only, intra-domain distractors)")

    return r1_benchmark


# ---------------------------------------------------------------------------
# Generate R2 Benchmark (from TEST split only)
# ---------------------------------------------------------------------------
# Concept descriptions for R2a cross-modal similarity.
# Each concept gets multiple paraphrased text descriptions to avoid
# measuring similarity to a single phrasing artifact.
CONCEPT_DESCRIPTIONS = {
    # --- Broad concepts (original) ---
    "time": [
        "A clock showing the time",
        "A timepiece displaying hours and minutes",
        "An instrument for measuring time",
        "A device that shows the current hour",
        "A watch or clock",
    ],
    "place": [
        "A building or structure",
        "An architectural landmark",
        "A place where people gather",
        "An urban or natural location",
        "A scenic view of a place",
    ],
    "tools": [
        "A tool used for construction or repair",
        "A hand tool for manual work",
        "An implement for building things",
        "A piece of equipment for crafting",
        "A utility tool for fixing things",
    ],
    # --- Pilot V2: visually distinct concepts ---
    "dog": [
        "A dog",
        "A canine animal",
        "A pet dog",
        "A domestic dog breed",
        "A dog photographed",
    ],
    "car": [
        "A car",
        "An automobile",
        "A motor vehicle",
        "A passenger car",
        "A vehicle for transportation",
    ],
    "flower": [
        "A flower",
        "A flowering plant",
        "A blossom",
        "A decorative flower",
        "A botanical specimen",
    ],
}


def generate_r2_benchmark(test_split):
    """
    R2: Image-text semantic similarity pairs.
    Uses ONLY the test split.

    Used by both R2a (cross-modal similarity) and R2b (output similarity).
    R2a compares image token activations vs text-only concept embedding.
    R2b compares baseline vs intervened last-token activations.
    """
    r2_benchmark = {}

    for concept in cfg.CONCEPTS:
        pairs = []
        concept_images = test_split.get(concept, [])
        descriptions = CONCEPT_DESCRIPTIONS.get(concept, ["An object"])

        for img_info in concept_images[:cfg.R2_MIN_PAIRS * 2]:  # Cap at 2x min
            desc = random.choice(descriptions)
            pairs.append({
                "image_path": img_info["image_path"],
                "text_description": desc,
            })

        r2_benchmark[concept] = pairs
        print(f"  R2 {concept}: {len(pairs)} pairs (TEST only)")

    return r2_benchmark


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 60
    pilot_str = " [PILOT MODE]" if cfg.PILOT_MODE else ""
    print(f"\n{sep}")
    print(f"  Multimodal Data Generator — LLaVA-1.5 7B{pilot_str}")
    print(f"  Concepts: {cfg.CONCEPTS}")
    print(f"  Min images per concept: {cfg.MIN_IMAGES_PER_CONCEPT}")
    print(f"  Train/test split: {cfg.TRAIN_SPLIT_RATIO:.0%} / {1-cfg.TRAIN_SPLIT_RATIO:.0%}")
    print(f"{sep}")

    # Build image dataset
    image_dataset = build_concept_image_dataset()

    # Train/test split
    print(f"\n{sep}")
    print("  Train / Test Split")
    print(f"{sep}")
    train_split, test_split = split_train_test(image_dataset)

    # Generate CAV training pairs (TRAIN only)
    print(f"\n{sep}")
    print("  Generating CAV Training Pairs (TRAIN split)")
    print(f"{sep}")
    cav_training = generate_cav_training_pairs(train_split)

    # Generate R1 benchmark (TEST only)
    print(f"\n{sep}")
    print("  Generating R1 Benchmark (TEST split)")
    print(f"{sep}")
    r1_benchmark = generate_r1_benchmark(test_split)

    # Generate R2 benchmark (TEST only)
    print(f"\n{sep}")
    print("  Generating R2 Benchmark (TEST split)")
    print(f"{sep}")
    r2_benchmark = generate_r2_benchmark(test_split)

    # Save experiment data
    output = {
        "config": {
            "model": cfg.MODEL_NAME,
            "concepts": cfg.CONCEPTS,
            "pilot_mode": cfg.PILOT_MODE,
            "min_images_per_concept": cfg.MIN_IMAGES_PER_CONCEPT,
            "train_split_ratio": cfg.TRAIN_SPLIT_RATIO,
            "token_position": cfg.TOKEN_POSITION,
        },
        "image_dataset": {
            concept: [
                {"image_path": img["image_path"], "source": img["source"]}
                for img in imgs
            ]
            for concept, imgs in image_dataset.items()
        },
        "train_split": {
            concept: [
                {"image_path": img["image_path"], "source": img["source"]}
                for img in imgs
            ]
            for concept, imgs in train_split.items()
        },
        "test_split": {
            concept: [
                {"image_path": img["image_path"], "source": img["source"]}
                for img in imgs
            ]
            for concept, imgs in test_split.items()
        },
        "cav_training": {
            concept: {
                "positive": data["positive"],
                "negative": data["negative"],
            }
            for concept, data in cav_training.items()
        },
        "r1_benchmark": r1_benchmark,
        "r2_benchmark": r2_benchmark,
    }

    os.makedirs(os.path.dirname(cfg.DATA_FILE), exist_ok=True)
    with open(cfg.DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{sep}")
    print(f"  Saved: {cfg.DATA_FILE}")
    print(f"  Image counts:")
    for concept in cfg.CONCEPTS + ["neutral"]:
        n_total = len(image_dataset.get(concept, []))
        n_train = len(train_split.get(concept, []))
        n_test = len(test_split.get(concept, []))
        status = "OK" if n_total >= cfg.MIN_IMAGES_PER_CONCEPT else "LOW"
        print(f"    {concept}: {n_total} total ({n_train} train / {n_test} test) [{status}]")

    # Source distribution
    print(f"\n  Source distribution:")
    for concept in cfg.CONCEPTS:
        sources = defaultdict(int)
        for img in image_dataset.get(concept, []):
            sources[img["source"]] += 1
        parts = [f"{src}={cnt}" for src, cnt in sorted(sources.items())]
        print(f"    {concept}: {', '.join(parts)}")

    # Fail-fast validation: ensure we have enough data to run the experiment
    fatal_errors = []
    for concept in cfg.CONCEPTS:
        n_total = len(image_dataset.get(concept, []))
        n_train = len(train_split.get(concept, []))
        n_test = len(test_split.get(concept, []))
        if n_total == 0:
            fatal_errors.append(f"  {concept}: 0 images downloaded!")
        elif n_total < cfg.MIN_IMAGES_PER_CONCEPT:
            print(f"  WARNING: {concept} has only {n_total} images "
                  f"(minimum: {cfg.MIN_IMAGES_PER_CONCEPT})")
        if n_train == 0:
            fatal_errors.append(f"  {concept}: 0 training images!")
        if n_test == 0:
            fatal_errors.append(f"  {concept}: 0 test images!")

    n_neutral = len(image_dataset.get("neutral", []))
    if n_neutral == 0:
        fatal_errors.append("  neutral: 0 images downloaded!")

    if fatal_errors:
        print(f"\n{'='*60}")
        print("  FATAL: Cannot proceed — missing image data:")
        for err in fatal_errors:
            print(err)
        print(f"\n  Possible causes:")
        print(f"    1. HuggingFace not authenticated:")
        print(f"         huggingface-cli login")
        print(f"       Or: docker run -e HF_TOKEN=hf_xxx ...")
        print(f"    2. ImageNet-1K license not accepted:")
        print(f"         https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        print(f"    3. No internet access in container")
        print(f"{'='*60}")
        raise SystemExit(1)

    print(f"{sep}\n")


if __name__ == "__main__":
    main()
