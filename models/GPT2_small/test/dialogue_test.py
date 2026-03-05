"""
Semantic Aphasia Interactive Simulator
=======================================
Interactive chat with GPT-2 where you can ablate concept CAVs
in real time to observe how the model's language changes.

Supports: time, place, tools (and combinations).

Usage (inside Docker):
    python dialogue_test.py
"""

import torch
import os
from functools import partial
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt2-small"
CAV_DIR = "cavs"
LAYER = 6
CAV_METHOD = "mean_diff"
ALPHA = 5.0


# ---------------------------------------------------------------------------
# Ablation Hook
# ---------------------------------------------------------------------------
def ablation_hook(resid_post, hook, cav, alpha):
    return resid_post - alpha * cav


def generate_response(model, prompt, max_new_tokens=30):
    input_ids = model.to_tokens(prompt)
    output_ids = model.generate(
        input_ids, max_new_tokens=max_new_tokens,
        temperature=0.7, verbose=False
    )
    return model.to_string(output_ids[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    model.eval()

    # Load CAVs
    concepts = {}
    for c in ["time", "place", "tools"]:
        path = os.path.join(CAV_DIR, f"{c}_{CAV_METHOD}_layer{LAYER}.pt")
        if os.path.exists(path):
            concepts[c] = torch.load(path).to(model.cfg.device)
            print(f"  Loaded CAV: {c.upper()}")
        else:
            print(f"  Warning: CAV for {c} not found ({path})")

    print("\n--- Semantic Aphasia Simulator ---")
    print(f"    Layer: {LAYER}, Method: {CAV_METHOD}, Alpha: {ALPHA}")
    print(f"    Available concepts: {list(concepts.keys())}")

    MODES = {
        "1": ("Normal", []),
        "2": ("Ablate TIME", ["time"]),
        "3": ("Ablate PLACE", ["place"]),
        "4": ("Ablate TOOLS", ["tools"]),
        "5": ("Ablate TIME+PLACE", ["time", "place"]),
        "6": ("Ablate TIME+TOOLS", ["time", "tools"]),
        "7": ("Ablate PLACE+TOOLS", ["place", "tools"]),
        "8": ("Ablate ALL", ["time", "place", "tools"]),
    }

    while True:
        print("\nSelect Mode:")
        for k, (label, _) in MODES.items():
            print(f"  [{k}] {label}")
        print("  [q] Quit")

        mode = input("\nChoice: ").strip()
        if mode == "q":
            break
        if mode not in MODES:
            print("Invalid selection.")
            continue

        label, ablate_concepts = MODES[mode]
        print(f"\nMode: {label}")

        # Build hooks
        hooks_to_apply = []
        for c in ablate_concepts:
            if c in concepts:
                hook_name = f"blocks.{LAYER}.hook_resid_post"
                hook_fn = partial(ablation_hook, cav=concepts[c], alpha=ALPHA)
                hooks_to_apply.append((hook_name, hook_fn))

        print("Type 'back' to change mode.\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "back":
                break
            if not user_input:
                continue

            if hooks_to_apply:
                with model.hooks(fwd_hooks=hooks_to_apply):
                    response = generate_response(model, user_input)
            else:
                response = generate_response(model, user_input)

            response_text = response[len(user_input):]
            print(f"Model: {response_text.strip()}")
            print("-" * 30)


if __name__ == "__main__":
    main()
