"""
CAV Refinement — Orthogonal Subspace Projection
=================================================
Purifies each CAV by removing its component in the subspace spanned
by the other concepts' CAVs using the correct oblique projection formula:

    v_pure = v_A - V (V^T V)^{-1} V^T v_A

where V = [v_B | v_C | ...] is the matrix of the other concept CAVs.

This accounts for non-orthogonality between v_B, v_C, etc., avoiding the
over-correction that naive individual projection subtraction introduces.

Input:  cavs/{concept}_svm_layer{L}.pt
Output: cavs_refined/{concept}_svm_layer{L}.pt
"""

import os
import torch
from itertools import combinations

CAV_DIR = "cavs"
OUTPUT_DIR = "cavs_refined"
CONCEPTS = ["time", "place", "tools"]
LAYERS = [6, 9, 10]


def load_cav(concept, layer, method="svm"):
    path = os.path.join(CAV_DIR, f"{concept}_{method}_layer{layer}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=True)
    return None


def subspace_projection(v, others):
    """
    Compute the projection of v onto the subspace spanned by 'others'.
    Uses V (V^T V)^{-1} V^T v to handle non-orthogonal basis vectors.
    """
    V = torch.stack(others, dim=1)           # [d_model, k]
    G = V.T @ V                               # Gram matrix [k, k]
    coeffs = torch.linalg.solve(G, V.T @ v)  # [k]
    return V @ coeffs                          # [d_model]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("CAV Refinement — Orthogonal Subspace Projection")
    print("=" * 60)

    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---")

        # Load and normalize all vectors for this layer
        vectors = {}
        for c in CONCEPTS:
            v = load_cav(c, layer)
            if v is not None:
                vectors[c] = v / v.norm()

        # Print original correlations
        print("Original correlations:")
        for c1, c2 in combinations(CONCEPTS, 2):
            if c1 in vectors and c2 in vectors:
                sim = torch.dot(vectors[c1], vectors[c2]).item()
                print(f"  {c1} vs {c2}: {sim:+.4f}")

        # Purify each concept by projecting out the subspace of the others
        refined_vectors = {}
        for target in CONCEPTS:
            if target not in vectors:
                continue

            others = [vectors[c] for c in CONCEPTS if c != target and c in vectors]
            if not others:
                refined_vectors[target] = vectors[target].clone()
                continue

            print(f"\nPurifying '{target}'...")

            proj = subspace_projection(vectors[target], others)
            v_pure = vectors[target] - proj

            # Re-normalize
            current_norm = v_pure.norm()
            if current_norm > 1e-6:
                v_pure = v_pure / current_norm
                print(f"  Residual norm before renorm: {current_norm:.4f}")
            else:
                print(f"  WARNING: Vector for {target} vanished!")

            refined_vectors[target] = v_pure

            # Save
            filename = f"{target}_svm_layer{layer}.pt"
            torch.save(v_pure, os.path.join(OUTPUT_DIR, filename))
            print(f"  Saved: {OUTPUT_DIR}/{filename}")

        # Verify: each refined vector must be orthogonal to original others
        print(f"\nVerification — refined vs original (must be ~0):")
        for target in CONCEPTS:
            if target not in refined_vectors:
                continue
            for other in CONCEPTS:
                if target == other or other not in vectors:
                    continue
                dot = torch.dot(refined_vectors[target], vectors[other]).item()
                status = "OK" if abs(dot) < 1e-5 else "FAIL"
                print(f"  refined_{target} . original_{other}: {dot:+.2e}  [{status}]")

        # Also show correlations between refined vectors (informational)
        print(f"\nCorrelations between refined vectors (layer {layer}):")
        for c1, c2 in combinations(CONCEPTS, 2):
            if c1 in refined_vectors and c2 in refined_vectors:
                sim = torch.dot(refined_vectors[c1], refined_vectors[c2]).item()
                print(f"  {c1} vs {c2}: {sim:+.4f}  (was {torch.dot(vectors[c1], vectors[c2]).item():+.4f})")

    print(f"\n{'=' * 60}")
    print("Done. Refined CAVs saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
