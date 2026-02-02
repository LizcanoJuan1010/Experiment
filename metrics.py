"""
Evaluation Metrics for Semantic Aphasia Experiment
===================================================
R1: Hypernym Classification Accuracy (Δacc)
R2: Semantic Similarity Drop (Δsim)

Returns both aggregate scores AND per-item results for robust ANOVA.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def get_embedding(model, text, layer=11):
    """Extract last-token embedding from a specified layer's residual stream."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        _, cache = model.run_with_cache(text, names_filter=[hook_name])
        embedding = cache[hook_name][0, -1, :]
    return embedding


def evaluate_r1(model, r1_dataset):
    """
    R1: Hypernym Classification Accuracy.

    For each MCQ, compares model loss across all candidate answers.
    The candidate with lowest loss (highest likelihood) is the model's choice.

    Returns:
        accuracy: float (aggregate)
        per_item: list of dicts with per-question results
    """
    correct_count = 0
    total = len(r1_dataset)
    per_item = []

    for item in tqdm(r1_dataset, desc="R1", leave=False):
        question = item["question"]
        correct_ans = item["options"]["correct"]
        distractors = item["options"]["distractors"]

        prompt = f"Question: {question}\nAnswer:"
        candidates = [correct_ans] + distractors
        candidate_scores = []

        for cand in candidates:
            full_text = prompt + " " + cand
            loss = model(full_text, return_type="loss")
            candidate_scores.append(-loss.item())

        best_idx = torch.argmax(torch.tensor(candidate_scores)).item()
        is_correct = candidates[best_idx] == correct_ans
        if is_correct:
            correct_count += 1

        per_item.append({
            "question": question,
            "correct_answer": correct_ans,
            "model_answer": candidates[best_idx],
            "is_correct": is_correct,
            "scores": {c: s for c, s in zip(candidates, candidate_scores)},
        })

    accuracy = correct_count / total if total > 0 else 0.0
    return accuracy, per_item


def evaluate_r2(model, r2_dataset, layer=11):
    """
    R2: Semantic Similarity.

    Computes cosine similarity between last-token embeddings of
    sentence pairs. High similarity = model recognizes paraphrases.

    Returns:
        mean_similarity: float (aggregate)
        per_pair: list of dicts with per-pair results
    """
    per_pair = []

    for pair in tqdm(r2_dataset, desc="R2", leave=False):
        s1, s2 = pair[0], pair[1]
        emb1 = get_embedding(model, s1, layer=layer)
        emb2 = get_embedding(model, s2, layer=layer)

        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

        per_pair.append({
            "s1": s1,
            "s2": s2,
            "similarity": sim,
        })

    similarities = [p["similarity"] for p in per_pair]
    mean_sim = sum(similarities) / len(similarities) if similarities else 0.0
    return mean_sim, per_pair
