"""
Evaluation Metrics for Semantic Aphasia Experiment
===================================================
R1: Hypernym Classification Accuracy (Δacc)
R2: Semantic Similarity Drop (Δsim)

BEA-aligned metrics (same MCQ engine as R1, different benchmark items):
B1: BEA Naming — Confrontation naming with intra-category distractors
B2: BEA Association — Functional semantic association
B3: BEA Odd-One-Out — Semantic intruder detection

Returns both aggregate scores AND per-item results for robust ANOVA.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

import experiment_config as cfg


def get_embedding(model, text, layer=11):
    """
    Extract sentence embedding from a specified layer's residual stream.

    Token position strategy controlled by cfg.TOKEN_POSITION:
      - "mean": Mean pooling over all tokens (recommended for sentence-level
                similarity; Reimers & Gurevych, 2019).
      - "last": Last token only (original, kept for backward compatibility).
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    token_method = getattr(cfg, "TOKEN_POSITION", "mean")
    with torch.no_grad():
        _, cache = model.run_with_cache(text, names_filter=[hook_name])
        act = cache[hook_name]  # [1, seq_len, d_model]
        if token_method == "mean":
            embedding = act[0].mean(dim=0)  # Mean pooling
        else:
            embedding = act[0, -1, :]       # Last token
    return embedding


def _score_candidate(model, prompt, candidate):
    """
    Score a candidate answer for MCQ evaluation.

    Scoring modes (controlled by cfg.R1_SCORING_MODE):
      - "continuation": Log-probability of answer tokens only (after prompt).
            Avoids prompt-length bias. Recommended.
      - "full": Negative mean loss over all tokens (legacy).
    """
    scoring_mode = getattr(cfg, "R1_SCORING_MODE", "continuation")

    if scoring_mode == "continuation":
        prompt_tokens = model.to_tokens(prompt)
        full_tokens = model.to_tokens(prompt + " " + candidate)
        n_prompt = prompt_tokens.shape[1]

        with torch.no_grad():
            logits = model(full_tokens)  # [1, seq_len, vocab]
            log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
            total_log_prob = 0.0
            n_answer_tokens = full_tokens.shape[1] - n_prompt
            for t in range(n_prompt, full_tokens.shape[1]):
                token_id = full_tokens[0, t]
                total_log_prob += log_probs[t - 1, token_id].item()
            return total_log_prob / max(n_answer_tokens, 1)
    else:
        # Legacy: mean loss over full text (prompt + answer)
        loss = model(prompt + " " + candidate, return_type="loss")
        return -loss.item()


def evaluate_r1(model, r1_dataset):
    """
    R1: Hypernym Classification Accuracy.

    For each MCQ, compares model score across all candidate answers.
    Scoring mode controlled by cfg.R1_SCORING_MODE.

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
            score = _score_candidate(model, prompt, cand)
            candidate_scores.append(score)

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

    Computes cosine similarity between sentence embeddings of
    sentence pairs. High similarity = model recognizes paraphrases.
    Embedding method controlled by cfg.TOKEN_POSITION (mean/last).

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


def evaluate_r2_ranking(model, synonym_pairs, non_synonym_pairs, layer=None):
    """
    R2-Rank: AUC-based ranking metric for semantic discrimination.

    Measures whether synonym pairs have higher cosine similarity than
    non-synonym pairs from the same category. AUC > 0.5 means the model
    preserves semantic granularity; ablation should reduce this.

    Returns:
        dict with r2_rank_auc, syn_mean, nonsyn_mean, gap
    """
    if layer is None:
        layer = getattr(cfg, "R2_EMBEDDING_LAYER", 7)

    syn_sims = []
    for s1, s2 in synonym_pairs:
        emb1 = get_embedding(model, s1, layer=layer)
        emb2 = get_embedding(model, s2, layer=layer)
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        syn_sims.append(sim)

    nonsyn_sims = []
    for s1, s2 in non_synonym_pairs:
        emb1 = get_embedding(model, s1, layer=layer)
        emb2 = get_embedding(model, s2, layer=layer)
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        nonsyn_sims.append(sim)

    # AUC: proportion of (synonym, non-synonym) pairs where synonym > non-synonym
    correct = sum(1 for s in syn_sims for n in nonsyn_sims if s > n)
    total = len(syn_sims) * len(nonsyn_sims)
    auc = correct / total if total > 0 else 0.5

    import numpy as np
    return {
        "r2_rank_auc": auc,
        "syn_mean": float(np.mean(syn_sims)) if syn_sims else 0.0,
        "nonsyn_mean": float(np.mean(nonsyn_sims)) if nonsyn_sims else 0.0,
        "gap": float(np.mean(syn_sims) - np.mean(nonsyn_sims))
            if syn_sims and nonsyn_sims else 0.0,
    }


# ===================================================================
# BEA-aligned metrics
# ===================================================================
# All three BEA benchmarks use the same MCQ format as R1:
#   {"question": str, "options": {"correct": str, "distractors": [...]}}
# The evaluation engine is identical — only the benchmark items differ.
# These wrappers provide named entry points for clarity in experiments.

def evaluate_bea_naming(model, dataset):
    """
    B1: BEA Confrontation Naming (Denominación por confrontación).

    Given a functional description, select the correct word among
    intra-category distractors. Uses loss-based MCQ scoring.

    Same engine as evaluate_r1() but with BEA naming benchmark items.
    """
    return _evaluate_mcq(model, dataset, desc="B1-Naming")


def evaluate_bea_association(model, dataset):
    """
    B2: BEA Semantic Association (Asociación semántica).

    Given a target word, select the most functionally associated word.
    Uses sentence log-probability scoring: builds a simple declarative
    sentence for each option and picks the one GPT-2 finds most natural.

    This avoids instruction-style prompts that non-instruction-tuned
    models (GPT-2, Pythia) cannot interpret, which caused chance-level
    baselines (~0.25).

    Method: For each candidate, compute mean log-prob of the continuation
    after a simple prompt like "A camel is related to ___".
    The model picks whichever completion is most natural.
    """
    correct_count = 0
    total = len(dataset)
    per_item = []

    for item in tqdm(dataset, desc="B2-Assoc", leave=False):
        concept = item["provenance"]["target_word"]
        correct_ans = item["options"]["correct"]
        distractors = item["options"]["distractors"]

        # Simple prompt natural for autoregressive LMs
        prompt = f"A {concept} is related to"
        candidates = [correct_ans] + distractors
        candidate_scores = []

        for cand in candidates:
            score = _score_candidate(model, prompt, cand)
            candidate_scores.append(score)

        best_idx = torch.argmax(torch.tensor(candidate_scores)).item()
        is_correct = candidates[best_idx] == correct_ans
        if is_correct:
            correct_count += 1

        per_item.append({
            "question": f"{prompt} ___",
            "correct_answer": correct_ans,
            "model_answer": candidates[best_idx],
            "is_correct": is_correct,
            "scores": {c: s for c, s in zip(candidates, candidate_scores)},
        })

    accuracy = correct_count / total if total > 0 else 0.0
    return accuracy, per_item


def evaluate_bea_oddoneout(model, dataset):
    """
    B3: BEA Odd-One-Out (Detección de intruso semántico).

    Given 4 words (3 from one category + 1 intruder), identify the word
    that does not belong. Uses loss-based MCQ scoring.

    Same engine as evaluate_r1() but with BEA odd-one-out benchmark items.
    """
    return _evaluate_mcq(model, dataset, desc="B3-OddOut")


def _evaluate_mcq(model, dataset, desc="MCQ"):
    """
    Generic MCQ evaluator.

    For each item, compares model score across all candidate answers.
    Scoring mode controlled by cfg.R1_SCORING_MODE.

    Args:
        model: HookedTransformer model
        dataset: list of dicts with "question" and "options" keys
        desc: progress bar description

    Returns:
        accuracy: float (aggregate)
        per_item: list of dicts with per-question results
    """
    correct_count = 0
    total = len(dataset)
    per_item = []

    for item in tqdm(dataset, desc=desc, leave=False):
        question = item["question"]
        correct_ans = item["options"]["correct"]
        distractors = item["options"]["distractors"]

        prompt = f"Question: {question}\nAnswer:"
        candidates = [correct_ans] + distractors
        candidate_scores = []

        for cand in candidates:
            score = _score_candidate(model, prompt, cand)
            candidate_scores.append(score)

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
