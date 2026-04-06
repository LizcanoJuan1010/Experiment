"""
Evaluation Metrics for CAV Ablation Experiments
================================================
Core metrics used by the 4 retained tests:
  - evaluate_bea_association() — CCT (Camel & Cactus Test)
  - evaluate_bea_naming()      — BEA Confrontation Naming
  - evaluate_r2_ranking()      — BEA Synonym/Antonym Discrimination
  - _evaluate_mcq()            — Generic MCQ engine (shared)

Returns both aggregate scores AND per-item results for robust ANOVA.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

import experiment_config as cfg


def get_embedding(model, text, layer=None):
    """
    Extract sentence embedding from a specified layer's residual stream.

    Token position strategy controlled by cfg.TOKEN_POSITION:
      - "mean": Mean pooling over all tokens (recommended for sentence-level
                similarity; Reimers & Gurevych, 2019).
      - "last": Last token only (original, kept for backward compatibility).
    """
    if layer is None:
        layer = cfg.R2_EMBEDDING_LAYER
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


# ===================================================================
# BEA-aligned metrics
# ===================================================================

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
    Distractors are associations of OTHER items in the same category.
    Uses loss-based MCQ scoring.

    Same engine as evaluate_r1() but with BEA association benchmark items.
    """
    return _evaluate_mcq(model, dataset, desc="B2-Assoc")


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
        layer = getattr(cfg, "R2_EMBEDDING_LAYER", 31)

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
