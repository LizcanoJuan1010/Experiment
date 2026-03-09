"""
Concept Registry
=================
Loads concept definitions from concepts/*.json files.
Single source of truth for all per-concept data.

Adding a new concept requires only creating a JSON file in concepts/
with at minimum: name, hypernym_label, curated_nodes (>= 20 words).
"""
import json
import os
import glob

_CONCEPTS_DIR = os.path.join(os.path.dirname(__file__), "concepts")
_registry = {}


def _load_all():
    """Scan concepts/ directory and load all JSON files."""
    if _registry:
        return
    pattern = os.path.join(_CONCEPTS_DIR, "*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("name")
        assert name, f"{path}: missing 'name' field"
        assert "hypernym_label" in data, f"{path}: missing 'hypernym_label'"
        assert "curated_nodes" in data, f"{path}: missing 'curated_nodes'"
        assert len(data["curated_nodes"]) >= 20, (
            f"{path}: curated_nodes has {len(data['curated_nodes'])} items, need >= 20"
        )
        _registry[name] = data


def get_concept(name):
    """Return concept definition dict, or raise KeyError."""
    _load_all()
    return _registry[name]


def get_all_concept_names():
    """Return sorted list of all available concept names."""
    _load_all()
    return sorted(_registry.keys())


def get_curated_nodes():
    """Return {concept: [words]} dict (replaces data_gen.CURATED_NODES)."""
    _load_all()
    return {name: data["curated_nodes"] for name, data in _registry.items()}


def get_hypernym_labels():
    """Return {concept: label} dict (replaces data_gen.HYPERNYM_LABELS)."""
    _load_all()
    return {name: data["hypernym_label"] for name, data in _registry.items()}


def get_curated_synonyms():
    """Return {concept: [(w1,w2),...]} dict (replaces data_gen.CURATED_SYNONYMS)."""
    _load_all()
    return {name: [tuple(p) for p in data.get("curated_synonyms", [])]
            for name, data in _registry.items()}


def get_curated_paraphrases():
    """Return {concept: [(s1,s2),...]} dict (replaces data_gen.CURATED_PARAPHRASES)."""
    _load_all()
    return {name: [tuple(p) for p in data.get("curated_paraphrases", [])]
            for name, data in _registry.items()}


def get_non_synonym_pairs():
    """Return {concept: [(w1,w2),...]} dict."""
    _load_all()
    return {name: [tuple(p) for p in data.get("non_synonym_pairs", [])]
            for name, data in _registry.items()}


def get_curated_antonyms():
    """Return {concept: [(w1,w2),...]} dict."""
    _load_all()
    return {name: [tuple(p) for p in data.get("curated_antonyms", [])]
            for name, data in _registry.items()}


def get_conceptnet_seeds():
    """Return {concept: {isa_targets: set, related_targets: set}} dict."""
    _load_all()
    result = {}
    for name, data in _registry.items():
        seeds = data.get("conceptnet_seeds")
        if seeds:
            result[name] = {
                "isa_targets": set(seeds.get("isa_targets", [])),
                "related_targets": set(seeds.get("related_targets", [])),
            }
    return result


def get_wordnet_roots():
    """Return {concept: [root_synset_names]} for WordNet validation."""
    _load_all()
    return {name: data.get("wordnet_hypernym_roots", [])
            for name, data in _registry.items()}
