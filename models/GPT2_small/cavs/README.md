# Vectores de Activacion de Concepto (CAVs)

CAVs extraidos de GPT-2-small para 30 conceptos semanticos concretos, usando dos metodos de extraccion, en las capas 3-6, 9-10 del residual stream.

## Metodos de extraccion

- **Mean Difference**: `CAV = normalize(mean(pos) - mean(neg))` — aproximacion directa.
- **SVM**: `CAV = normalize(clf.coef_[0])` — vector normal al hiperplano de LinearSVC (Kim et al. 2018).

Ambos metodos producen vectores **L2-normalizados** (norma unitaria). Para vectores sin normalizar usados en el experimento multi-capa, ver [`cavs_steering/`](../cavs_steering/).

## Contenido

### Vectores CAV

| Patron | Descripcion | Capas |
|--------|-------------|-------|
| `{concepto}_svm_layer{L}.pt` | CAV SVM normalizado | 3, 4, 5, 6, 9, 10 |
| `{concepto}_mean_diff_layer{L}.pt` | CAV Mean Diff normalizado | 6, 9, 10 |
| `{concepto}_acts_layer{L}.pt` | Activaciones raw (pos+neg) | 3, 4, 5, 6, 9, 10 |

Conceptos: los 30 definidos en `concepts/*.json` (animales, objetos, profesiones, personajes).

Los CAVs de wolf se extraen por separado con `extract_wolf_cavs.py` en las 12 capas para el Caperucita Test.

### Reportes de validacion

| Archivo | Descripcion |
|---------|-------------|
| `extraction_report.json` | Metricas de extraccion: SVM CV accuracy, similitudes coseno entre metodos |
| `extraction_report_individual.json` | Metricas por concepto y por capa |
| `statistical_validation.json` | TCAV permutation test, selectividad, bootstrap CI, Cohen's d |

## Generado por

- [`cav_extraction.py`](../cav_extraction.py) — Extraccion de vectores CAV (30 conceptos)
- [`extract_wolf_cavs.py`](../extract_wolf_cavs.py) — Extraccion de CAVs wolf (12 capas)
- [`cav_statistical_tests.py`](../cav_statistical_tests.py) — Validacion estadistica
