# Vectores de Activacion de Concepto (CAVs)

Contiene los CAVs extraidos de GPT-2-small para los tres conceptos semanticos (time, place, tools), usando dos metodos de extraccion, en las capas 3-6, 9-10 del residual stream.

## Metodos de extraccion

- **Mean Difference**: `CAV = normalize(mean(pos) - mean(neg))` — aproximacion directa.
- **SVM**: `CAV = normalize(clf.coef_[0])` — vector normal al hiperplano de LinearSVC (Kim et al. 2018).

Ambos metodos producen vectores **L2-normalizados** (norma unitaria). Para vectores sin normalizar usados en el experimento multi-capa, ver [`cavs_steering/`](../cavs_steering/).

## Contenido

### Vectores CAV (48 archivos .pt)

| Patron | Descripcion | Capas |
|--------|-------------|-------|
| `{concepto}_svm_layer{L}.pt` | CAV SVM normalizado | 3, 4, 5, 6, 9, 10 |
| `{concepto}_mean_diff_layer{L}.pt` | CAV Mean Diff normalizado | 6, 9, 10 |
| `{concepto}_acts_layer{L}.pt` | Activaciones raw (pos+neg) | 3, 4, 5, 6, 9, 10 |

Conceptos: `time`, `place`, `tools`.

### Reportes de validacion

| Archivo | Descripcion |
|---------|-------------|
| `extraction_report.json` | Metricas de extraccion: SVM CV accuracy, similitudes coseno entre metodos, correlaciones entre conceptos |
| `statistical_validation.json` | Tests estadisticos: TCAV permutation test, selectividad, bootstrap CI, Cohen's d |

## Generado por

- [`cav_extraction.py`](../cav_extraction.py) — Extraccion de vectores CAV
- [`cav_statistical_tests.py`](../cav_statistical_tests.py) — Validacion estadistica

## Notas

- Los vectores SVM tienen CV accuracy 94-99% (5-fold stratified).
- La correlacion time-place es la mas alta (~0.30 SVM, ~0.47 mean_diff), indicando entanglement. Ver [`cavs_refined/`](../cavs_refined/) para vectores purificados ortogonalmente.
