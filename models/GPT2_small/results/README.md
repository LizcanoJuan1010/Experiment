# Resultados del Experimento Factorial Original

Resultados del primer experimento de ablacion conceptual: intervencion single-layer usando CAVs normalizados con tecnicas de sustraccion y proyeccion.

## Diseno experimental

- **Conceptos**: time, place, tools
- **Tecnicas**: sustraccion (`act -= alpha * CAV`) y proyeccion (`act -= alpha * (act . CAV) * CAV`)
- **Capas**: 6 (semantica), 10 (sintactica), multi (6+9+10)
- **Alphas**: 0.0, 1.5, 3.5, 6.0
- **Metodos CAV**: mean_diff, svm

## Archivos

| Archivo | Descripcion |
|---------|-------------|
| `results_factorial.csv` | Resultados completos: R1, R2 por cada combinacion de factores |
| `results_baseline.json` | Metricas base sin intervencion |
| `results_specificity.csv` | Test de especificidad: efecto on-target vs off-target |
| `aphasia_test_results.json` | Test cualitativo: generacion de texto bajo ablacion |
| `vision_aphasia_test_results.json` | Test de ablacion en dominio visual (BLIP) |
| `figures/` | Figuras de publicacion (PDF + PNG) |

### Figuras (`figures/`)

| Figura | Descripcion |
|--------|-------------|
| `fig01_dose_response_r1` | Curva dosis-respuesta R1 |
| `fig02_dose_response_r2` | Curva dosis-respuesta R2 |
| `fig03_removal_ratio` | Ratio de remocion conceptual |
| `fig04_specificity_heatmap` | Mapa de calor de especificidad |
| `fig05_technique_comparison` | Comparacion sustraccion vs proyeccion |
| `fig06_cav_cosine_heatmap` | Similitud coseno entre CAVs |
| `fig07_svm_accuracy` | Precision SVM por capa y concepto |
| `fig08_cross_method_cosine` | Coseno entre metodos (SVM vs mean_diff) |
| `fig10_validation_dashboard` | Dashboard de validacion |
| `fig11_tcav_null_dist` | Distribucion nula TCAV |
| `fig12_selectivity` | Test de selectividad |
| `fig13_bootstrap_ci` | Intervalos de confianza bootstrap |
| `fig14_cohens_d_ci` | Cohen's d con intervalos de confianza |

Cada figura existe en formato `.pdf` (vectorial) y `.png` (rasterizado).

## Generado por

- [`experiment_runner.py`](../experiment_runner.py) — Experimento factorial
- [`aphasia_test.py`](../aphasia_test.py) — Test cualitativo de texto
- [`vision_aphasia_test.py`](../vision_aphasia_test.py) — Test visual
- [`generate_figures.py`](../generate_figures.py) — Figuras F01-F08
- [`generate_figures_deep.py`](../generate_figures_deep.py) — Figuras F10-F14

## Resultado principal

Efectos debiles: ΔR1 maximo ~0.07 (7%). Esto motivo el desarrollo del enfoque multi-capa con vectores sin normalizar (ver [`results_multilayer/`](../results_multilayer/)).
