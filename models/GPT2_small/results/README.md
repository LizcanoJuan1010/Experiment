# Resultados Principales

Resultados de todos los experimentos de ablacion conceptual en GPT-2-small.

## Experimentos completados

### CCT (Camel and Cactus Test)

| Archivo | Descripcion |
|---------|-------------|
| `cct_test_results.json` | 4,320 evaluaciones: 27 conceptos x 2 metodos x 4 capas x 5 alphas |

### Benchmarks BEA

| Archivo | Descripcion |
|---------|-------------|
| `bea_naming_test_results.json` | BEA naming con ablacion factorial |
| `bea_oddoneout_test_results.json` | BEA odd-one-out con ablacion factorial |
| `bea_synonym_test_results.json` | BEA synonym con ablacion factorial |
| `bea_anova_results.json` | ANOVA: method x layer x alpha para los 3 tests BEA |
| `bea_anova_table_alpha.tex` | Tabla LaTeX ANOVA (efecto alpha) |
| `bea_anova_table_layer.tex` | Tabla LaTeX ANOVA (efecto capa) |

### Caperucita Test

| Archivo | Descripcion |
|---------|-------------|
| `caperucita_test_results.json` | 320 condiciones de ablacion del concepto wolf |

### Experimento factorial original

| Archivo | Descripcion |
|---------|-------------|
| `results_factorial.csv` | R1, R2 por combinacion de factores |
| `results_baseline.json` | Metricas base sin intervencion |
| `results_specificity.csv` | Efecto on-target vs off-target |

### Tests cualitativos

| Archivo | Descripcion |
|---------|-------------|
| `aphasia_test_results.json` | Generacion de texto bajo ablacion |
| `vision_aphasia_test_results.json` | Ablacion en dominio visual (BLIP) |

### Validacion

| Archivo | Descripcion |
|---------|-------------|
| `validation_report.json` | Suite T01-T20: 17 PASS, 2 SKIP, 1 FAIL |
| `corpus_training_report.json` | Metadata de extraccion de corpus |

## Figuras (`figures/`)

### Figuras principales (PDF + PNG)

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

### Figuras BEA (`figures/bea_tests/`)

6 plots de interaccion method x layer para naming, odd-one-out y synonym.

## Generado por

- [`experiment_runner.py`](../experiment_runner.py) — Experimento factorial
- [`test/cct_test.py`](../test/cct_test.py) — CCT test
- [`test/bea_naming_test.py`](../test/bea_naming_test.py) — BEA naming
- [`test/bea_oddoneout_test.py`](../test/bea_oddoneout_test.py) — BEA odd-one-out
- [`test/bea_synonym_test.py`](../test/bea_synonym_test.py) — BEA synonym
- [`test/caperucita_test.py`](../test/caperucita_test.py) — Caperucita test
- [`test/aphasia_test.py`](../test/aphasia_test.py) — Test cualitativo
- [`bea_anova_analysis.py`](../bea_anova_analysis.py) — ANOVA BEA
- [`generate_figures.py`](../generate_figures.py) — Figuras F01-F08
- [`generate_figures_deep.py`](../generate_figures_deep.py) — Figuras F10-F14
