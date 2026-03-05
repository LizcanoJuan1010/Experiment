# Afasia Semantica Artificial en GPT-2

Simulacion experimental de afasia semantica categoria-especifica en GPT-2-small mediante ablacion causal con Concept Activation Vectors (CAVs). El proyecto extrae representaciones lineales de tres conceptos superordinados (tiempo, lugar, herramientas), las valida estadisticamente, y las usa para remover selectivamente conocimiento conceptual del modelo — analogamente a como lesiones cerebrales focales producen deficits semanticos selectivos en pacientes humanos.

> **Analisis integral del proyecto:** Ver [../../ANALISIS_TESIS.md](../../ANALISIS_TESIS.md) — verificacion de hipotesis, hallazgos principales, que falta mejorar y factores para futuras pruebas.

## Estructura del Proyecto

```
TESIS CODIGO/
├── cavs/                          # CAVs normalizados (extraccion original)
├── cavs_refined/                  # CAVs purificados ortogonalmente
├── cavs_steering/                 # Vectores de steering SVM sin normalizar
├── cavs_steering_meandiff/        # Vectores de steering mean-diff (comparacion)
├── results/                       # Resultados experimento factorial original
├── results_multilayer/            # Resultados experimento multi-capa SVM
├── results_multilayer_meandiff/   # Resultados multi-capa mean-diff (comparacion)
├── Benchmark_construction/        # Construccion de benchmarks R1 y R2
├── images/                        # Imagenes de prueba (dominio visual)
├── docs/                          # Documentacion de tesis
├── *.py                           # Scripts del pipeline (ver tabla abajo)
└── experiment_data.json           # Datos de entrenamiento generados
```

### Directorios

| Directorio | Descripcion |
|------------|-------------|
| [`cavs/`](cavs/) | CAVs normalizados (Mean Diff + SVM) para capas 3-6, 9-10. Incluye reportes de validacion estadistica. |
| [`cavs_refined/`](cavs_refined/) | CAVs purificados via proyeccion de subespacio ortogonal (eliminan entanglement entre conceptos). |
| [`cavs_steering/`](cavs_steering/) | Vectores SVM sin normalizar para steering multi-capa (12 capas). **Usado en el mejor experimento.** |
| [`cavs_steering_meandiff/`](cavs_steering_meandiff/) | Vectores mean-diff sin normalizar (comparacion contra SVM). |
| [`results/`](results/) | Experimento factorial original: ablacion single-layer con CAVs normalizados + figuras de publicacion. |
| [`results_multilayer/`](results_multilayer/) | Experimento multi-capa con SVM: 39 condiciones, ΔR1 hasta +0.33. **Mejor resultado.** |
| [`results_multilayer_meandiff/`](results_multilayer_meandiff/) | Experimento multi-capa con mean-diff: ΔR1 hasta +0.70 (mas fuerte, menos riguroso). |
| [`Benchmark_construction/`](Benchmark_construction/) | Scripts y datos para construir benchmarks R1 (MCQ) y R2 (similitud semantica). |
| [`images/`](images/) | Imagenes de prueba para test de ablacion en dominio visual (BLIP). |

### Scripts

#### Configuracion y datos

| Script | Descripcion |
|--------|-------------|
| [`experiment_config.py`](experiment_config.py) | Configuracion central: modelo, capas, conceptos, hiperparametros, umbrales de validacion |
| [`data_gen.py`](data_gen.py) | Genera datos de entrenamiento: nodos conceptuales, oraciones pos/neg, benchmarks R1/R2 |
| [`conceptnet_extractor.py`](conceptnet_extractor.py) | Extrae nodos semanticos de ConceptNet 5.7 |
| [`corpus_extractor.py`](corpus_extractor.py) | Extrae oraciones reales de corpora NLTK (Brown, Gutenberg) |

#### Extraccion y refinamiento de CAVs

| Script | Descripcion |
|--------|-------------|
| [`cav_extraction.py`](cav_extraction.py) | Extrae CAVs normalizados (Mean Diff + SVM) de las activaciones de GPT-2 |
| [`refine_cavs.py`](refine_cavs.py) | Purificacion ortogonal: elimina entanglement entre conceptos via proyeccion de subespacio |
| [`cav_statistical_tests.py`](cav_statistical_tests.py) | Validacion estadistica: TCAV permutation test, selectividad, bootstrap CI, Cohen's d |

#### Experimentos

| Script | Descripcion |
|--------|-------------|
| [`experiment_runner.py`](experiment_runner.py) | Experimento factorial original: sustraccion/proyeccion, single-layer, CAVs normalizados |
| [`experiment_multilayer.py`](experiment_multilayer.py) | **Experimento multi-capa**: steering con SVM sin normalizar, intervencion simultanea en multiples capas |

#### Tests cualitativos

| Script | Descripcion |
|--------|-------------|
| [`aphasia_test.py`](aphasia_test.py) | Generacion de texto con CAVs normalizados (sweep de todos los factores) |
| [`aphasia_test_steering.py`](aphasia_test_steering.py) | Generacion de texto con vectores de steering multi-capa |
| [`dialogue_test.py`](dialogue_test.py) | Simulador interactivo de dialogo con ablacion en tiempo real |
| [`vision_aphasia_test.py`](vision_aphasia_test.py) | Test de ablacion en dominio visual (modelo BLIP) |

#### Validacion

| Script | Descripcion |
|--------|-------------|
| [`validation_tests.py`](validation_tests.py) | Suite de 20 tests automatizados (T01-T20): datos, modelo, extraccion, estadistica |
| [`metrics.py`](metrics.py) | Metricas R1 (accuracy por hiperonimo) y R2 (similitud coseno) |

#### Analisis y visualizacion

| Script | Descripcion |
|--------|-------------|
| [`generate_figures.py`](generate_figures.py) | Figuras de publicacion F01-F08 (sin modelo, usa resultados pre-computados) |
| [`generate_figures_deep.py`](generate_figures_deep.py) | Figuras de activacion profunda F10-F14 (requiere modelo) |
| [`analyze_aphasia_results.py`](analyze_aphasia_results.py) | Analisis de resultados del test de afasia |
| [`check_baselines.py`](check_baselines.py) | Verificacion de problemas en baselines |

## Pipeline de Ejecucion

```bash
# 1. Generar datos de entrenamiento y benchmarks
python data_gen.py

# 2. Extraer CAVs normalizados (capas 3-10)
python cav_extraction.py

# 3. Validacion estadistica de CAVs
python cav_statistical_tests.py

# 4. Purificacion ortogonal
python refine_cavs.py

# 5. Experimento factorial original (single-layer)
python experiment_runner.py

# 6. Experimento multi-capa con steering (mejor enfoque)
python experiment_multilayer.py

# 7. Tests cualitativos
python aphasia_test.py
python aphasia_test_steering.py

# 8. Suite de validacion completa
python validation_tests.py

# 9. Generar figuras
python generate_figures.py
python generate_figures_deep.py
```

## Conceptos Analizados

Tres conceptos a nivel superordinado (Rosch, 1978), con nodos semanticos de ConceptNet:

| Concepto | Nodos | Ejemplos | Oraciones (pos/neg) |
|----------|-------|----------|---------------------|
| **Time** | 38 | hour, minute, day, week, month, year, decade | 423 / 423 |
| **Place** | 38 | city, town, village, country, mountain, river | 770 / 770 |
| **Tools** | 34 | hammer, screwdriver, wrench, pliers, saw, drill | 100 / 100 |

## Metodos

### Extraccion de CAVs

Los CAVs se extraen del residual stream de GPT-2-small (`blocks.{L}.hook_resid_post`, d=768), usando la media de activaciones sobre todos los tokens (mean pooling).

- **Mean Difference**: `CAV = normalize(mean(pos) - mean(neg))`
- **SVM**: `CAV = normalize(clf.coef_[0])` — vector normal al hiperplano de LinearSVC (Kim et al. 2018)

Ambos metodos producen vectores de norma unitaria. SVM alcanza 94-99% de precision en cross-validation 5-fold.

### Purificacion ortogonal

Para eliminar entanglement entre conceptos (correlacion time-place ~0.30-0.47):

```
v_puro = v_A - V (V^T V)^{-1} V^T v_A
```

donde `V = [v_B | v_C]` son los CAVs de los otros conceptos. Ortogonalidad verificada a precision de maquina (~1e-7). Detalle completo en [`cavs/README.md`](cavs/).

### Ablacion multi-capa (Feature Steering)

El experimento clave usa vectores SVM **sin normalizar** con intervencion simultanea en multiples capas:

```
act' = act - alpha * steering_vector
```

Tres correcciones respecto al enfoque original:
1. **Sin normalizar**: conserva la escala natural del vector (norma ~1-4 vs activaciones ~50-400)
2. **Multi-capa**: interviene en las 12 capas simultaneamente (evita bypass via residual stream)
3. **Mean-centering** (opcional): centra activaciones antes de entrenar el SVM (Zou et al. 2023)

## Resultados Principales

| Experimento | Mejor ΔR1 | Factor clave | Validacion |
|-------------|-----------|--------------|------------|
| Factorial original (single-layer, normalizado) | +0.07 | alpha | CV 94-99%, Cohen's d |
| **Multi-capa SVM (sin normalizar)** | **+0.333** | modo multi-capa | CV 80-91%, Cohen's d 3.1-4.9 |
| Multi-capa mean-diff (comparacion) | +0.70 | modo multi-capa | Sin CV |

El enfoque multi-capa con SVM produce ablacion 5x mas fuerte que el original, con validacion estadistica completa. Los controles de amplificacion (sumar en vez de restar el vector) confirman la direccion correcta: R1 sube para los 3 conceptos.

## Documentacion Adicional

| Archivo | Descripcion |
|---------|-------------|
| [`avances.md`](avances.md) | Informe detallado de avances, fundamentacion teorica y decisiones de diseno |
| [`tesis.md`](tesis.md) | Documento de tesis |
| [`cav_extraction_process.mmd`](cav_extraction_process.mmd) | Diagrama Mermaid del proceso de extraccion |
| [`validation_report.json`](validation_report.json) | Resultados de la suite de validacion (T01-T20) |
| [`corpus_training_report.json`](corpus_training_report.json) | Reporte de extraccion de corpus |

## Referencias

- Kim, B. et al. (2018). *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)*. ICML.
- Turner, A. et al. (2023). *Activation Addition: Steering Language Models Without Optimization*. arXiv:2308.10248.
- Zou, A. et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv:2310.01405.
- Belrose, N. et al. (2023). *LEACE: Perfect Linear Concept Erasure in Closed Form*. NeurIPS.
- Ravfogel, S. et al. (2020). *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection*. ACL.
- Rosch, E. (1978). *Principles of Categorization*. In Cognition and Categorization.
- Speer, R. et al. (2017). *ConceptNet 5.5: An Open Multilingual Graph of General Knowledge*. AAAI.
