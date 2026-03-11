# Afasia Semantica Artificial en GPT-2 Small

Simulacion experimental de afasia semantica categoria-especifica en GPT-2-small (125M parametros) mediante ablacion causal con Concept Activation Vectors (CAVs). Se extraen representaciones lineales de 30 conceptos concretos del residual stream del modelo, se validan estadisticamente, y se usan para remover selectivamente conocimiento conceptual — analogamente a como lesiones cerebrales focales producen deficits semanticos selectivos en pacientes humanos (Warrington & Shallice, 1984; Caramazza & Shelton, 1998).

La evaluacion principal se realiza mediante el **CCT (Camel and Cactus Test)**, adaptacion computacional de la prueba neuropsicologica *Camel and Cactus Test* (Bozeat et al., 2000), complementada con tres benchmarks **BEA** (association, naming, odd-one-out) y un test narrativo de **Caperucita Roja** para ablacion del concepto "wolf".

## Hallazgos Principales

### 1. La ablacion funciona y es controlable

La ablacion via CAV produce una **caida media de 40.5% en accuracy** sobre items que el modelo responde correctamente sin intervencion. La relacion dosis-respuesta es monotonicamente creciente en el **81.5%** de las condiciones (96.8% permitiendo una violacion), confirmando que el parametro alpha controla la intensidad de ablacion de forma principiada (Turner et al., 2023).

| Alpha | Delta on-target | Delta off-target |
|-------|----------------|-----------------|
| 3.0   | 0.225          | 0.228           |
| 6.0   | 0.358          | 0.330           |
| 10.0  | 0.451          | 0.404           |
| 15.0  | 0.488          | 0.437           |
| 20.0  | 0.506          | 0.479           |

### 2. SVM es drasticamente superior a Mean Difference

El metodo de extraccion del CAV determina la especificidad de la ablacion.

| Metodo     | Delta on-target | Delta off-target | Ratio especificidad |
|------------|----------------|-----------------|---------------------|
| **SVM**    | **0.428**       | **0.199**        | **2.15**            |
| Mean Diff  | 0.383           | 0.552            | 0.69 (anti-especifico) |

Los CAVs SVM (Kim et al., 2018) encuentran el hiperplano de maxima separacion, produciendo direcciones mas limpias. Mean Difference captura varianza compartida, resultando en mas dano colateral que dano al concepto objetivo (ratio < 1.0).

### 3. La capa 6 es optima para ablacion especifica

| Capa       | Delta on-target | Delta off-target | Ratio |
|------------|----------------|-----------------|-------|
| **L6**     | **0.461**       | **0.325**        | **1.42** |
| L9         | 0.288           | 0.309            | 0.93  |
| L10        | 0.260           | 0.294            | 0.88  |
| All layers | 0.613           | 0.575            | 1.07  |

La capa 6 (media del modelo) codifica representaciones mas semanticas y menos sintacticas (Tenney et al., 2019; Jawahar et al., 2019). Las capas tardias son anti-especificas, consistente con su rol en procesamiento sintactico-posicional.

### 4. Especificidad por concepto

12 de 27 conceptos (44%) muestran especificidad positiva (ratio >= 1.2):

| Concepto   | Delta on-target | Delta off-target | Ratio |
|------------|----------------|-----------------|-------|
| Musician   | 0.769           | 0.321            | **2.40** |
| Tailor     | 0.767           | 0.347            | **2.21** |
| Doctor     | 0.610           | 0.370            | 1.65  |
| Writer     | 0.545           | 0.329            | 1.66  |
| Candle     | 0.538           | 0.328            | 1.64  |
| Painter    | 0.511           | 0.351            | 1.46  |
| Snowman    | 0.515           | 0.352            | 1.46  |
| Chef       | 0.480           | 0.366            | 1.31  |
| King       | 0.475           | 0.374            | 1.27  |
| Barber     | 0.432           | 0.340            | 1.27  |
| Pirate     | 0.411           | 0.342            | 1.20  |

### 5. Configuracion optima

**SVM + L6 + alpha entre 6 y 10** ofrece el mejor balance entre potencia y especificidad.

### 6. Resultados BEA (ANOVA)

Los tres benchmarks BEA confirman los hallazgos del CCT:

- **Metodo (SVM vs Mean Diff)**: efecto significativo en las 3 tareas (p<0.001)
- **Capa**: efecto significativo en naming y odd-one-out
- **Alpha**: efecto significativo en todas las tareas

### 7. Caperucita Test (Wolf Ablation)

Test narrativo con 320 condiciones de ablacion del concepto "wolf" en la historia de Caperucita Roja. Mide wolf words, ICU scores, circumlocution y text length ratio. Demuestra que la ablacion conceptual afecta la generacion narrativa de forma categoria-especifica.

### Limitaciones

La especificidad global es modesta (ratio 1.08). En un modelo de 125M parametros con d=768, los conceptos comparten espacio representacional significativamente (Elhage et al., 2022). Esto motiva la extension a modelos mas grandes donde la mayor dimensionalidad podria permitir representaciones mas separadas.

## Diseno Experimental

### CCT (Camel and Cactus Test)

Adaptacion computacional de la prueba neuropsicologica (Bozeat et al., 2000). El modelo recibe `"A {concept} is related to"` con 4 candidatos y se selecciona el de mayor log-probabilidad media. Solo se evaluan items correctos en baseline (estrategia baseline-correct-only).

**Diseno factorial:**
- 27 conceptos evaluados (3 filtrados por <5 items correctos)
- 2 metodos CAV: Mean Difference, SVM
- 4 condiciones de capa: L6, L9, L10, todas (0-11)
- 5 intensidades alpha: 3.0, 6.0, 10.0, 15.0, 20.0
- Por condicion: 1 evaluacion on-target + 3 off-target muestreados
- **Total: 4,320 evaluaciones**

### Benchmarks BEA

- **Association**: MCQ de asociacion semantica (ConceptNet + WordNet + nodos curados)
- **Naming**: Confrontation naming con distractores intra-categoria
- **Odd-One-Out**: Deteccion de intruso semantico (3 items del concepto + 1 distractor)

### Caperucita Test

- 8 prompts (6 wolf-dependent + 2 control)
- 2 estilos: instruction (paradigma clinico) + completion (contexto narrativo)
- Wolf CAVs extraidos en las 12 capas del modelo
- 320 condiciones de ablacion

## Conceptos

30 conceptos concretos a nivel basico (Rosch, 1978), mas wolf para el test narrativo:

| Categoria      | Conceptos |
|----------------|-----------|
| Animales       | camel, squirrel, bee, horse, penguin, bird, spider |
| Objetos        | lock, toothbrush, violin, candle, fire, snowman |
| Profesiones    | fisherman, painter, knight, baker, doctor, barber, photographer, tailor, gardener, pirate, soldier, chef, musician, carpenter, writer, dentist |
| Personajes     | king |
| Narrativo      | wolf (exclusivo para Caperucita Test) |

## Metodos

### Extraccion de CAVs

Del residual stream de GPT-2-small (`blocks.{L}.hook_resid_post`, d=768), usando mean pooling:

- **Mean Difference**: `CAV = normalize(mean(pos) - mean(neg))`
- **SVM (Linear SVC)**: `CAV = normalize(clf.coef_[0])` con cross-validation 5-fold y grid search sobre C

### Ablacion por proyeccion

```
act' = act - alpha * proj(act, CAV)
```

donde `proj(act, CAV) = (act . CAV) * CAV`.

### Validacion estadistica

- **TCAV permutation test** (Kim et al., 2018): 30 permutaciones, p < 0.05
- **Selectividad** (Hewitt & Liang, 2019): control task
- **Bootstrap CI** (Efron & Tibshirani, 1993): 1000 resamples, IC 95%
- **Cohen's d**: tamano de efecto para separacion de clases

## Estructura del Proyecto

```
GPT2_small/
├── concepts/                      # Definiciones JSON por concepto (30 + wolf)
├── cavs/                          # CAVs normalizados (Mean Diff + SVM, 30 conceptos)
├── cavs_steering/                 # Vectores de steering SVM sin normalizar (multi-capa)
├── cavs_steering_meandiff/        # Vectores de steering mean-diff (comparacion)
├── results/                       # Resultados principales
│   ├── cct_test_results.json      # CCT: 4,320 evaluaciones
│   ├── bea_naming_test_results.json
│   ├── bea_oddoneout_test_results.json
│   ├── bea_synonym_test_results.json
│   ├── bea_anova_results.json     # ANOVA para los 3 tests BEA
│   ├── caperucita_test_results.json
│   ├── aphasia_test_results.json
│   ├── results_factorial.csv
│   ├── validation_report.json     # Suite T01-T20
│   └── figures/                   # Figuras de publicacion (14 PDFs + BEA plots)
├── results_multilayer/            # Resultados multi-capa SVM
├── results_multilayer_meandiff/   # Resultados multi-capa mean-diff (comparacion)
├── results_sweep/                 # Sweep comprehensivo (diseno, pendiente de ejecucion)
├── Benchmark_construction/        # Scripts y output de benchmarks (R1, R2, BEA)
├── test/                          # Tests experimentales
│   ├── cct_test.py                # CCT: test principal
│   ├── bea_naming_test.py         # BEA naming con ablacion
│   ├── bea_oddoneout_test.py      # BEA odd-one-out con ablacion
│   ├── bea_synonym_test.py        # BEA synonym con ablacion
│   ├── caperucita_test.py         # Narrativa Caperucita Roja
│   ├── aphasia_test.py            # Generacion de texto con ablacion
│   ├── aphasia_test_steering.py   # Steering multi-capa
│   ├── dialogue_test.py           # Dialogo interactivo
│   └── quick_baseline_test.py     # Validacion rapida de baselines
├── docs/                          # Documentacion y diagramas
├── experiment_config.py           # Configuracion central (100+ parametros)
├── data_gen.py                    # Generacion de datos de entrenamiento
├── gen_wolf_data.py               # Generacion de datos para concepto wolf
├── extract_wolf_cavs.py           # Extraccion de CAVs para wolf (12 capas)
├── cav_extraction.py              # Extraccion de CAVs (30 conceptos)
├── cav_statistical_tests.py       # Validacion estadistica de CAVs
├── experiment_runner.py           # Experimento factorial original
├── experiment_multilayer.py       # Experimento multi-capa
├── experiment_sweep.py            # Sweep comprehensivo
├── metrics.py                     # Metricas: R1, R2, BEA (association, naming, oddoneout)
├── anova_analysis.py              # ANOVA, effect sizes, post-hoc (CCT)
├── bea_anova_analysis.py          # ANOVA para tests BEA
├── generate_figures.py            # Figuras F01-F08
├── generate_figures_deep.py       # Figuras F10-F14
├── validation_tests.py            # Suite de 20 tests automatizados
├── conceptnet_extractor.py        # Extraccion de ConceptNet
├── corpus_extractor.py            # Extraccion de corpora (Brown, Gutenberg)
├── concept_registry.py            # Carga de definiciones de conceptos
├── Dockerfile                     # Entorno reproducible (CUDA 12.1 + PyTorch 2.2)
├── run_pipeline.sh                # Pipeline completo automatizado
├── run_cct.sh                     # Runner para CCT test
└── run_caperucita.sh              # Runner para Caperucita test
```

## Pipeline de Ejecucion

```bash
# Entorno Docker (recomendado)
docker build -t tesis-gpt2 .
docker run --gpus all -it tesis-gpt2 bash

# Pipeline completo
bash run_pipeline.sh

# O paso a paso:
python data_gen.py                            # 1. Generar datos (30 conceptos)
python validation_tests.py --no-model         # 2. Validar datos
python cav_extraction.py                      # 3. Extraer CAVs
python cav_statistical_tests.py --from-saved  # 4. Validacion estadistica
python experiment_runner.py                   # 5. Experimento factorial
python experiment_multilayer.py               # 6. Experimento multi-capa
python validation_tests.py                    # 7. Validacion completa

# Wolf (para Caperucita Test)
python gen_wolf_data.py                       # 8. Generar datos wolf
python extract_wolf_cavs.py                   # 9. Extraer CAVs wolf (12 capas)

# Tests
python -m Benchmark_construction.build_bea_association
python -m Benchmark_construction.build_bea_naming
python -m Benchmark_construction.build_bea_oddoneout
python -m test.cct_test                       # CCT (4,320 evaluaciones)
python -m test.bea_naming_test                # BEA naming
python -m test.bea_oddoneout_test             # BEA odd-one-out
python -m test.bea_synonym_test               # BEA synonym
python -m test.caperucita_test                # Caperucita (320 condiciones)

# Analisis
python anova_analysis.py                      # ANOVA CCT
python bea_anova_analysis.py                  # ANOVA BEA
python generate_figures.py                    # Figuras F01-F08
python generate_figures_deep.py               # Figuras F10-F14
```

## Requisitos

- NVIDIA GPU con CUDA 12.1+ (GPT-2 small usa ~0.5GB VRAM)
- ~8GB RAM de sistema
- Docker (recomendado) o Python 3.10+ con dependencias en Dockerfile

## Resultados Cuantitativos

### Baselines del modelo (sin ablacion)

Accuracy en benchmark BEA de asociacion semantica (chance = 0.25):

| Concepto    | Accuracy | Items | Correctos |
|-------------|----------|-------|-----------|
| Bee         | 0.680    | 25    | 17        |
| Horse       | 0.640    | 25    | 16        |
| Penguin     | 0.625    | 24    | 15        |
| Violin      | 0.625    | 24    | 15        |
| Camel       | 0.600    | 25    | 15        |
| Doctor      | 0.600    | 25    | 15        |
| Spider      | 0.600    | 25    | 15        |
| Squirrel    | 0.560    | 25    | 14        |
| Baker       | 0.480    | 25    | 12        |
| Bird        | 0.480    | 25    | 12        |
| Fire        | 0.480    | 25    | 12        |
| Lock        | 0.440    | 25    | 11        |
| Pirate      | 0.440    | 25    | 11        |
| Candle      | 0.417    | 24    | 10        |
| Writer      | 0.417    | 24    | 10        |
| Toothbrush  | 0.400    | 25    | 10        |
| Knight      | 0.375    | 24    | 9         |
| Painter     | 0.360    | 25    | 9         |
| Dentist     | 0.348    | 23    | 8         |
| Carpenter   | 0.333    | 24    | 8         |
| Musician    | 0.320    | 25    | 8         |
| Barber      | 0.280    | 25    | 7         |
| King        | 0.280    | 25    | 7         |
| Tailor      | 0.250    | 24    | 6         |
| Chef        | 0.238    | 21    | 5         |
| Gardener    | 0.208    | 24    | 5         |
| Snowman     | 0.227    | 22    | 5         |

### Monotonia dosis-respuesta

| Criterio | Condiciones | Porcentaje |
|----------|-------------|------------|
| Estrictamente monotonico | 176 / 216 | 81.5% |
| Con 1 o menos violaciones | 209 / 216 | 96.8% |

### Conceptos filtrados

3 conceptos excluidos por <5 items correctos en baseline: fisherman (0.167), photographer (0.174), soldier (0.174).

### Validacion (Suite T01-T20)

17 PASS, 2 SKIP, 1 FAIL (T10: tools R1 reliability = 0.133, no critico).

## Referencias

- Bozeat, S., et al. (2000). *Non-verbal semantic impairment in semantic dementia*. Neuropsychologia, 38(9), 1207-1215.
- Caramazza, A., & Shelton, J. R. (1998). *Domain-specific knowledge systems in the brain*. Journal of Cognitive Neuroscience, 10(1), 1-34.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
- Elhage, N., et al. (2022). *Toy Models of Superposition*. Anthropic. arXiv:2209.10652.
- Hewitt, J., & Liang, P. (2019). *Designing and Interpreting Probes with Control Tasks*. EMNLP.
- Jawahar, G., et al. (2019). *What Does BERT Look At?* BlackboxNLP Workshop, ACL.
- Kim, B., et al. (2018). *Interpretability Beyond Feature Attribution: TCAV*. ICML.
- Lambon Ralph, M. A., et al. (2017). *The neural and computational bases of semantic cognition*. Nature Reviews Neuroscience, 18(1), 42-55.
- Miller, G. A. (1995). *WordNet: A Lexical Database for English*. Communications of the ACM, 38(11), 39-41.
- Nanda, N., et al. (2023). *Emergent Linear Representations in World Models*. BlackboxNLP Workshop, ACL.
- Park, K., et al. (2023). *The Linear Representation Hypothesis*. arXiv:2311.03658.
- Patterson, K., et al. (2007). *Where do you know what you know?* Nature Reviews Neuroscience, 8(12), 976-987.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT*. EMNLP.
- Rosch, E. (1978). *Principles of Categorization*. Lawrence Erlbaum.
- Speer, R., et al. (2017). *ConceptNet 5.5*. AAAI.
- Tenney, I., et al. (2019). *BERT Rediscovers the Classical NLP Pipeline*. ACL.
- Turner, A., et al. (2023). *Activation Addition: Steering Language Models Without Optimization*. arXiv:2308.10248.
- Warrington, E. K., & Shallice, T. (1984). *Category specific semantic impairments*. Brain, 107(3), 829-854.
- Zou, A., et al. (2023). *Representation Engineering*. arXiv:2310.01405.
