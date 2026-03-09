# Afasia Semantica Artificial en GPT-2 Small

Simulacion experimental de afasia semantica categoria-especifica en GPT-2-small (125M parametros) mediante ablacion causal con Concept Activation Vectors (CAVs). El proyecto extrae representaciones lineales de 30 conceptos concretos del espacio de activaciones del modelo, las valida estadisticamente, y las usa para remover selectivamente conocimiento conceptual — analogamente a como lesiones cerebrales focales producen deficits semanticos selectivos en pacientes humanos (Warrington & Shallice, 1984; Caramazza & Shelton, 1998).

La evaluacion se realiza mediante el **CCT (Camel and Cactus Test)**, adaptacion computacional de la prueba neuropsicologica *Camel and Cactus Test* (Bozeat et al., 2000), que mide asociacion semantica con y sin ablacion conceptual.

## Hallazgos Principales

### 1. La ablacion funciona y es controlable

La ablacion via CAV produce una **caida media de 40.5% en accuracy** sobre items que el modelo responde correctamente sin intervencion. La relacion dosis-respuesta es monotonicamente creciente en el **81.5%** de las condiciones (96.8% permitiendo una violacion), confirmando que el parametro α controla la intensidad de ablacion de forma principiada (Turner et al., 2023).

| Alpha (α) | Delta on-target | Delta off-target |
|-----------|----------------|-----------------|
| 3.0       | 0.225          | 0.228           |
| 6.0       | 0.358          | 0.330           |
| 10.0      | 0.451          | 0.404           |
| 15.0      | 0.488          | 0.437           |
| 20.0      | 0.506          | 0.479           |

### 2. SVM es drasticamente superior a Mean Difference

El hallazgo mas importante del experimento: el metodo de extraccion del CAV determina la especificidad de la ablacion.

| Metodo     | Delta on-target | Delta off-target | Ratio especificidad |
|------------|----------------|-----------------|---------------------|
| **SVM**    | **0.428**       | **0.199**        | **2.15**            |
| Mean Diff  | 0.383           | 0.552            | 0.69 (anti-especifico) |

Los CAVs extraidos con SVM (Kim et al., 2018) encuentran el hiperplano de maxima separacion en el espacio de activaciones, produciendo direcciones mas limpias. Mean Difference simplemente resta promedios de activaciones, capturando varianza compartida entre conceptos — esto resulta en **mas dano colateral que dano al concepto objetivo** (ratio < 1.0).

### 3. La capa 6 es optima para ablacion especifica

| Capa       | Delta on-target | Delta off-target | Ratio |
|------------|----------------|-----------------|-------|
| **L6**     | **0.461**       | **0.325**        | **1.42** |
| L9         | 0.288           | 0.309            | 0.93  |
| L10        | 0.260           | 0.294            | 0.88  |
| All layers | 0.613           | 0.575            | 1.07  |

La capa 6 (media del modelo) codifica representaciones mas semanticas y menos sintacticas (Tenney et al., 2019; Jawahar et al., 2019), lo que explica su mejor balance entre potencia y especificidad. Las capas tardias (L9, L10) son anti-especificas, consistente con su rol en procesamiento sintactico-posicional. La condicion all_layers es la mas potente pero con baja especificidad — la intervencion masiva disrumpe el modelo de forma global.

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

Los conceptos con representaciones mas distintivas en el espacio de activaciones (menor superposicion con otros conceptos) muestran mayor especificidad, consistente con la hipotesis del hub semantico (Patterson et al., 2007; Lambon Ralph et al., 2017).

### 5. Configuracion optima

La combinacion **SVM + L6 + α ∈ [6, 10]** ofrece el mejor balance entre potencia de ablacion y especificidad. Esta configuracion:
- Produce caidas significativas en accuracy on-target
- Minimiza el dano colateral a conceptos no-objetivo
- Mantiene monotonia dosis-respuesta
- Es coherente con la teoria de representaciones lineales (Park et al., 2023; Nanda et al., 2023)

### 6. Ejemplos de ablacion completa (delta = 1.0)

> Datos completos en [`results/cct_test_results.json`](results/cct_test_results.json).
> Cada ejemplo se localiza filtrando por `ablated_concept`, `cav_method`, `layer`, `alpha` y `on_target: true`.

De 4,320 evaluaciones, **211 condiciones on-target alcanzaron delta >= 0.8** y 19 conceptos distintos lograron ablacion completa (delta = 1.0, accuracy = 0.0). A continuacion se muestran 10 casos representativos:

#### Ejemplo 1: TAILOR — SVM, L6, α=6.0
> `cct_test_results.json` → filtrar: `ablated_concept: "tailor"`, `cav_method: "svm"`, `layer: "L6"`, `alpha: 6.0`, `on_target: true`

**Capa unica, alpha bajo — la ablacion mas eficiente del experimento.** 6/6 items fallados.

| Pregunta | Correcta | Modelo dijo |
|----------|----------|------------|
| A tailor is related to ___ | fabric | **greenhouse** |
| A tailor is related to ___ | sew | **light** |
| A tailor is related to ___ | seam | **stick** |
| A tailor is related to ___ | linen | **novel** |
| A tailor is related to ___ | garmentmaker | **ceremony** |
| A tailor is related to ___ | garment-worker | **floss** |

#### Ejemplo 2: DOCTOR — SVM, all_layers, α=10.0
> `cct_test_results.json` → filtrar: `ablated_concept: "doctor"`, `cav_method: "svm"`, `layer: "all_layers"`, `alpha: 10.0`, `on_target: true`

**El caso mas estadisticamente convincente.** 15/15 items fallados — set grande, ablacion total.

| Pregunta | Correcta | Modelo dijo |
|----------|----------|------------|
| A doctor is related to ___ | hospital | **rival** |
| A doctor is related to ___ | stethoscope | **hammer** |
| A doctor is related to ___ | medicine | **river** |
| A doctor is related to ___ | prescription | **fly** |
| A doctor is related to ___ | clinic | **spark** |
| A doctor is related to ___ | nurse | **bill** |
| A doctor is related to ___ | treatment | **hair** |
| A doctor is related to ___ | cure | **manuscript** |
| A doctor is related to ___ | ward | **hatch** |
| A doctor is related to ___ | checkup | **editor** |
| A doctor is related to ___ | illness | **burning** |
| A doctor is related to ___ | doctor-patient relation | **jockey** |
| A doctor is related to ___ | medical practitioner | **sword** |
| A doctor is related to ___ | medical man | **horse** |
| A doctor is related to ___ | theologian | **frame** |

#### Ejemplo 3: HORSE — SVM, all_layers, α=10.0
> `cct_test_results.json` → filtrar: `ablated_concept: "horse"`, `cav_method: "svm"`, `layer: "all_layers"`, `alpha: 10.0`, `on_target: true`

**Set mas grande del experimento.** 16/16 items fallados.

| Pregunta | Correcta | Modelo dijo |
|----------|----------|------------|
| A horse is related to ___ | gallop | **art** |
| A horse is related to ___ | mane | **cast** |
| A horse is related to ___ | hoof | **ocean** |
| A horse is related to ___ | rider | **lamp** |
| A horse is related to ___ | bridle | **wax** |
| A horse is related to ___ | mare | **emperor** |
| A horse is related to ___ | stallion | **hook** |
| A horse is related to ___ | trot | **programme** |
| A horse is related to ___ | equestrian | **girl** |
| A horse is related to ___ | barn | **bake** |
| A horse is related to ___ | foal | **spin** |
| A horse is related to ___ | horseshoe | **glue** |
| A horse is related to ___ | jockey | **salon** |
| A horse is related to ___ | horse's foot | **scissors** |
| A horse is related to ___ | horsemeat | **nest** |
| A horse is related to ___ | horseflesh | **angle** |

#### Ejemplo 4: LOCK — mean_diff, L6, α=6.0
> `cct_test_results.json` → filtrar: `ablated_concept: "lock"`, `cav_method: "mean_diff"`, `layer: "L6"`, `alpha: 6.0`, `on_target: true`

**Incluso mean_diff logra ablacion total en L6 con alpha bajo.** 11/11 items fallados.

| Pregunta | Correcta | Modelo dijo |
|----------|----------|------------|
| A lock is related to ___ | key | **light** |
| A lock is related to ___ | door | **insect** |
| A lock is related to ___ | padlock | **floss** |
| A lock is related to ___ | safe | **colony** |
| A lock is related to ___ | security | **water** |
| A lock is related to ___ | deadbolt | **horseback** |
| A lock is related to ___ | keyhole | **root** |
| A lock is related to ___ | metal | **eight** |
| A lock is related to ___ | tumbler | **watercolor** |
| A lock is related to ___ | fastening | **apron** |
| A lock is related to ___ | holdfast | **gum** |

#### Ejemplo 5: CAMEL — SVM, all_layers, α=15.0
> `cct_test_results.json` → filtrar: `ablated_concept: "camel"`, `cav_method: "svm"`, `layer: "all_layers"`, `alpha: 15.0`, `on_target: true`

15/15 items fallados. Las respuestas correctas son empujadas 5-15 puntos log-prob por debajo.

| Pregunta | Correcta | Modelo dijo |
|----------|----------|------------|
| A camel is related to ___ | hump | **roll** |
| A camel is related to ___ | dromedary | **robe** |
| A camel is related to ___ | oasis | **exhibit** |
| A camel is related to ___ | nomad | **tree** |
| A camel is related to ___ | dune | **bridge** |
| A camel is related to ___ | cud | **shield** |
| A camel is related to ___ | saddle | **character** |
| A camel is related to ___ | arid | **crown** |
| A camel is related to ___ | sahara | **wood** |
| A camel is related to ___ | water | **call** |
| A camel is related to ___ | bactrian | **stick** |
| A camel is related to ___ | camelus | **focus** |
| A camel is related to ___ | genus camelus | **emperor** |
| A camel is related to ___ | even-toed ungulate | **light** |
| A camel is related to ___ | artiodactyl | **march** |

#### Ejemplo 6: MUSICIAN — SVM, L9, α=20.0
> `cct_test_results.json` → filtrar: `ablated_concept: "musician"`, `cav_method: "svm"`, `layer: "L9"`, `alpha: 20.0`, `on_target: true`

8/8 items fallados. Capa unica (L9).

| Pregunta | Correcta | Modelo dijo |
|----------|----------|------------|
| A musician is related to ___ | concert | **burning** |
| A musician is related to ___ | chord | **shield** |
| A musician is related to ___ | band | **girl** |
| A musician is related to ___ | song | **pot** |
| A musician is related to ___ | piano | **health** |
| A musician is related to ___ | guitar | **royal** |
| A musician is related to ___ | perform | **scissors** |
| A musician is related to ___ | musical group | **flash** |

#### Ejemplo 7: CANDLE — SVM, all_layers, α=10.0
> `cct_test_results.json` → filtrar: `ablated_concept: "candle"`, `cav_method: "svm"`, `layer: "all_layers"`, `alpha: 10.0`, `on_target: true`

10/10 items fallados. Concepto de tipo objeto (no profesion ni animal).

| Pregunta | Correcta | Modelo dijo |
|----------|----------|------------|
| A candle is related to ___ | wick | **symptom** |
| A candle is related to ___ | flame | **mechanism** |
| A candle is related to ___ | light | **diagnosis** |
| A candle is related to ___ | glow | **razor** |
| A candle is related to ___ | lantern | **patient** |
| A candle is related to ___ | candelabra | **gate** |
| A candle is related to ___ | scented | **theologist** |
| A candle is related to ___ | chandelier | **molar** |
| A candle is related to ___ | candlewick | **attack** |
| A candle is related to ___ | lamp | **pastry** |

#### Ejemplo 8: PIRATE — mean_diff, all_layers, α=15.0
> `cct_test_results.json` → filtrar: `ablated_concept: "pirate"`, `cav_method: "mean_diff"`, `layer: "all_layers"`, `alpha: 15.0`, `on_target: true`

11/11 items fallados. Diferencias de log-prob extremas (sail: -58.6 vs distractor: -23.9).

| Pregunta | Correcta | Modelo dijo |
|----------|----------|------------|
| A pirate is related to ___ | treasure | **microphone** |
| A pirate is related to ___ | sword | **feather** |
| A pirate is related to ___ | parrot | **landscape** |
| A pirate is related to ___ | gold | **aperture** |
| A pirate is related to ___ | buccaneer | **plumage** |
| A pirate is related to ___ | sail | **yell** |
| A pirate is related to ___ | eyepatch | **oven** |
| A pirate is related to ___ | thief | **bloom** |
| A pirate is related to ___ | stealer | **outcry** |
| A pirate is related to ___ | plunderer | **medicine** |
| A pirate is related to ___ | pillager | **bake** |

#### Ejemplo 9: SQUIRREL — SVM, all_layers, α=10.0
> `cct_test_results.json` → filtrar: `ablated_concept: "squirrel"`, `cav_method: "svm"`, `layer: "all_layers"`, `alpha: 10.0`, `on_target: true`

14/14 items fallados. Las respuestas se vuelven completamente aleatorias.

| Pregunta | Correcta | Modelo dijo |
|----------|----------|------------|
| A squirrel is related to ___ | acorn | **gallery** |
| A squirrel is related to ___ | tree | **digital** |
| A squirrel is related to ___ | bushy | **cud** |
| A squirrel is related to ___ | forest | **fixing** |
| A squirrel is related to ___ | chipmunk | **hygiene** |
| A squirrel is related to ___ | burrow | **rose** |
| A squirrel is related to ___ | rodent | **canvas** |
| A squirrel is related to ___ | oak | **filling** |
| A squirrel is related to ___ | scurry | **anesthesia** |
| A squirrel is related to ___ | gray | **medicine** |
| A squirrel is related to ___ | woodland | **decree** |
| A squirrel is related to ___ | sciuridae | **digital** |
| A squirrel is related to ___ | family sciuridae | **throne** |
| A squirrel is related to ___ | gnawer | **map** |

#### Ejemplo 10: WRITER — mean_diff, L6, α=6.0
> `cct_test_results.json` → filtrar: `ablated_concept: "writer"`, `cav_method: "mean_diff"`, `layer: "L6"`, `alpha: 6.0`, `on_target: true`

10/10 items fallados. Capa unica, alpha bajo.

| Pregunta | Correcta | Modelo dijo |
|----------|----------|------------|
| A writer is related to ___ | pen | **trial** |
| A writer is related to ___ | book | **worm** |
| A writer is related to ___ | editor | **yellow** |
| A writer is related to ___ | essay | **gray** |
| A writer is related to ___ | ink | **freeze** |
| A writer is related to ___ | typewriter | **sting** |
| A writer is related to ___ | author | **grill** |
| A writer is related to ___ | communicator | **hindquarters** |
| A writer is related to ___ | literate | **prayer** |
| A writer is related to ___ | literate person | **swarm** |

#### Resumen de ejemplos

| # | Concepto | Metodo | Capa | α | Items | Observacion |
|---|----------|--------|------|---|-------|-------------|
| 1 | tailor | SVM | L6 | 6.0 | 6 | Capa unica + alpha mas bajo → ablacion mas eficiente |
| 2 | doctor | SVM | all | 10.0 | 15 | Set grande, 15/15 items fallados |
| 3 | horse | SVM | all | 10.0 | 16 | Set mas grande (16 items), destruccion total |
| 4 | lock | mean_diff | L6 | 6.0 | 11 | mean_diff tambien logra ablacion total en L6 |
| 5 | camel | SVM | all | 15.0 | 15 | Animal, 15 items, scores empujados -15 puntos |
| 6 | musician | SVM | L9 | 20.0 | 8 | Capa unica L9 |
| 7 | candle | SVM | all | 10.0 | 10 | Objeto (no profesion ni animal) |
| 8 | pirate | mean_diff | all | 15.0 | 11 | Gaps de log-prob extremos (sail: -58.6) |
| 9 | squirrel | SVM | all | 10.0 | 14 | 14 items, respuestas totalmente aleatorias |
| 10 | writer | mean_diff | L6 | 6.0 | 10 | Profesion, capa unica, alpha bajo |

**Patron clave:** Tras la ablacion, las respuestas del modelo son **semanticamente aleatorias** (doctor→hospital se convierte en doctor→rival, horse→gallop en horse→art). El modelo pierde la asociacion semantica del concepto ablacionado, analogamente a un paciente con lesion focal en el hub semantico (Patterson et al., 2007).

### Limitaciones

La especificidad global es modesta (ratio 1.08). En un modelo de 125M parametros con dimensionalidad d=768, los conceptos comparten espacio representacional de forma significativa. Esto es esperable dado el principio de superposicion (Elhage et al., 2022): los modelos pequenos codifican mas features que dimensiones disponibles, creando interferencia entre representaciones conceptuales. Esta limitacion motiva la extension a modelos mas grandes (Pythia-2.8B, d=2560) donde la mayor dimensionalidad podria permitir representaciones mas separadas.

## Diseno Experimental

### CCT (Camel and Cactus Test)

Adaptacion computacional de la prueba neuropsicologica Camel and Cactus (Bozeat et al., 2000), usada clinicamente para evaluar deficits semanticos en pacientes con demencia semantica y afasia progresiva.

**Formato:** Para cada item, el modelo recibe un prompt `"A {concept} is related to"` con 4 candidatos (1 correcto + 3 distractores). Se selecciona el candidato cuya continuacion tiene mayor log-probabilidad media (scoring por probabilidad de oracion, no instrucciones MCQ).

**Estrategia baseline-correct-only:** Solo se evalua la ablacion sobre items que el modelo responde correctamente sin intervencion. Esto garantiza que el baseline es 1.0 por construccion — cualquier caida en accuracy es directamente atribuible a la ablacion, eliminando ruido de items que el modelo ya falla.

**Diseno factorial:**
- 27 conceptos evaluados (3 filtrados por <5 items correctos)
- 2 metodos CAV: Mean Difference, SVM (Kim et al., 2018)
- 4 condiciones de capa: L6, L9, L10, todas (0-11)
- 5 intensidades α: 3.0, 6.0, 10.0, 15.0, 20.0
- Por condicion: 1 evaluacion on-target + 3 off-target muestreados
- **Total: 4,320 evaluaciones**

### Benchmark BEA (Association)

Benchmark de asociacion semantica construido automaticamente usando tres fuentes:
1. **ConceptNet 5.5** (Speer et al., 2017) — via dataset HuggingFace `conceptnet5/conceptnet5`
2. **WordNet** (Miller, 1995) — relaciones taxonomicas (hypernyms, hyponyms, meronyms)
3. **Nodos curados** — asociaciones manuales por concepto

Los distractores se seleccionan de otros conceptos, filtrados para evitar solapamiento semantico con la respuesta correcta.

## Conceptos

30 conceptos concretos a nivel basico (Rosch, 1978), seleccionados para cubrir categorias semanticas diversas:

| Categoria      | Conceptos |
|----------------|-----------|
| Animales       | camel, squirrel, bee, horse, penguin, bird, spider |
| Objetos        | lock, toothbrush, violin, candle, fire, snowman |
| Profesiones    | fisherman, painter, knight, baker, doctor, barber, photographer, tailor, gardener, pirate, soldier, chef, musician, carpenter, writer, dentist |
| Personajes     | king |

Cada concepto tiene nodos semanticos extraidos de ConceptNet, oraciones de entrenamiento extraidas de corpora reales (Brown, Gutenberg) via NLTK, y controles psicolinguisticos (longitud de palabra, frecuencia Zipf).

## Metodos

### Extraccion de CAVs

Los CAVs se extraen del residual stream de GPT-2-small (`blocks.{L}.hook_resid_post`, d=768), usando mean pooling sobre todos los tokens (Reimers & Gurevych, 2019).

- **Mean Difference**: `CAV = normalize(mean(pos) - mean(neg))` — rapido pero captura varianza compartida
- **SVM (Linear SVC)**: `CAV = normalize(clf.coef_[0])` — encuentra el hiperplano de maxima separacion (Kim et al., 2018). Cross-validation 5-fold con grid search sobre C ∈ {0.01, 0.1, 1.0, 10.0}.

### Ablacion por proyeccion

```
act' = act - α · proj(act, CAV)
```

donde `proj(act, CAV) = (act · CAV) · CAV` es la proyeccion escalar del vector de activaciones sobre la direccion del CAV. El parametro α controla la intensidad de ablacion (Turner et al., 2023).

### Validacion estadistica

- **TCAV permutation test** (Kim et al., 2018): 30 permutaciones, p < 0.05
- **Selectividad** (Hewitt & Liang, 2019): control task para verificar que los CAVs capturan informacion semantica, no artefactos
- **Bootstrap CI** (Efron & Tibshirani, 1993): 1000 resamples, IC 95% para estabilidad del CAV
- **Cohen's d**: tamaño de efecto para separacion de clases

## Estructura del Proyecto

```
GPT2_small/
├── concepts/                      # Definiciones JSON por concepto (30 conceptos)
├── cavs/                          # CAVs normalizados (Mean Diff + SVM)
├── cavs_steering/                 # Vectores de steering SVM sin normalizar
├── cavs_steering_meandiff/        # Vectores de steering mean-diff
├── results/                       # Resultados: factorial, CCT, validacion
│   ├── cct_test_results.json      # ← Resultados principales del CCT
│   ├── results_factorial.csv      # Experimento factorial original
│   ├── results_specificity.csv    # Test de especificidad
│   ├── validation_report.json     # Suite de validacion T01-T20
│   └── figures/                   # Figuras de publicacion
├── results_multilayer/            # Resultados multi-capa SVM
├── results_multilayer_meandiff/   # Resultados multi-capa mean-diff
├── Benchmark_construction/        # Scripts de construccion de benchmarks
│   └── output/                    # Benchmarks generados (BEA association, naming, oddoneout)
├── test/                          # Tests experimentales
│   ├── cct_test.py                # ← CCT: test principal de asociacion con ablacion
│   ├── aphasia_test.py            # Generacion de texto con ablacion
│   ├── aphasia_test_steering.py   # Steering multi-capa
│   ├── caperucita_test.py         # Test narrativo (Caperucita Roja)
│   └── dialogue_test.py           # Simulador de dialogo interactivo
├── experiment_config.py           # Configuracion central
├── data_gen.py                    # Generacion de datos de entrenamiento
├── cav_extraction.py              # Extraccion de CAVs
├── cav_statistical_tests.py       # Validacion estadistica
├── experiment_runner.py           # Experimento factorial
├── experiment_multilayer.py       # Experimento multi-capa
├── metrics.py                     # Metricas: R1, R2, BEA association
├── validation_tests.py            # Suite de 20 tests automatizados
├── Dockerfile                     # Entorno reproducible (CUDA 12.1 + PyTorch 2.2)
└── run_pipeline.sh                # Pipeline completo automatizado
```

## Pipeline de Ejecucion

```bash
# Entorno Docker (recomendado)
docker build -t tesis-gpt2 .
docker run --gpus all -it tesis-gpt2 bash

# Pipeline completo (7 pasos)
bash run_pipeline.sh

# O paso a paso:
python data_gen.py                          # 1. Generar datos
python validation_tests.py --no-model       # 2. Validar datos
python cav_extraction.py                    # 3. Extraer CAVs
python cav_statistical_tests.py --from-saved # 4. Validacion estadistica
python experiment_runner.py                 # 5. Experimento factorial
python experiment_multilayer.py             # 6. Experimento multi-capa
python validation_tests.py                  # 7. Validacion completa

# CCT Test (requiere pipeline + benchmark BEA)
python -m Benchmark_construction.build_bea_association
python -m test.cct_test
```

## Requisitos

- NVIDIA GPU con CUDA 12.1+ (GPT-2 small usa ~0.5GB VRAM)
- ~8GB RAM de sistema
- Docker (recomendado) o Python 3.10+ con dependencias en Dockerfile

## Resultados Cuantitativos Completos

### Baselines del modelo (sin ablacion)

Accuracy del modelo en el benchmark BEA de asociacion semantica (chance = 0.25):

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

Promedio: ~0.43. Todos los conceptos superan el azar (0.25), confirmando que GPT-2 small posee conocimiento de asociacion semantica evaluable.

### Monotonia dosis-respuesta

| Criterio | Condiciones | Porcentaje |
|----------|-------------|------------|
| Estrictamente monotonico | 176 / 216 | 81.5% |
| Con ≤1 violacion | 209 / 216 | 96.8% |

### Conceptos filtrados

3 conceptos excluidos por tener <5 items correctos en baseline:
- **Fisherman** (4/24 correctos, 0.167)
- **Photographer** (4/23 correctos, 0.174)
- **Soldier** (4/23 correctos, 0.174)

## Referencias

- Bozeat, S., Lambon Ralph, M. A., Patterson, K., Garrard, P., & Hodges, J. R. (2000). *Non-verbal semantic impairment in semantic dementia*. Neuropsychologia, 38(9), 1207-1215.
- Caramazza, A., & Shelton, J. R. (1998). *Domain-specific knowledge systems in the brain: The animate-inanimate distinction*. Journal of Cognitive Neuroscience, 10(1), 1-34.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
- Elhage, N., et al. (2022). *Toy Models of Superposition*. Anthropic. arXiv:2209.10652.
- Hewitt, J., & Liang, P. (2019). *Designing and Interpreting Probes with Control Tasks*. EMNLP.
- Jawahar, G., Sagot, B., & Seddah, D. (2019). *What Does BERT Look At? An Analysis of BERT's Attention*. BlackboxNLP Workshop, ACL.
- Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., & Sayres, R. (2018). *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)*. ICML.
- Lambon Ralph, M. A., Jefferies, E., Patterson, K., & Rogers, T. T. (2017). *The neural and computational bases of semantic cognition*. Nature Reviews Neuroscience, 18(1), 42-55.
- Miller, G. A. (1995). *WordNet: A Lexical Database for English*. Communications of the ACM, 38(11), 39-41.
- Nanda, N., Lee, A., & Berber, M. (2023). *Emergent Linear Representations in World Models of Self-Supervised Sequence Models*. BlackboxNLP Workshop, ACL.
- Park, K., Choe, Y. J., & Veitch, V. (2023). *The Linear Representation Hypothesis and the Geometry of Large Language Models*. arXiv:2311.03658.
- Patterson, K., Nestor, P. J., & Rogers, T. T. (2007). *Where do you know what you know? The representation of semantic knowledge in the human brain*. Nature Reviews Neuroscience, 8(12), 976-987.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
- Rosch, E. (1978). *Principles of Categorization*. In E. Rosch & B. B. Lloyd (Eds.), Cognition and Categorization. Lawrence Erlbaum.
- Speer, R., Chin, J., & Havasi, C. (2017). *ConceptNet 5.5: An Open Multilingual Graph of General Knowledge*. AAAI.
- Tenney, I., Das, D., & Pavlick, E. (2019). *BERT Rediscovers the Classical NLP Pipeline*. ACL.
- Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023). *Activation Addition: Steering Language Models Without Optimization*. arXiv:2308.10248.
- Warrington, E. K., & Shallice, T. (1984). *Category specific semantic impairments*. Brain, 107(3), 829-854.
- Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A., Goel, S., Li, N., Lin, Z., Forsyth, M., Prabhu, R., Khoja, T., Fang, Z., & Hendrycks, D. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv:2310.01405.
