# Mapeo de circuitos conceptuales: Análisis de la representación en modelos Transformer mediante intervención causal simulando afasia semántica

**Trabajo de grado de Ciencia de Datos**

**Autor:** Juan José Lizcano Barbosa
**Tutor:** Jorge Andrés Alvarado Valencia
**Institución:** Pontificia Universidad Javeriana — Facultad de Ciencias
**Bogotá, D.C. — 2026**

---

## Resumen

Este proyecto desarrolla y aplica métodos de interpretabilidad mecanicista para identificar y caracterizar las estructuras internas que codifican conceptos lingüísticos en modelos de lenguaje basados en arquitectura Transformer. El procedimiento extrae **Vectores de Activación de Conceptos (CAV)** del residual stream del modelo, los valida estadísticamente, y los emplea como dirección de **ablación causal por proyección ortogonal** para inducir un déficit funcional análogo a la *afasia semántica* clínica. La evaluación cuantitativa se realiza mediante el **Camel and Cactus Test (CCT)**, prueba neuropsicológica de asociación semántica adaptada computacionalmente para tres modelos de distinta escala y modalidad: **GPT-2 small (124M)**, **Pythia-2.8B** y **LLaVA-1.5-7B**.

El objetivo es responder, mediante un Diseño de Experimentos (DoE) factorial, en qué medida la representación de un concepto admite una localización lineal de capa única, qué método de extracción del CAV captura mejor la dirección causal del concepto, cómo se comporta la curva dosis-respuesta de la intervención, cómo escala el efecto entre arquitecturas, y si las categorías conceptuales con mayor anclaje referencial concreto exhiben representaciones más nítidamente lineales.

---

## 1. Marco teórico

### 1.1. Interpretabilidad mecanicista

La interpretabilidad mecanicista es una rama de la inteligencia artificial explicable (XAI) que busca comprender el funcionamiento interno de un modelo observando sus piezas y sus conexiones, no solo sus salidas. Se concibe como un proceso de ingeniería inversa aplicado a redes neuronales: identificar los elementos básicos que cumplen funciones específicas y describir cómo se relacionan entre sí para generar el comportamiento final del modelo (He et al., 2024; Nanda, 2022).

### 1.2. Arquitectura Transformer

La arquitectura Transformer (Vaswani et al., 2017) reemplaza redes recurrentes y convolucionales mediante el mecanismo de **autoatención**, que permite capturar dependencias entre puntos arbitrarios de una secuencia en paralelo. El cálculo central de cada cabeza de atención es:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V$$

Sobre esta base, Anthropic Research (2021) propone un marco matemático que descompone un Transformer de dos capas como composición de operadores tensoriales y sumas residuales:

- el embedding $I_d \otimes W_E$ que lleva tokens al espacio residual,
- los bloques de atención implementados como sumas residuales de matrices de atención $A_h$ multiplicadas por proyecciones $W_{ov}^h$,
- el unembedding $I_d \otimes W_U$ que mapea el residual a la salida.

Esta representación algebraica es la que habilita el análisis del modelo como composición funcional y, por tanto, la posibilidad de intervenir circuitos específicos.

### 1.3. Vectores de Activación de Conceptos (CAV) y ablación causal

Un **CAV** (Kim et al., 2018) es una dirección lineal en el espacio de activaciones de una capa que codifica un concepto. Se obtiene contrastando activaciones positivas (estímulos donde el concepto está presente) con activaciones negativas (estímulos donde no), y normalizando el vector resultante. La proyección de una activación sobre el CAV cuantifica la presencia del concepto en esa activación, y la **resta** de esa proyección permite remover causalmente la información conceptual:

$$a' = a - \alpha \cdot \mathrm{proj}(a, \mathrm{CAV}) = a - \alpha \cdot (a \cdot \mathrm{CAV}) \cdot \mathrm{CAV}$$

donde $\alpha$ controla la intensidad de la intervención.

### 1.4. Afasia semántica como análogo biológico

La **afasia semántica** es un trastorno del lenguaje caracterizado por la pérdida de comprensión de conceptos y sus relaciones lógicas: anomia, pobreza léxica, parafasias semánticas, estereotipias (Reilly et al., 2024; Rubio Díaz, 2015; Toledo Rodríguez, 2021). En este trabajo, la ablación de un CAV se interpreta como una **simulación funcional** de afasia categoría-específica: al bloquear la dirección residual asociada al concepto "camello", el modelo debería exhibir, en condiciones controladas, fallas en tareas que requieren acceso a esa representación semántica, de forma análoga a cómo un paciente con lesión focal pierde acceso a categorías específicas (Castellana, 2011; Ramos-Galarza, 2023).

### 1.5. Camel and Cactus Test (CCT)

El **Camel and Cactus Test** (Bozeat et al., 2000) es un instrumento neuropsicológico diseñado para evaluar la integridad de la memoria semántica en pacientes con demencia semántica y afasia progresiva. Presenta un ítem objetivo y cuatro opciones, y solicita la opción semánticamente más relacionada. Su validez clínica descansa en que el rendimiento depende de representaciones conceptuales taxonómicas, funcionales y de co-ocurrencia, no de percepción inmediata ni memoria episódica. Por esa razón es el instrumento adoptado en este trabajo: convertirlo en un protocolo computacional permite medir, con la misma lógica que se aplica al cerebro humano, la integridad del sistema semántico de un Transformer antes y después de la intervención.

---

## 2. Diseño experimental

El estudio adopta un Diseño de Experimentos (DoE) factorial con **cinco factores** de intervención. Se aplican rigurosamente los principios de aleatorización, replicación y bloques. La variable de respuesta es la caída de precisión $\Delta\mathrm{acc}$ (R1) inducida por la ablación.

### Factor A — Método de extracción del CAV

| Nivel | Descripción |
|---|---|
| Mean Difference | $\mathrm{CAV} = \mathrm{normalize}(\mu_{\text{pos}} - \mu_{\text{neg}})$. Asume traslación lineal entre distribuciones (Turner et al., 2023; Panickssery et al., 2024). |
| SVM lineal | $\mathrm{CAV} = \mathrm{normalize}(w_{\mathrm{SVM}})$. El concepto se captura mejor mediante la dirección normal al hiperplano de máxima separación entre clases (Kim et al., 2018). |

### Factor B — Capa de intervención

| Nivel | Profundidad | Justificación |
|---|---|---|
| Capa media | ~50% | Almacén principal de conocimiento factual y semántico (Geva et al., 2021; Meng et al., 2022). |
| Capa tardía temprana | ~75% | "Mover heads" que copian información ya procesada hacia la posición final del token (Meng et al., 2022). |
| Capa tardía profunda | ~83% | Régimen de promoción al vocabulario; las contribuciones MLP modulan tokens del output (logit lens, nostalgebraist, 2020). |
| Todas las capas | global | Cota superior empírica del efecto causal y contraste para evaluar localización vs. distribución de la representación. |

### Factor C — Intensidad $\alpha$

Cinco niveles validados en la literatura de feature steering (Turner et al., 2023; Anthropic, 2024):

| Nivel | $\alpha$ | Régimen esperado |
|---|---|---|
| Muy baja | 3 | Primeros signos medibles de afasia, alta especificidad. |
| Baja | 6 | Dosis efectiva: degradación on-target consistente. |
| Media | 10 | Degradación sustancial; emergen efectos colaterales. |
| Alta | 15 | Perturbación que se propaga a capas posteriores. |
| Muy alta | 20 | Colapso casi total de la representación; verificación del techo del efecto. |

### Factor D — Modelo

| Nivel | Modelo | Rol experimental |
|---|---|---|
| 0 | GPT-2 small (124M, 12 bloques) | Base empírica de la literatura de circuitos (Wang et al., 2022; Olsson et al., 2022). Permite barrido factorial completo. |
| 1 | Pythia-2.8B (32 bloques) | Banco de pruebas estándar para escalado en interpretabilidad; permite contrastar la hipótesis de representación lineal en mayor capacidad. |
| 2 | LLaVA-1.5-7B (Vicuna-7B + CLIP-ViT) | Introduce modalidad multimodal: pone a prueba si el anclaje perceptivo modifica la geometría del CAV textual (Patel & Pavlick, 2022). |

### Factor E — Categoría del concepto

| Nivel | Categoría | Propiedad distintiva |
|---|---|---|
| 0 | Animales | Entidades animadas, anclaje perceptivo, representaciones lineales bien separadas (Marks & Tegmark, 2023; Konkle & Caramazza, 2013). |
| 1 | Profesiones | Roles humanos institucionalizados; conocimiento de mundo aprendido distribucionalmente. |
| 2 | Objetos | Entidades físicas inanimadas; cierre de la tricotomía animado-humano / animado-no-humano / inanimado. |

---

## 3. Hipótesis

- **H1A — Método de extracción del CAV.** Los efectos causales serán cualitativamente consistentes entre Mean Difference y SVM. Si el concepto está codificado como una dirección lineal (Park, Choe & Veitch, 2023), ambos estimadores deben recuperar direcciones próximas y producir patrones de degradación comparables.
- **H1B — Localización de la intervención.** La ablación en capas medias producirá mayor degradación que en capas tardías, y la intervención global establecerá la cota superior del efecto.
- **H1C — Intensidad de la intervención.** Existirá una relación dosis-respuesta positiva y estadísticamente significativa entre $\alpha$ y $\Delta\mathrm{acc}$.
- **H1D — Escala del modelo.** El efecto de la ablación variará con la escala. Se espera que Pythia conserve mayor integridad que GPT-2 small, reflejando que mayor capacidad representacional dispersa la información conceptual.
- **H1E — Categoría conceptual.** Animales y objetos exhibirán direcciones lineales más nítidas y, por tanto, efectos más pronunciados y específicos que profesiones, dado que estas últimas se constituyen por integración de conocimiento institucional aprendido distribucionalmente.

---

## 4. Conceptos

Treinta conceptos concretos a nivel básico (Rosch, 1978):

| Categoría | Conceptos |
|---|---|
| Animales (7) | camel, squirrel, bee, horse, penguin, bird, spider |
| Objetos (6) | lock, toothbrush, violin, candle, fire, snowman |
| Profesiones (16) | fisherman, painter, knight, baker, doctor, barber, photographer, tailor, gardener, pirate, soldier, chef, musician, carpenter, writer, dentist |
| Personajes (1) | king |

---

## 5. Métrica

**R1 — $\Delta\mathrm{acc}$ (caída de precisión en clasificación CCT).**

$$\Delta\mathrm{acc} = \mathrm{acc}_{\text{base}} - \mathrm{acc}_{\text{int}}$$

donde $\mathrm{acc}_{\text{base}}$ es la precisión del modelo sin intervención y $\mathrm{acc}_{\text{int}}$ la precisión bajo la condición de ablación. Una alta degradación indica pérdida de conocimiento categorial.

---

## 6. Procedimiento experimental

### 6.1. Adaptación computacional del CCT

El CCT se adapta como prueba textual: el modelo recibe un prompt de la forma `"A {concepto} is related to ___"` con cuatro opciones — una correcta y tres distractores semánticamente plausibles pero pertenecientes a categorías incorrectas. La respuesta del modelo es el token con mayor log-softmax entre las cuatro opciones. Se trabaja con 30 conceptos, cada uno con 20–25 preguntas asociadas.

### 6.2. Estrategia *baseline-correct-only*

Solo se someten a ablación los ítems que el modelo responde correctamente sin intervención. Por construcción, $\mathrm{acc}_{\text{base}} = 1.0$ para cada ítem incluido. Cualquier caída observada bajo ablación es atribuible causalmente a la intervención sobre el residual stream y no a limitaciones preexistentes del modelo. Se exige `min_correct_items = 5` por concepto para que entre al análisis.

### 6.3. Construcción de los CAVs

Generación de datos:
1. JSON inicial de conceptos.
2. Enriquecimiento vía **ConceptNet** y **WordNet**: vecinos a 1 nodo de distancia, en inglés, longitud 2–30 caracteres.
3. Generación de frases positivas con cinco formatos (rotando posición del sujeto y predicado) y de frases negativas (alterando elementos distintos al concepto).

Extracción:
1. Carga del modelo (GPT-2 small / Pythia-2.8B / LLaVA-1.5-7B). Las activaciones del residual stream se recolectan con **TransformerLens** para los modelos solo-texto.
2. Para cada concepto y capa configurada, se obtiene un tensor de activaciones por estímulo.
3. **Mean Difference**: $\mathrm{CAV} = \mathrm{normalize}(\mu_{\text{pos}} - \mu_{\text{neg}})$.
4. **SVM lineal**: `LinearSVC` con `GridSearchCV` sobre $C \in \{0.01, 0.1, 1.0, 10.0\}$ y validación cruzada estratificada de 5 pliegues. `max_iter` = 10 000 (GPT-2), 30 000 (LLaVA), 50 000 (Pythia). Umbral de calidad: `accuracy ≥ 0.70`. CAVs por debajo se descartan.

Validación estadística:
- **Cohen's d** y **Hedges' g** (corrección por muestras pequeñas) sobre las proyecciones positivas y negativas, con IC 95 % vía aproximación t no-central. Umbrales convencionales: large ≥ 0.8, medium ≥ 0.5, small ≥ 0.2.
- **Similitud coseno** entre CAV-MeanDiff y CAV-SVM para evaluar convergencia entre métodos.

### 6.4. Aplicación de hooks y ablación

Se implementan hooks en todas las capas (`range(N_capas)`: 0–11 para GPT-2, 0–31 para Pythia y LLaVA), de modo que la ablación pueda aplicarse a cualquier capa individual o globalmente. Para cada combinación del diseño factorial se construye el hook correspondiente al CAV objetivo y se evalúan los efectos:
- **on-target**: `ablated_concept == measured_concept` (concepto cuya dirección se eliminó).
- **off-target**: conceptos distintos al ablacionado, para medir efectos colaterales.

Los resultados se persisten en `cct_test_results.json` por modelo, registrando para cada celda de ablación el concepto ablacionado, el concepto medido, el método de CAV, la capa intervenida, el valor de $\alpha$, las precisiones de baseline y de intervención, y $\Delta\mathrm{acc}$.

### 6.5. Plan de análisis estadístico

Modelo principal: **GLMM con enlace logit** sobre la variable binaria `is_correct_bool`. Efectos fijos: los cinco factores del diseño más sus interacciones de segundo orden. Intercepto aleatorio por `ablated_concept` (30 niveles) para controlar variabilidad entre conceptos.

Pruebas inferenciales: ANOVA Tipo III (Wald), pruebas post-hoc de Tukey y Bonferroni, medias marginales estimadas (`emmeans`), pruebas de razón de verosimilitudes para interacciones globales. Nivel de significancia $\alpha = 0.05$. Implementación cruzada en R y Python como validación numérica.

---

## 7. Resultados preliminares

El dataset unificado del CCT comprende **101 840 ítems** y **10 463 condiciones experimentales** (Pythia-2.8B: 4 800 condiciones; GPT-2 small: 4 320; LLaVA-1.5-7B: 1 343). Para garantizar comparabilidad entre modelos, el análisis principal se restringe a las condiciones que comparten los mismos cinco conceptos en los tres modelos (`tier all_3`), modalidad `all_tokens`, dirección on-target — 200 condiciones por modelo.

### 7.1. Estadísticos descriptivos

| Modelo | N | Media $\Delta\mathrm{acc}$ | DE | Mediana | Máx |
|---|---:|---:|---:|---:|---:|
| GPT-2 small (124M) | 200 | 0.348 | 0.283 | 0.218 | 1.000 |
| Pythia-2.8B | 200 | 0.243 | 0.203 | 0.226 | 0.889 |
| LLaVA-1.5-7B | 200 | 0.503 | 0.442 | 0.545 | 1.000 |

LLaVA-1.5-7B presenta distribución bimodal: muchas condiciones con baja degradación junto a un conjunto relevante con degradación total ($\Delta = 1.0$).

### 7.2. Relación dosis-respuesta (Factor C)

| Modelo | $\alpha=3$ | $\alpha=6$ | $\alpha=10$ | $\alpha=15$ | $\alpha=20$ |
|---|---:|---:|---:|---:|---:|
| GPT-2 small | 0.170 | 0.315 | 0.405 | 0.414 | 0.438 |
| Pythia-2.8B | 0.117 | 0.228 | 0.258 | 0.295 | 0.318 |
| LLaVA-1.5-7B | 0.088 | 0.363 | 0.592 | 0.704 | 0.767 |

La relación es **monotónicamente creciente en los tres modelos**, confirmando H1C. GPT-2 satura entre $\alpha=10$ y $\alpha=20$; Pythia es la más gradual; LLaVA exhibe la mayor sensibilidad. Figura: [analysis/figures/01_dose_response.png](analysis/figures/01_dose_response.png).

### 7.3. Efecto del método de extracción (Factor A)

| Modelo | Mean-Diff | SVM |
|---|---:|---:|
| GPT-2 small | 0.237 | 0.460 |
| Pythia-2.8B | 0.215 | 0.271 |
| LLaVA-1.5-7B | 0.508 | 0.498 |

La ventaja del SVM es marcada en GPT-2 small, moderada en Pythia, y se diluye en LLaVA — sugiriendo que la mayor escala o el anclaje multimodal acortan la distancia entre los dos estimadores. Figura: [analysis/figures/03_cav_method.png](analysis/figures/03_cav_method.png).

### 7.4. Efecto de la localización (Factor B)

| Modelo | Capa media (~50%) | Capa tardía (~75%) | Capa profunda (~83%) |
|---|---:|---:|---:|
| GPT-2 small | 0.373 | 0.304 | 0.251 |
| Pythia-2.8B | 0.285 | 0.251 | 0.200 |
| LLaVA-1.5-7B | 0.435 | 0.612 | 0.589 |

En GPT-2 y Pythia se observa el gradiente decreciente esperado por H1B (capas medias > tardías). En LLaVA el patrón se invierte: las capas tardías producen mayor degradación, posiblemente reflejando diferencias arquitectónicas entre el módulo de lenguaje y el ensamble multimodal. Figura: [analysis/figures/04_layer_heatmap.png](analysis/figures/04_layer_heatmap.png).

### 7.5. Síntesis

- **H1C confirmada** — relación dosis-respuesta monotónicamente creciente en los tres modelos.
- **H1A parcialmente confirmada** — los métodos coinciden cualitativamente en escalas grandes; en GPT-2 small el SVM domina al Mean Difference.
- **H1B confirmada en arquitecturas solo-texto** y **revertida en LLaVA**, lo que motiva análisis adicional sobre la organización funcional de modelos multimodales.
- **H1D confirmada parcialmente** — Pythia presenta menor degradación media que GPT-2 small, consistente con que la mayor capacidad representacional dispersa el concepto en más direcciones.
- **H1E** y los análisis de especificidad por categoría, así como el modelo GLMM completo, se reportan en `analysis/results/experiment_report.json` y se desarrollan en el documento de tesis.

---

## 8. Estructura del repositorio

```
.
├── README.md                                # Este documento (alcance: tesis CCT)
├── analysis/                                # Análisis transversal a los 3 modelos
│   ├── run_analysis.py                      # Script principal de análisis cross-model
│   ├── figures/                             # Figuras agregadas (01_dose_response, 02_specificity, ...)
│   ├── deep_viz/                            # Visualizaciones avanzadas (PCA, trayectorias, heatmaps)
│   └── results/                             # experiment_report.json
├── models/
│   ├── GPT2_small/                          # Pipeline para GPT-2 small (124M)
│   │   ├── concepts/                        # Definiciones JSON de los 30 conceptos
│   │   ├── cavs/                            # CAVs extraídos (no versionados, ver .gitignore)
│   │   ├── results/                         # cct_test_results.json del modelo
│   │   ├── test/cct_test.py                 # Adaptación del CCT
│   │   ├── experiment_config.py
│   │   ├── experiment_runner.py
│   │   ├── cav_extraction.py
│   │   ├── cav_statistical_tests.py
│   │   └── Dockerfile
│   ├── Pythia_28B/                          # Pipeline para Pythia-2.8B
│   │   ├── cavs/                            # No versionados
│   │   ├── data/
│   │   ├── results/
│   │   ├── experiment_config.py
│   │   ├── experiment_runner.py
│   │   ├── cav_extraction.py
│   │   └── Dockerfile
│   └── LLaVA_1_5_7B/                        # Pipeline para LLaVA-1.5-7B (multimodal)
│       ├── cavs/                            # No versionados
│       ├── data/
│       ├── results/
│       ├── experiment_config.py
│       ├── experiment_runner.py
│       ├── cav_extraction.py
│       ├── data_gen_multimodal.py
│       ├── pyvene_utils.py
│       └── Dockerfile
└── results/                                 # Resultados unificados cross-model
    ├── unified_cct_results.csv              # 101 840 ítems, todas las condiciones
    ├── unified_cct_results_agg.csv          # Agregados por condición
    └── cross_model_analysis_results.json
```

> **Nota sobre los CAVs.** Los archivos `.pt` en `models/*/cavs/` no se versionan (ver [.gitignore](.gitignore)). Son derivados reproducibles a partir de `cav_extraction.py` y los datos de `concepts/`. Cualquier clon puede regenerarlos ejecutando el pipeline.

---

## 9. Pipeline de ejecución

Cada modelo dispone de su propio entorno Docker reproducible. La estructura común es:

```bash
# 1. Construir entorno (por modelo)
cd models/GPT2_small        # o Pythia_28B / LLaVA_1_5_7B
docker build -t tesis-cct-gpt2 .
docker run --gpus all -it tesis-cct-gpt2 bash

# 2. Pipeline por modelo
python data_gen.py                  # Generación de datos (positivos/negativos por concepto)
python cav_extraction.py            # Extracción de CAVs (Mean Diff + SVM)
python cav_statistical_tests.py     # Validación estadística (Cohen's d, similitud coseno)
python experiment_runner.py         # Experimento factorial: aplica los hooks de ablación
python -m test.cct_test             # Evaluación CCT bajo todas las condiciones del DoE

# 3. Análisis transversal a los 3 modelos
python analysis/run_analysis.py     # Construye unified_cct_results.csv y figuras
```

### Requisitos por modelo

| Modelo | VRAM aprox. | RAM | Notas |
|---|---|---|---|
| GPT-2 small (124M) | ~0.5 GB | 8 GB | Permite barrido factorial completo en GPU consumer. |
| Pythia-2.8B | ~12 GB | 16 GB | A100 / H100 recomendada. |
| LLaVA-1.5-7B | ~16 GB | 24 GB | Imágenes preprocesadas con CLIP-ViT. |

---

## 10. Referencias

- Anthropic Research. (2021). *A Mathematical Framework for Transformer Circuits.*
- Anthropic. (2024). *Steering Vectors for Language Models.*
- Bau, D., et al. (2018). *GAN Dissection.* ICLR.
- Bau, D., et al. (2020). *Understanding the role of individual units in a deep neural network.* PNAS.
- Bozeat, S., et al. (2000). *Non-verbal semantic impairment in semantic dementia.* Neuropsychologia, 38(9), 1207–1215.
- Caramazza, A., & Shelton, J. R. (1998). *Domain-specific knowledge systems in the brain.* Journal of Cognitive Neuroscience, 10(1), 1–34.
- Castellana, A. (2011). *Trastornos del lenguaje en lesiones cerebrales focales.*
- Chiang, W.-L., et al. (2023). *Vicuna: An Open-Source Chatbot.*
- Chughtai, B., Chan, L., & Nanda, N. (2023). *A Toy Model of Universality.*
- Conmy, A., et al. (2023). *Towards Automated Circuit Discovery.* NeurIPS.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall/CRC.
- Elhage, N., et al. (2022). *Toy Models of Superposition.* arXiv:2209.10652.
- Escamilla, M. (2020). *Afasia semántica: comprensión vs. expresión.*
- Ganguli, D., et al. (2022). *Predictability and Surprise in Large Generative Models.*
- Geva, M., et al. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP.
- Gurnee, W., & Tegmark, M. (2024). *Language Models Represent Space and Time.* ICLR.
- Gurnee, W., et al. (2024). *Universal Neurons in GPT2 Language Models.*
- He, J., et al. (2024). *Survey on Mechanistic Interpretability.*
- IntuitionLabs. (2025). *Activation analysis in transformer models.*
- Jawahar, G., et al. (2019). *What Does BERT Look At?* BlackboxNLP.
- Kim, B., et al. (2018). *Interpretability Beyond Feature Attribution: TCAV.* ICML.
- Konkle, T., & Caramazza, A. (2013). *Tripartite Organization of the Ventral Stream.* Journal of Neuroscience.
- Kriegeskorte, N., et al. (2008). *Representational similarity analysis.* Frontiers in Systems Neuroscience.
- Lambon Ralph, M. A., et al. (2017). *The neural and computational bases of semantic cognition.* Nature Reviews Neuroscience.
- Li, H. (2024). *Transformer architecture: a primer.*
- Liu, H., et al. (2024). *Visual Instruction Tuning.* (LLaVA).
- Marks, S., & Tegmark, M. (2023). *The Geometry of Truth.* arXiv:2310.06824.
- Meng, K., et al. (2022). *Locating and Editing Factual Associations in GPT.* NeurIPS.
- Nanda, N. (2022). *A comprehensive mechanistic interpretability explainer.*
- nostalgebraist. (2020). *Interpreting GPT: the logit lens.*
- Olsson, C., et al. (2022). *In-context Learning and Induction Heads.* Anthropic.
- Panickssery, A., Rimsky, N., et al. (2024). *Steering Llama-2 with Contrastive Activation Addition.*
- Park, K., Choe, Y., & Veitch, V. (2023). *The Linear Representation Hypothesis.* arXiv:2311.03658.
- Patel, R., & Pavlick, E. (2022). *Mapping Language Models to Grounded Conceptual Spaces.* ICLR.
- Patterson, K., et al. (2007). *Where do you know what you know?* Nature Reviews Neuroscience.
- Ramos-Galarza, C. (2023). *Modelos computacionales y trastornos del lenguaje.*
- Reilly, J., et al. (2024). *Conceptual knowledge in long-term memory.*
- Ribeiro, F. (2023). *Camel and Cactus Test: aplicaciones clínicas.*
- Rosch, E. (1978). *Principles of Categorization.* Lawrence Erlbaum.
- Rubio Díaz, J. (2015). *Pobreza léxica en afasia semántica.*
- Speer, R., et al. (2017). *ConceptNet 5.5.* AAAI.
- Talon, J., et al. (2024). *Interpretability and AI ethics.*
- Tenney, I., et al. (2019). *BERT Rediscovers the Classical NLP Pipeline.* ACL.
- Toledo Rodríguez, M. (2021). *Parafasias y estereotipias en afasia semántica.*
- Turner, A., et al. (2023). *Activation Addition: Steering Language Models Without Optimization.* arXiv:2308.10248.
- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
- Wang, K., et al. (2022). *Interpretability in the Wild: Indirect Object Identification in GPT-2 small.* ICLR.
- Warrington, E. K., & Shallice, T. (1984). *Category specific semantic impairments.* Brain, 107(3), 829–854.
- Zou, A., et al. (2023). *Representation Engineering.* arXiv:2310.01405.
