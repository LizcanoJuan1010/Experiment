# Resultados — Pythia 2.8B

Resultados del experimento de ablacion conceptual en Pythia 2.8B (EleutherAI).
30 conceptos, 3 capas de extraccion (L16, L24, L27), 2 metodos CAV, 5 intensidades.

---

## CCT Test (Camel and Cactus Test)

**Archivo:** `cct_test_results.json`
**Evaluaciones:** 4,800 (30 conceptos x 2 metodos x 4 condiciones de capa x 5 alphas)

### Diseno factorial

| Factor | Valores |
|--------|---------|
| Conceptos | 30 (animales, profesiones, objetos) |
| Metodos CAV | `mean_diff`, `svm` |
| Capas | L16, L24, L27, all_layers |
| Alphas | 3.0, 6.0, 10.0, 15.0, 20.0 |
| Benchmark | BEA Association (728 items, 30 conceptos) |

### Resultados principales

#### Especificidad por metodo

| Metodo | On-target | Off-target | Indice especificidad |
|--------|-----------|------------|----------------------|
| `svm` | 0.304 | 0.163 | **1.87x** |
| `mean_diff` | 0.265 | 0.508 | 0.52x |

**Hallazgo clave:** SVM es el unico metodo con especificidad real en Pythia 2.8B.
`mean_diff` produce mas dano colateral (off-target) que efecto dirigido (on-target).
Esto contrasta con modelos mas pequenos donde mean_diff suele ser competitivo.

#### Efecto por capa (SVM)

| Capa | On-target (delta acc) |
|------|-----------------------|
| L16 | 0.340 |
| L24 | 0.202 |
| L27 | 0.192 |
| all_layers | **0.480** |

La capa 16 (mitad del modelo, 32 capas totales) es la mas efectiva de las capas
individuales. La condicion `all_layers` produce el mayor efecto total pero
potencialmente menor especificidad.

#### Curva dosis-respuesta (SVM, L16)

| Alpha | Delta accuracy |
|-------|----------------|
| 3.0 | 0.168 |
| 6.0 | 0.282 |
| 10.0 | 0.380 |
| 15.0 | 0.432 |
| 20.0 | 0.440 |

El efecto satura entre alpha=15 y alpha=20. Punto de operacion optimo: **alpha=15**.

#### Efecto por categoria semantica (SVM, todos los alphas/capas)

| Categoria | Conceptos | Delta acc medio |
|-----------|-----------|-----------------|
| Profesiones | baker, barber, carpenter, chef, dentist, doctor, fisherman, gardener, king, knight, musician, painter, photographer, pirate, soldier, tailor, writer | **0.331** |
| Objetos | candle, fire, lock, snowman, toothbrush, violin | 0.235 |
| Animales | bee, bird, camel, horse, penguin, spider, squirrel | 0.214 |

Las profesiones son mas ablacionables: sus representaciones son mas localizadas
en el espacio de activaciones de Pythia.

#### Baselines (sin ablacion)

Accuracy promedio del modelo en el benchmark BEA Association sin intervencion:
- Maximo: horse (0.720), camel (0.640), baker (0.680)
- Minimo: fire (0.280), snowman (0.318), fisherman (0.250)
- Media: ~0.48 (el modelo no es perfecto en este benchmark)

Solo conceptos con baseline >= 5 respuestas correctas fueron incluidos en el analisis.

#### Ablacion maxima por concepto (mejor configuracion individual)

| Concepto | Max delta | Concepto | Max delta |
|----------|-----------|----------|-----------|
| barber | 1.000 | chef | 1.000 |
| carpenter | 1.000 | dentist | 1.000 |
| doctor | 1.000 | fisherman | 1.000 |
| gardener | 1.000 | king | 1.000 |
| knight | 1.000 | lock | 1.000 |
| musician | 1.000 | photographer | 1.000 |
| soldier | 1.000 | spider | 1.000 |
| toothbrush | 1.000 | writer | 1.000 |
| baker | 0.941 | snowman | 0.857 |
| fire | 0.857 | pirate | 0.800 |
| candle | 0.889 | painter | 0.778 |
| penguin | 0.706 | violin | 0.786 |
| bird | 0.769 | camel | 0.625 |
| bee | 0.692 | horse | 0.611 |
| squirrel | 0.500 | tailor | 0.778 |

16 de 30 conceptos alcanzan ablacion total (delta=1.0) con la configuracion optima.

---

## Pendientes

Los siguientes tests aun no se han ejecutado para Pythia 2.8B:

| Test | Script | Estado |
|------|--------|--------|
| BEA Naming | `test/bea_naming_test.py` | Pendiente |
| BEA Odd-one-out | `test/bea_oddoneout_test.py` | Pendiente |
| BEA Synonym | `test/bea_synonym_test.py` | Pendiente |
| Caperucita | `test/caperucita_test.py` | Pendiente |
| ANOVA BEA | `bea_anova_analysis.py` | Pendiente |

---

## Archivos en este directorio

| Archivo | Descripcion |
|---------|-------------|
| `cct_test_results.json` | 4,800 evaluaciones CCT con ablacion factorial |
| `corpus_training_report.json` | Metadata de extraccion de corpus para CAV training |
| `validation_report.json` | Reporte de validacion del pipeline |

---

## Comparacion con GPT-2 small

| Aspecto | GPT-2 small (117M) | Pythia 2.8B |
|---------|--------------------|-------------|
| Metodo optimo | mean_diff competitivo | SVM necesario |
| Especificidad SVM | ~1.5-2x | **1.87x** |
| Especificidad mean_diff | ~1.0x | 0.52x (inverted) |
| Capa optima | L6 (mitad de 12) | L16 (mitad de 32) |
| Alpha optimo | ~10 | ~15 |
| Conceptos ablacionables al 100% | ~10/27 | **16/30** |

Pythia 2.8B muestra mayor capacidad de ablacion selectiva con SVM, pero
`mean_diff` produce interferencia masiva, sugiriendo que sus representaciones
conceptuales son mas entrelazadas y requieren una direccion de separacion
mas precisa.

---

## Generado por

- [`test/cct_test.py`](../test/cct_test.py) — CCT factorial ablation test
- [`cav_extraction.py`](../cav_extraction.py) — Extraccion de CAVs (30 conceptos x 2 metodos x 3 capas)
- [`data_gen.py`](../data_gen.py) — Generacion de datos de entrenamiento
- [`experiment_runner.py`](../experiment_runner.py) — Hook de ablacion
- [`experiment_config.py`](../experiment_config.py) — Configuracion del experimento
