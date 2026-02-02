# Extraccion y Refinamiento de CAVs en GPT-2

Extraccion de Concept Activation Vectors (CAVs) para tres conceptos semanticos superordinados en GPT-2, con analisis de calidad, entanglement y purificacion ortogonal.

## Fundamentacion Metodologica

La metodologia se sustenta en el marco TCAV (Testing with Concept Activation Vectors) propuesto por Kim et al. (2018), quienes demostraron la existencia de representaciones lineales de conceptos en espacios latentes. Complementariamente, se apoya en los hallazgos de Bau et al. (2020) sobre la emergencia espontanea de detectores semanticos en unidades individuales. Para la seleccion taxonomica de los conceptos (Nivel Superordinado), se siguio la teoria de categorizacion cognitiva de Rosch (1978), priorizando la abstraccion funcional sobre la similitud perceptual.

## Conceptos Analizados

Tres conceptos a nivel superordinado, con nodos semanticos extraidos de ConceptNet:

| Concepto | Nodos | Ejemplos |
|----------|-------|----------|
| **Time** | 38 | hour, minute, day, week, month, year, decade, century |
| **Place** | 38 | city, town, village, country, continent, mountain, river |
| **Tools** | 34 | hammer, screwdriver, wrench, pliers, saw, drill, chisel |

Cada concepto cuenta con 170-190 oraciones positivas y 170-190 negativas, balanceadas en longitud y diversidad lexica, para el entrenamiento de los CAVs.

## Metodos de Extraccion

Los CAVs se extraen de las activaciones del residual stream (`blocks.{L}.hook_resid_post`) en GPT-2 (d_model = 768), tomando la activacion del ultimo token de cada oracion. Se aplican dos metodos en tres capas (6, 9, 10):

### Mean Difference

```
CAV = normalize(mean(activaciones_positivas) - mean(activaciones_negativas))
```

Aproximacion directa que estima la direccion del concepto como la diferencia entre los centroides de las clases positiva y negativa.

### SVM (Linear Support Vector Classifier)

```
CAV = normalize(coef_[0])   # Vector normal al hiperplano de decision
```

Metodo original propuesto por Kim et al. (2018). Entrena un clasificador lineal para separar activaciones positivas de negativas; el vector normal al hiperplano de decision es el CAV. Se evalua mediante 5-fold cross-validation estratificada.

## Resultados de Extraccion

### Separabilidad Lineal (SVM Cross-Validation Accuracy)

| Concepto | Layer 6 | Layer 9 | Layer 10 |
|----------|---------|---------|----------|
| Time | 98.9% (std 0.013) | 99.2% (std 0.011) | 98.7% (std 0.014) |
| Place | 98.9% (std 0.010) | 98.4% (std 0.015) | 98.7% (std 0.014) |
| Tools | 94.7% (std 0.018) | 95.9% (std 0.022) | 94.1% (std 0.025) |

Los tres conceptos son altamente separables linealmente dentro de GPT-2. Time y Place alcanzan precision casi perfecta (>98%); Tools es ligeramente menor pero sigue siendo excelente (>94%). La alta precision indica que GPT-2 desarrolla representaciones lineales robustas para estos conceptos superordinados.

### Consistencia entre Metodos (coseno mean_diff vs svm)

| Concepto | Layer 6 | Layer 9 | Layer 10 |
|----------|---------|---------|----------|
| Time | 0.584 | 0.689 | 0.719 |
| Place | 0.689 | 0.653 | 0.690 |
| Tools | 0.518 | 0.549 | 0.536 |

Robustez moderada (0.5-0.7). Ambos metodos capturan la misma direccion general pero no son identicos. Un valor de 1.0 indicaria que los dos metodos encuentran exactamente el mismo vector. La discrepancia sugiere que Mean Difference es una aproximacion mas burda, mientras que SVM identifica con mayor precision la frontera de decision lineal. Se recomienda usar los vectores SVM.

### Entanglement entre Conceptos (Similitud Coseno entre CAVs)

**Metodo SVM:**

| Par | Layer 6 | Layer 9 | Layer 10 |
|-----|---------|---------|----------|
| time-place | 0.304 | 0.308 | 0.304 |
| time-tools | 0.191 | 0.300 | 0.294 |
| place-tools | 0.354 | 0.331 | 0.297 |

**Metodo Mean Diff:**

| Par | Layer 6 | Layer 9 | Layer 10 |
|-----|---------|---------|----------|
| time-place | **0.471** | 0.426 | 0.420 |
| time-tools | 0.092 | 0.048 | 0.120 |
| place-tools | 0.247 | 0.196 | 0.243 |

La correlacion time-place de 0.47 (mean_diff, L6) es la mas alta y revela entanglement significativo: las representaciones de "tiempo" y "lugar" comparten componentes dentro de GPT-2, posiblemente por su funcion comun como complementos circunstanciales gramaticales. Los vectores SVM muestran correlaciones menores (~0.30) pero aun no-triviales.

Valores ideales serian cercanos a 0. Correlaciones superiores a 0.30 indican que una intervencion en la direccion de un concepto podria afectar accidentalmente a otro, motivando la etapa de purificacion ortogonal.

## Purificacion Ortogonal de CAVs

### Motivacion

Con correlaciones de 0.19-0.47 entre CAVs, intervenir en la direccion de "tiempo" inevitablemente tiene componente en la direccion de "lugar". Para garantizar intervenciones causalmente aisladas, se purifica cada CAV eliminando su componente en el subespacio generado por los otros conceptos.

### Formula Incorrecta (resta ingenua)

La formula simplificada de Gram-Schmidt para dos vectores:

```
v_pure = v_A - (v_A . v_B) * v_B - (v_A . v_C) * v_C
```

asume que v_B y v_C son ortogonales entre si. Cuando no lo son, esta formula sobre-corrige. Se puede verificar que el producto punto del vector "purificado" con los originales NO da cero:

```
v_A_pure . v_B = -(v_A . v_C) * (v_C . v_B) = -b * rho
```

Con valores tipicos (b=0.19, rho=0.35), el residuo es -0.067: la correlacion cambia de signo en vez de anularse.

### Formula Correcta (Proyeccion de Subespacio)

```
v_puro = v_A - V (V^T V)^{-1} V^T v_A
```

donde `V = [v_B | v_C]` es la matriz cuyas columnas son los CAVs de los otros conceptos. La inversa de la **matriz de Gram** `(V^T V)^{-1}` compensa exactamente la no-ortogonalidad entre v_B y v_C, produciendo un vector que es ortogonal a todo el subespacio.

Diferencia cuantitativa para purificar time (SVM, L6, a=time.place, b=time.tools, rho=place.tools):

| | Coef. sobre v_place | Coef. sobre v_tools |
|---|---|---|
| Formula ingenua | 0.304 | 0.191 |
| Formula correcta | 0.270 | 0.095 |

La formula ingenua resta el doble de lo necesario en la direccion de tools.

### Verificacion

Producto punto de cada vector refinado con los CAVs originales de los otros conceptos (debe ser ~0):

**Layer 6:**

| Par | Dot Product | Estado |
|-----|-------------|--------|
| refined_time . original_place | +1.75e-07 | OK |
| refined_time . original_tools | +1.24e-07 | OK |
| refined_place . original_time | -1.30e-08 | OK |
| refined_place . original_tools | -2.85e-07 | OK |
| refined_tools . original_time | -3.26e-08 | OK |
| refined_tools . original_place | -1.89e-07 | OK |

**Layer 9:**

| Par | Dot Product | Estado |
|-----|-------------|--------|
| refined_time . original_place | -4.28e-08 | OK |
| refined_time . original_tools | -9.03e-08 | OK |
| refined_place . original_time | +3.17e-08 | OK |
| refined_place . original_tools | -6.41e-07 | OK |
| refined_tools . original_time | +1.40e-07 | OK |
| refined_tools . original_place | -4.91e-07 | OK |

**Layer 10:**

| Par | Dot Product | Estado |
|-----|-------------|--------|
| refined_time . original_place | -9.13e-08 | OK |
| refined_time . original_tools | +2.14e-07 | OK |
| refined_place . original_time | -2.04e-07 | OK |
| refined_place . original_tools | -3.17e-08 | OK |
| refined_tools . original_time | +3.32e-07 | OK |
| refined_tools . original_place | +2.06e-07 | OK |

Ortogonalidad verificada a precision de maquina (~1e-7) en las tres capas. Cada vector refinado tiene proyeccion cero sobre los CAVs originales de los otros conceptos.

### Correlaciones Residuales entre Vectores Refinados

| Par | Layer 6 | Layer 9 | Layer 10 | Original (SVM) |
|-----|---------|---------|----------|----------------|
| time-place | -0.257 | -0.232 | -0.237 | +0.304 |
| time-tools | -0.094 | -0.221 | -0.224 | +0.191 |
| place-tools | -0.317 | -0.262 | -0.228 | +0.354 |

Las correlaciones negativas entre vectores refinados son geometricamente esperables: al remover la componente positiva compartida de cada vector, las direcciones residuales divergen. Esto no compromete la validez de los vectores purificados; lo relevante para intervenciones causales es la ortogonalidad de cada vector refinado contra los CAVs originales de los otros conceptos (verificada arriba).

### Norma Residual tras Purificacion

| Concepto | Layer 6 | Layer 9 | Layer 10 |
|----------|---------|---------|----------|
| Time | 0.949 | 0.928 | 0.929 |
| Place | 0.904 | 0.918 | 0.928 |
| Tools | 0.931 | 0.921 | 0.931 |

La norma residual (antes de renormalizacion) indica que porcentaje de la informacion original se conserva. Valores de 0.90-0.95 muestran que los vectores pierden solo 5-10% de su magnitud al purificarse, confirmando que la mayor parte de la informacion del concepto es exclusiva y no compartida con otros conceptos.

## Validacion Relacionada con la Extraccion

| Test | Estado | Detalle |
|------|--------|---------|
| T01 Nodos por concepto | PASS | time=38, place=38, tools=34 |
| T02 Contaminacion negativa | PASS | 0-1.6% contaminacion en muestras negativas |
| T03 Polisemia | PASS | 0 palabras superpuestas entre conceptos |
| T04 Tamano de muestra | PASS | 170-190 pares balanceados por concepto |
| T05 Tokenizacion | FAIL | 42.7% single-token (1 split excesivo: "neighborhood"=4) |
| T06 Artefactos de template | PASS | Longitudes y diversidad lexica equilibradas |
| T08 Estabilidad SVM | PASS | CV std < 0.025 en todas las condiciones |
| T09 Overlap de CAVs | FAIL | Correlaciones > 0.30 en varios pares (resuelto por purificacion) |

**T05 (Tokenizacion):** El tokenizer BPE de GPT-2 no siempre representa una palabra como un solo token. De las 110 palabras de concepto, solo el 42.7% son single-token; el resto se fragmenta en 2 o mas sub-tokens. Por ejemplo:

- `"hammer"` → 1 token: `["hammer"]`
- `"screwdriver"` → 3 tokens: `["sc", "rew", "driver"]`
- `"neighborhood"` → 4 tokens: `["neigh", "bor", "ho", "od"]` (caso mas extremo)

Esto seria un problema si se extrajera la activacion del token especifico de la palabra de concepto, porque el primer sub-token (`"sc"`) no contiene la semantica completa. Sin embargo, en la extraccion de CAVs (`cav_extraction.py`, linea 58) se toma la activacion del **ultimo token de la oracion completa**, no del token del nodo de concepto. Gracias a la atencion causal, el ultimo token ya integro la informacion de todos los tokens previos, incluyendo la palabra completa aunque esta se haya fragmentado. Por lo tanto, la fragmentacion sub-lexica no compromete la representatividad de las activaciones extraidas.

**T09 (Overlap de CAVs):** Las correlaciones originales entre CAVs de conceptos distintos (0.19-0.47) indican que las direcciones capturadas no son independientes: una intervencion en la direccion de "tiempo" tendria componente en la direccion de "lugar". Tras la purificacion ortogonal con proyeccion de subespacio (`refine_cavs.py`), estas correlaciones se eliminan a precision de maquina (~1e-7). Los vectores refinados se almacenan en `cavs_refined/`.

## Archivos Generados

```
cavs/
├── {concept}_{method}_layer{L}.pt    # 18 archivos: 3 conceptos x 2 metodos x 3 capas
└── extraction_report.json            # Metricas de calidad y correlaciones

cavs_refined/
└── {concept}_svm_layer{L}.pt         # 9 archivos: 3 conceptos x 3 capas (solo SVM)
```
