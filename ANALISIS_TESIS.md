# Analisis Integral del Proyecto: Que Queda, Que Falta y Que Descubrimos

---

## 1. VERIFICACION DE HIPOTESIS: ¿Se logro lo planteado en la tesis?

La tesis planteo 4 hipotesis principales. A continuacion el veredicto para cada una:

### H1A: La Proyeccion Ortogonal (T2) produce mayor degradacion que la Sustraccion (T1)

| Veredicto | **PARCIALMENTE CONFIRMADA** |
|-----------|----------------------------|

**Evidencia a favor:**
- En el experimento factorial single-layer, projection produce los mayores Delta R1 observados (place: +0.700 con mean_diff+projection vs ~0.033 con subtraction)
- La proyeccion es consistentemente mas agresiva en todas las condiciones

**Evidencia en contra / matices:**
- Los efectos enormes de projection+mean_diff son **anomalos** (tools MEJORA de 0.133 a 1.0, time sube a 1.0) — esto sugiere **corrupcion representacional**, no ablacion selectiva
- Con SVM (el metodo estadisticamente validado), projection produce Delta R1 de solo +0.033 en single-layer, apenas mayor que subtraction
- El **mejor resultado real** (Delta R1 = +0.333 para place) viene del multi-layer con **sustraccion** de vectores SVM sin normalizar, no de proyeccion

**Conclusion para la tesis:** Se puede afirmar que la proyeccion es mas agresiva, pero hay que matizar que "mas agresivo" no significa "mejor" — la proyeccion tiende a corromper representaciones en lugar de borrar selectivamente el concepto. La sustraccion multi-capa resulto mas controlada y confiable.

---

### H1B: Las Capas Medias (L1) causan mayor degradacion que las Capas Tardias (L2)

| Veredicto | **NO CONFIRMADA directamente — REFORMULADA** |
|-----------|----------------------------------------------|

**Lo que se encontro:**
- En single-layer: L6 (media) y L10 (tardia) producen efectos similares y generalmente debiles (Delta R1 ~ 0.00-0.03)
- La verdadera diferencia no esta entre capas individuales, sino entre **modos de intervencion**:
  - Single (1 capa): Delta R1 ~ 0.03
  - Multi early (3 capas): Delta R1 ~ 0.10
  - **Multi all (12 capas): Delta R1 ~ 0.33** (10x mejor)

**Hallazgo real:** Los conceptos NO estan localizados en una capa especifica — estan **distribuidos a traves de todas las capas**. La intervencion efectiva requiere intervenir simultaneamente en multiples capas. Este es uno de los descubrimientos mas importantes del proyecto.

**Conclusion para la tesis:** La hipotesis original sobre capas medias vs tardias no se sostiene como fue planteada. En su lugar, se descubrio algo mas interesante: la representacion distribuida de conceptos, lo cual se puede reportar como un hallazgo que supera la hipotesis original.

---

### H1C: Relacion dosis-respuesta positiva entre intensidad (alpha) y degradacion

| Veredicto | **CONFIRMADA** |
|-----------|----------------|

**Evidencia solida:**
- Test T17 (monoticidad): 83.3% de las curvas son monotonicas — PASS
- Para place con multi-all:
  - alpha=1.0 → Delta R1 = 0.10
  - alpha=4.0 → Delta R1 = 0.33
- Para time con multi-all:
  - alpha=1.0 → Delta R1 = 0.10
  - alpha=4.0 → Delta R1 = 0.13
- Los controles de amplificacion confirman la direccionalidad (amplificar mejora R1, restar la empeora)
- En LLaVA extreme ablation: alpha 6→10→15→20 muestra degradacion progresiva (respuestas coherentes → parafasias → perseveracion → colapso completo)

**Conclusion para la tesis:** Esta es la hipotesis mejor soportada. Hay una relacion clara dosis-respuesta que es consistente a traves de modelos, conceptos y configuraciones.

---

### H1D: Hiperonimos causan mayor degradacion que hiponimos

| Veredicto | **NO EVALUADA** |
|-----------|-----------------|

Esta hipotesis **no fue implementada experimentalmente**. El proyecto trabajo exclusivamente con conceptos a nivel de hiperonimo (time, place, tools). No se incluyeron hiponimos como condicion experimental separada.

**Nota para la tesis:** Esto queda como trabajo futuro. Se podria comparar, por ejemplo, ablacionar "animal" (hiperonimo) vs "golden_retriever" (hiponimo) y medir cual produce mayor degradacion.

---

## 2. QUE QUEDA / QUE YA ESTA HECHO Y ES PUBLICABLE

### 2.1 Resultados solidos que se pueden incluir directamente en la tesis

| Resultado | Modelo | Evidencia | Calidad |
|-----------|--------|-----------|---------|
| Los CAVs SVM capturan estructura conceptual genuina | GPT-2 | TCAV p<0.05 en 9/9, selectividad >0.30, Cohen's d 3.1-4.9 | **Alta** |
| SVM es superior a Mean Difference | GPT-2 | TCAV: SVM 9/9 sig. vs mean_diff 1/9 sig. | **Alta** |
| La ablacion multi-capa es ~10x mas efectiva que single | GPT-2 | Delta R1: 0.333 vs 0.033 | **Alta** |
| Los controles de amplificacion confirman direccionalidad | GPT-2 | Tools amplification: 0.133→0.900 (+0.767) | **Alta** |
| La especificidad esta confirmada | GPT-2 | Delta R1 off-target ~ 0 en 6/6 condiciones | **Alta** |
| Relacion dosis-respuesta | GPT-2/LLaVA | 83.3% curvas monotonicas + evidencia cualitativa | **Alta** |
| Patrones cualitativos analogos a afasia | GPT-2/LLaVA | Perseveracion, parafasias, colapso gradual | **Media-Alta** |
| Conceptos visuales son mas linealmente separables | LLaVA | TCAV 0.9-1.0 vs GPT-2 0.7-0.83 | **Media** |
| Los conceptos tienen representaciones distribuidas | GPT-2 | Multi-all >> multi-early >> single | **Alta** |
| SVM converge en las 12 capas con CV 80-91% | GPT-2 | Cross-validation 5-fold | **Alta** |
| Degradacion cualitativa en LLaVA con extreme ablation | LLaVA | Cat→chimpanzee→umbrella→colapso total | **Media-Alta** |

### 2.2 Figuras publicables ya generadas

14 figuras (PDF+PNG) para GPT-2:
- Curvas dosis-respuesta (F01-F02)
- Heatmap de especificidad (F04)
- Comparacion sustraccion vs proyeccion (F05)
- Dashboard de validacion (F10)
- Distribucion nula TCAV (F11)
- Bootstrap CI (F13)
- Cohen's d CI (F14)

### 2.3 Framework y pipeline completo

- **20+ scripts Python** organizados y documentados
- **298 tensores PyTorch** (CAVs extraidos)
- **~1,500 condiciones experimentales** ejecutadas
- **Suite de 20 tests automatizados** (17 PASS, 1 FAIL, 2 INFO)
- **6 benchmarks** construidos con proveniencia documentada
- **Docker reproducible**

---

## 3. QUE FALTA Y QUE SE DEBERIA MEJORAR

### 3.1 Critico (deberia abordarse para completar la tesis)

| # | Problema | Impacto | Solucion propuesta | Esfuerzo |
|---|---------|---------|-------------------|----------|
| 1 | **Baseline de Tools invalido** (R1=0.133) | Todos los resultados de ablacion de tools son dificilmente interpretables | Redisenar el benchmark R1 para tools con items que GPT-2 pueda resolver (baseline > 0.50) | Medio |
| 2 | **Pythia-2.8B incompleto** | No hay evidencia de escalabilidad; muchos tests SKIP | Ejecutar al menos el experimento multilayer en Pythia para confirmar que los hallazgos se mantienen a mayor escala | Alto |
| 3 | **LLaVA baselines invalidos** (2/3 conceptos < 50%) | Solo labrador tiene baseline funcional; golden_retriever y tabby_cat no medibles | Cambiar conceptos a categorias visualmente mas distintas (ej: perro vs auto vs flor) y reconstruir benchmarks | Alto |
| 4 | **R2 tiene efecto techo** (~0.99) | La metrica R2 (similitud semantica) es insensible a la ablacion; no aporta informacion util | Crear benchmark R2 mas discriminativo (pares de oraciones mas largos o relaciones semanticas mas sutiles) | Medio |
| 5 | **Falta ANOVA formal** | La tesis plantea ANOVA como metodo estadistico pero no se ejecuto | Ejecutar ANOVA factorial sobre los 72 resultados del factorial y los 39 del multilayer | Bajo |

### 3.2 Importante (mejoraria la tesis significativamente)

| # | Mejora | Justificacion |
|---|--------|---------------|
| 6 | **Ejecutar el sweep completo** y analizar las 1,296 condiciones | Los scripts estan listos; falta solo ejecutar y generar figuras con los "sweet spots" optimos |
| 7 | **Agregar mas conceptos** (de los 22 definidos en experiment_config.py) | Solo se usaron 3 de 22; mas conceptos darian mayor generalidad |
| 8 | **Metricas de generacion post-ablacion** | Agregar perplejidad, coherencia y diversidad lexica para cuantificar la degradacion cualitativa |
| 9 | **Purificacion ortogonal completa** | refine_cavs.py existe pero los CAVs refinados no se usaron en los experimentos principales |
| 10 | **Hipotesis H1D** (hiperonimos vs hiponimos) | Fue planteada pero nunca evaluada; incluir al menos un experimento piloto |

### 3.3 Opcional (trabajo futuro genuino)

| # | Linea | Descripcion |
|---|-------|-------------|
| 11 | Ablacion no-lineal | Probar MLP probes para conceptos que no sean linealmente separables |
| 12 | Interaccion entre conceptos | Ablacionar 2+ conceptos simultaneamente |
| 13 | Recuperacion post-ablacion | Fine-tuning para restaurar un concepto borrado |
| 14 | Transfer de CAVs | Probar si los CAVs de GPT-2 funcionan en Pythia u otros modelos de la misma familia |
| 15 | SAE (Sparse Autoencoder) | La tesis lo menciona en el marco teorico pero no se implemento; podria ser una comparacion poderosa |

---

## 4. QUE DESCUBRIMOS: HALLAZGOS PRINCIPALES

### Descubrimiento 1: Los conceptos son lineales pero distribuidos

Los clasificadores SVM logran separar activaciones conceptuales con 80-91% de precision en **todas las 12 capas** de GPT-2. Pero la ablacion en una sola capa produce efectos minimos (~3%). Solo al intervenir en las 12 capas simultaneamente se logra un efecto fuerte (33%).

**Implicacion:** Los conceptos se representan como **direcciones lineales distribuidas** a lo largo de toda la red, no localizados en una capa especifica. Esto tiene paralelos con la representacion distribuida en el cerebro.

### Descubrimiento 2: SVM captura informacion genuina; Mean Difference captura ruido

El TCAV permutation test revelo una disociacion clara:
- SVM: estadisticamente significativo en **todas** las 9 condiciones evaluadas
- Mean Difference: significativo en solo **1 de 9** condiciones

Pero paradojicamente, mean_diff es mas **estable** (bootstrap coseno ~0.97 vs ~0.80 para SVM). Esto resuelve una tension metodologica: estabilidad ≠ informatividad.

### Descubrimiento 3: La amplificacion revela conocimiento latente

El caso mas revelador es Tools: el modelo tiene un baseline de solo 13.3% en la tarea R1, pero al **amplificar** el vector de concepto (sumar en lugar de restar), R1 sube a 90%. Esto demuestra que:
- El modelo **SI tiene** conocimiento sobre herramientas
- Ese conocimiento no se activa correctamente en el benchmark original
- Los CAVs capturan la direccion correcta del concepto

### Descubrimiento 4: Patrones cualitativos analogos a la afasia clinica

La degradacion producida por la ablacion muestra patrones notablemente similares a los sintomas de afasia semantica:

| Patron observado | Equivalente clinico | Ejemplo concreto |
|-----------------|---------------------|------------------|
| Repeticion de fragmentos ("day-day-day", "catChain catChain") | Perseveracion | GPT-2 y LLaVA con alpha alto |
| Sustitucion semantica (gato → chimpance → paraguas) | Parafasias semanticas | LLaVA extreme ablation alpha=10 |
| Perdida de coherencia pero preservacion gramatical | Afasia semantica vs. agramatismo | GPT-2 genera oraciones gramaticales pero semanticamente vacias |
| Degradacion gradual proporcional a la intensidad | Severidad proporcional al tamano de lesion | Relacion dosis-respuesta confirmada |
| Colapso completo a intensidad extrema | Afasia global | Alpha=20 en LLaVA: tokens sin sentido |

### Descubrimiento 5: La especificidad confirma representaciones separables

Cuando se ablaciona un concepto, los otros conceptos **no se ven afectados** (Delta R1 off-target ~ 0). Esto confirma que los conceptos estan representados en direcciones suficientemente separadas en el espacio de activaciones.

### Descubrimiento 6: Los conceptos visuales concretos son mas separables que los abstractos

Los TCAV scores de LLaVA (conceptos visuales: 0.9-1.0) son consistentemente mas altos que los de GPT-2 (conceptos textuales/abstractos: 0.7-0.83). Los SVMs de LLaVA alcanzan CV accuracy de 0.96-1.0 en las 32 capas. Esto sugiere que las categorias perceptuales concretas forman representaciones mas linealmente separables.

### Descubrimiento 7: La norma del vector SVM decrece con la profundidad

En todos los modelos (GPT-2 y LLaVA), la norma del vector SVM decrece sistematicamente de las capas tempranas a las profundas (ej: LLaVA golden_retriever: 0.89 en L0 → 0.14 en L31). Esto sugiere que las representaciones se **comprimen** a medida que avanzan por la red, concentrandose en menos dimensiones en las capas superiores.

---

## 5. FACTORES A TENER EN CUENTA PARA FUTURAS PRUEBAS

### 5.1 Diseno experimental

| Factor | Recomendacion | Razon |
|--------|---------------|-------|
| **Tamano de muestra de corpus** | Minimo 200 oraciones pos/neg por concepto | Tools (N=100) tuvo problemas de convergencia del SVM; time (N=423) y place (N=770) fueron robustos |
| **Baseline minimo** | Verificar R1 baseline > 50% ANTES de ejecutar ablacion | Si el modelo ya falla en la tarea sin intervencion, no se puede medir la degradacion |
| **Seleccion de conceptos** | Elegir conceptos con alta diversidad visual/semantica | En LLaVA, golden_retriever y labrador son demasiado similares (coseno 0.865) |
| **Numero de capas** | Intervenir en TODAS las capas siempre | Single-layer es demasiado debil; multi-all es 10x mas efectivo |
| **Tipo de vector** | Usar SVM sin normalizar para steering | Los CAVs normalizados son demasiado debiles para producir efectos observables |
| **Cuantizacion** | Considerar el impacto de la cuantizacion 4-bit | En LLaVA, la cuantizacion puede limitar la efectividad de la intervencion |

### 5.2 Metricas

| Metrica | Estado actual | Mejora sugerida |
|---------|--------------|-----------------|
| R1 (accuracy) | Funcional pero con baseline invalido para tools | Redisenar items para garantizar baselines validos |
| R2 (similitud coseno) | Efecto techo (~0.99), insensible | Usar pares con mayor variabilidad semantica, o medir en capas intermedias |
| R2b (disrupcion de salida) | Solo para LLaVA; funcional | Extender a modelos de texto; es mas sensible que R2 |
| Perplejidad | No implementada | Seria la metrica mas sensible y granular para medir degradacion |
| Analisis cualitativo | Manual, no sistematico | Automatizar clasificacion de patrones (perseveracion, parafasias, etc.) |

### 5.3 Modelos

| Modelo | Recomendacion |
|--------|---------------|
| GPT-2 Small | Mantener como baseline principal; resultados completos y solidos |
| Pythia-2.8B | Completar pipeline para validar escalabilidad |
| LLaVA-1.5-7B | Cambiar conceptos y mejorar benchmarks antes de reportar resultados |
| Modelos mas grandes (7B+ texto) | Considerar LLaMA-2-7B o Mistral-7B como extension natural |
| Modelos tipo BERT | Para comparar encoder-only vs decoder-only |

### 5.4 Conceptos

| Tipo | Conceptos probados | Conceptos sugeridos para expansion |
|------|-------------------|-----------------------------------|
| Temporal | time | season, era, moment (mas granulares) |
| Espacial | place | direction, distance, shape |
| Objetos | tools | food, clothing, vehicles |
| Abstractos | (ninguno) | emotion, color, number, justice |
| Visuales concretos | golden_retriever, labrador, tabby_cat | dog vs car vs flower vs building |

---

## 6. RESUMEN EJECUTIVO PARA LA TESIS

### Lo que se logro

1. Se **demostro experimentalmente** que los conceptos semanticos tienen representaciones lineales en transformers (GPT-2 Small), validado con multiples tests estadisticos (TCAV, selectividad, Cohen's d, bootstrap)
2. Se **logro simular un deficit** analogo a la afasia semantica, con una degradacion cuantificable de hasta 33% en precision categorica (R1)
3. Se **confirmo la relacion dosis-respuesta** (H1C) y la especificidad de la ablacion
4. Se **identificaron patrones cualitativos** (perseveracion, parafasias, colapso gradual) que son analogos a los sintomas clinicos de afasia semantica
5. Se **extendio el framework** al dominio multimodal (LLaVA) con resultados preliminares prometedores
6. Se **construyo un pipeline reproducible** con 20+ scripts, 20 tests automatizados, y Dockerfiles

### Lo que no se logro completamente

1. **H1B** (localizacion): La hipotesis de capas medias vs tardias no se confirmo como fue planteada — se descubrio algo diferente (distribucion a traves de todas las capas)
2. **H1D** (hiperonimos vs hiponimos): No fue evaluada experimentalmente
3. **Pythia-2.8B**: Pipeline parcial, sin resultados de ablacion
4. **LLaVA**: Baselines invalidos limitan las conclusiones cuantitativas
5. **ANOVA formal**: Planteado en la metodologia pero no ejecutado

### Fortalezas del proyecto

- Framework experimental **robusto y reproducible**
- Validacion estadistica **rigurosa** (TCAV, selectividad, bootstrap, Cohen's d)
- ~1,500 condiciones experimentales evaluadas
- Combinacion de evidencia **cuantitativa y cualitativa**
- Extension a **dominio multimodal** (texto + vision)
- Resultados que **aportan al campo de interpretabilidad mecanicista**

### Debilidades a reconocer

- Resultados principales solo en GPT-2 Small (124M parametros)
- Baseline de tools invalido
- R2 insensible (efecto techo)
- Falta ANOVA formal
- LLaVA con problemas de baselines y seleccion de conceptos
