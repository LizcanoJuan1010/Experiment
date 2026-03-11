# Figuras — GPT-2 Small (30 Conceptos)

Todas las figuras usan el pipeline de extracción CAV con 30 conceptos.
Solo formato `.png`. Scripts en `models/GPT2_small/`.

---

## Análisis Geométrico de CAVs  (`generate_cav_figures.py`)

Sin necesidad de correr el modelo — solo requiere los `.pt` en `cavs/`.

### fig21 — Espacio de Activaciones t-SNE
t-SNE (2D) de los ejemplos positivos de los 30 conceptos en cada capa (6, 9, 10).
Puntos coloreados por categoría semántica (Animal / Objeto / Profesión).
Líneas finas = convex hull por concepto; líneas gruesas discontinuas = convex hull por categoría.
Muestra si el modelo separa categorías en el espacio de representación y cómo evoluciona esa estructura capa a capa.

### fig22 — Red de Similitud Conceptual
Grafo de fuerza dirigida: nodos = 30 conceptos, aristas = pares con coseno SVM > 0.20.
Grosor de arista proporcional a la similitud. Un arista inesperada (e.g., baker–chef) tiene
significado semántico: el modelo codifica ambos en direcciones similares.

### fig23 — Heatmap de Emergencia por Capa
Dos heatmaps 30×3 (conceptos × capas):
- **Izquierda**: Exactitud validación cruzada SVM — qué tan separable es cada concepto.
- **Derecha**: d de Cohen (effect size) — magnitud de la separación positivos/negativos.
Conceptos agrupados por categoría. Identifica en qué capa se codifica mejor cada concepto.

### fig24 — Matriz de Interferencia entre CAVs
Matriz asimétrica 30×30 en la capa 9. Entrada [i,j] = proyección media de los ejemplos
positivos del concepto j sobre CAV_i, normalizada por z-score.
Valores altos fuera de la diagonal = *semantic bleed*: el CAV_i responde a ejemplos del
concepto j, indicando subespacio representacional compartido.

### fig25 — Biplot PCA de Direcciones CAV
PCA (2D) de los 30 vectores CAV unitarios mostrados como flechas desde el origen, por capa.
Conceptos con ángulos similares usan direcciones parecidas en el modelo; flechas ortogonales
indican independencia. CP1 y CP2 con su % de varianza explicada.

---

## Resultados Experimentales — CCT  (`generate_cav_figures.py`)

### fig26 — Dosis-Respuesta y Especificidad (CCT)
Dos paneles:
- **Izquierda**: Curvas dosis-respuesta (Δ exactitud vs α) on-target, por método y capa.
  Muestra cómo aumenta el efecto de ablación al incrementar la intensidad.
- **Derecha**: Especificidad en α=6 — barras comparando el efecto on-target vs off-target.
  El SVM muestra efecto off-target muy bajo (≈0.07) frente a Diferencia de Medias (≈0.5):
  el SVM es más preciso en eliminar solo el concepto objetivo.

### fig27 — Distribuciones de Proyección CAV
Para cada uno de los 30 conceptos en la capa 9 (SVM): violines KDE de la proyección
de los ejemplos positivos (azul) vs negativos (naranja) sobre el CAV.
Ordenados por d de Cohen descendente. Muestra visualmente la separación que cada CAV captura.

### fig28 — Comparación de Impacto por Configuración de Capas
Responde: ¿es mejor intervenir en una sola capa o en todas a la vez?
- **Heatmaps superiores** (uno por método): Δ exactitud por concepto × configuración de capa
  (Capa 6 / Capa 9 / Capa 10 / Todas las capas), en α=6.
- **Barras inferiores**: media por categoría semántica en cada configuración.
  Permite identificar si "todas las capas" supera consistentemente a capas individuales,
  y si alguna categoría (animales/objetos/profesiones) responde diferente al steering.

---

## Tests BEA  (`bea_anova_analysis.py`)

### bea_tests/naming_interaction_method_layer
Nombramiento (α=3.0): Δ exactitud media por capa. Interacción Método × Capa.

### bea_tests/naming_interaction_method_alpha
Nombramiento (Capa=6): Δ exactitud media por alfa. Curva dosis-respuesta.

### bea_tests/oddoneout_interaction_method_layer
Elemento Diferente (α=3.0): Δ exactitud media por capa.

### bea_tests/oddoneout_interaction_method_alpha
Elemento Diferente (Capa=6): Δ exactitud media por alfa.

### bea_tests/synonym_interaction_method_layer
Sinónimos (α=3.0): Δ AUC media por capa.

### bea_tests/synonym_interaction_method_alpha
Sinónimos (Capa=6): Δ AUC media por alfa.

---

## Regenerar figuras

```bash
cd models/GPT2_small

# fig21–fig28 (sin modelo, solo cavs/*.pt y results/*.json):
python generate_cav_figures.py

# Figuras BEA (interacciones ANOVA):
python bea_anova_analysis.py

# fig15–fig20 (requiere GPT-2 Small vía TransformerLens):
python generate_figures_deep.py
```
