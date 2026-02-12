# Informe de Avances: Metodología de Extracción, Validación y Fundamentación Teórica

Este documento presenta una descripción exhaustiva de los avances realizados en la metodología de investigación. Se detalla no solo **qué** se ha hecho, sino **por qué** se ha hecho, fundamentando cada decisión técnica en principios de interpretabilidad mecanicista, neuropsicología cognitiva y rigor estadístico.

## 1. Inspiración y Fundamentos Teóricos: ¿Por qué este enfoque?

### 1.1. De la Correlación a la Causalidad
La Inteligencia Artificial Explicable (XAI) tradicional suele basarse en correlaciones (e.g., "cuando aparece esta palabra, el modelo suele predecir X"). Sin embargo, correlación no implica causalidad. Nuestra investigación busca dar un salto hacia la **Interpretabilidad Mecanística**, cuyo objetivo es realizar una "ingeniería inversa" del modelo.

Nos inspiramos en la **Neuropsicología Cognitiva**, específicamente en la **Afasia Semántica**. En el cerebro humano, lesiones en áreas específicas provocan la pérdida selectiva de conceptos (e.g., un paciente puede olvidar qué es un "animal" pero recordar qué es una "herramienta").
*   **Hipótesis de Isomorfismo Funcional**: Planteamos que si los Modelos de Lenguaje (LLMs) como GPT-2 realmente entienden conceptos, deberían tener estructuras internas análogas a los circuitos semánticos del cerebro.
*   **Objetivo**: Si logramos identificar y "lesionar" (ablar) artificialmente estas estructuras y observamos una degradación conceptual específica (similar a la afasia), habremos demostrado causalmente que el modelo *posee* ese concepto.

## 2. Construcción del Dataset: La Importancia del "Ground Truth"

Para enseñar a una IA a encontrar un concepto, primero necesitamos definir inequívocamente qué es ese concepto. No podemos confiar en datos ruidosos.

### 2.1. Fuente de Conocimiento: ConceptNet 5.7
Utilizamos **ConceptNet**, una base de conocimiento de sentido común, específicamente la relación taxonómica `IsA` (Es un).
*   **¿Por qué?**: ConceptNet ofrece relaciones semánticas curadas por humanos y expertos. A diferencia de extraer texto crudo de internet, esto nos da una "verdad fundamental" (Ground Truth) estructurada.
*   **Conceptos Seleccionados**: `Time` (Tiempo), `Place` (Lugar) y `Tools` (Herramientas). Se eligieron por ser categorías ontológicamente distintas y fundamentales en la cognición humana.

### 2.2. Curaduría y Limpieza Rigurosa (El proceso de "Blacklist")
Uno de los mayores desafíos en XAI es el ruido. Si entrenamos un clasificador con ejemplos malos, obtendremos un vector sin sentido ("Garbage In, Garbage Out").
*   **El Problema**: ConceptNet contiene ruido. Por ejemplo, para "Time", podría incluir "light" (luz) porque viaja en el tiempo, o términos polisémicos como "Second" (que puede ser tiempo o posición).
*   **Nuestra Solución**: Implementamos un sistema de **Blacklist (Lista Negra) manual** en `config.py`. Revisamos cada término candidato y eliminamos:
    *   Términos polisémicos (e.g., "watch" puede ser mirar o reloj).
    *   Asociaciones débiles o metafóricas.
    *   Errores de la base de datos.
    *   **Resultado**: Un dataset de entrenamiento de alta "pureza semántica", asegurando que el vector extraído represente *exclusivamente* el concepto objetivo.

## 3. Metodología de Extracción de CAVs: El Enfoque de Triangulación

Un **Concept Activation Vector (CAV)** es una dirección en el espacio matemático del modelo que representa un concepto. Para asegurar que encontramos la dirección correcta, no confiamos en un solo algoritmo. Usamos un enfoque **Dual**.

### 3.1. Método 1: Máquina de Soporte Vectorial (Linear SVM)
*   **¿Qué hace?**: Busca el "hiperplano óptimo" que separa mejor las activaciones de los ejemplos del concepto (Positivos) de los ejemplos aleatorios (Negativos). El CAV es el vector perpendicular a este plano.
*   **Inspiración**: Teoría de Aprendizaje Estadístico (Vapnik). Si las representaciones del modelo son linealmente separables, significa que el modelo ha "desenredado" (disentangled) ese concepto en esa capa.
*   **Implementación**: Usamos `LinearSVC` con búsqueda de hiperparámetros (Grid Search) para el parámetro de regularización $C$. Esto evita el sobreajuste y asegura que el vector generalice.

### 3.2. Método 2: Diferencia de Medias (Mean Difference)
*   **¿Qué hace?**: Calcula el promedio de todas las activaciones del concepto y le resta el promedio de las activaciones neutras.
*   **¿Por qué usarlo si ya tenemos SVM?**: El SVM es poderoso pero a veces puede ser engañado por "outliers" (ejemplos extremos) que definen el margen. La Diferencia de Medias es estadísticamente más robusta y estable, ya que representa el "centro de masa" del concepto.
*   **Criterio de Consistencia**: Solo confiamos en un CAV si el vector obtenido por SVM es muy similar (Similitud Coseno alta) al vector obtenido por Medias. Esto se llama **Triangulación Metodológica**.

### 3.3. Detalles Técnicos de Implementación

Para garantizar la reproducibilidad y rigor técnico, el proceso de extracción sigue estos pasos exactos implementados en `cav_extraction.py`:

#### A. Recolección de Activaciones (Forward Hooks)
Para cada frase del dataset (positiva o negativa), pasamos la entrada por el modelo `gpt2-small`.
*   **Punto de Extracción**: Utilizamos el hook `blocks.{layer}.hook_resid_post`, que captura el estado residual del stream justo después bloque del Transformer en la capa $L$ y antes de entrar a la siguiente.
*   **Estrategia de Tokenización y Pooling**: Dado que una frase tiene múltiples tokens (e.g., "Time is money" -> [Time, is, money]), obtenemos una matriz de activaciones $[N_{tokens}, d_{model}]$.
    *   *Decisión*: Usamos **Mean Pooling** (promedio de todos los tokens de la secuencia).
    *   *Justificación*: A diferencia de usar solo el último token (que en GPT-2 suele acumular información), el promedio global ha demostrado capturar mejor la semántica distribuida de toda la oración en tareas de clasificación lineal (Reimers & Gurevych, 2019), reduciendo el ruido posicional.

#### B. Algoritmo SVM Lineal (Detalle)
El cálculo del CAV mediante SVM no es un ajuste simple. Seguimos un protocolo estricto para asegurar que el hiperplano sea óptimo:
1.  **Entrenamiento**: Usamos `LinearSVC` de Scikit-Learn.
2.  **Validación Cruzada**: Aplicamos **Stratified K-Fold** con $K=5$ particiones. Esto asegura que cada pliegue tenga la misma proporción de ejemplos positivos y negativos.
3.  **Búsqueda de Hiperparámetros (Grid Search)**: Buscamos el valor óptimo de regularización $C$ en el rango `[0.01, 0.1, 1.0, 10.0]`.
    *   *Por qué*: Un $C$ mal ajustado puede llevar a sobreajuste (el vector "memoriza" ejemplos difíciles) o subajuste (el vector es trivial).
4.  **Extracción del Vector**: El CAV es el vector de coeficientes $w$ del mejor modelo, normalizado a norma unitaria: $v_{CAV} = \frac{w}{||w||_2}$.

#### C. Diagnóstico de Capas (Layer Probing)
Antes de elegir las capas 6, 9 y 10, ejecutamos un diagnóstico automatizado `run_layer_diagnostic`:
*   Entrenamos sondas lineales (`LogisticRegression`) en **todas las 12 capas** del modelo.
*   Medimos la precisión (Accuracy) para distinguir el concepto "Time" vs "Random".
*   *Resultado*: Las capas intermedias (3-5) mostraron la mayor precisión bruta (~73%), pero las capas tardías (9-10) mantuvieron una precisión alta (~62%) con mayor abstracción. Seleccionamos un rango medio-tardío para capturar tanto características semánticas como funciones de alto nivel.

### 3.4. Selección de Capas: "Layer Probing"

## 4. Validación Estadística: Certificando la Calidad Científica

Extraer un vector es fácil; probar que significa algo es difícil. Hemos implementado una batería de tests inspirada en los estándares más altos de la literatura de XAI (Kim et al., Hewitt & Liang).

### 4.1. Test de Permutación TCAV (Significancia)
*   **La Pregunta**: "¿Es este vector realmente mejor clasificando el concepto que un vector aleatorio?"
*   **El Test**: Revolvemos las etiquetas de los datos (hacemos que "tiempo" sea "lugar" y viceversa) miles de veces y extraemos vectores "falsos".
*   **Criterio**: Si nuestro CAV real no es mejor que el 95% de los vectores falsos (p-value < 0.05), lo descartamos. Esto descarta resultados que sean fruto del azar.

### 4.2. Test de Selectividad (Control de Memorización)
*   **La Pregunta**: "¿El modelo entiende el concepto o solo memorizó las palabras del entrenamiento?"
*   **Inspiración**: Hewitt & Liang (2019) demostraron que los probes pueden aprender a memorizar datos superficiales.
*   **El Test**: Comparamos la precisión del CAV con etiquetas reales vs. etiquetas de control (Control Tasks). La diferencia se llama **Selectividad**. Una alta selectividad prueba que el modelo está usando la estructura semántica interna y no solo patrones superficiales.

### 4.3. Estabilidad Bootstrap (Intervalos de Confianza)
*   **La Pregunta**: "¿Si cambiamos un poco los datos de entrada, el vector apunta a otro lado?"
*   **El Test**: Usamos la técnica de Bootstrap (remuestreo con reemplazo) para generar cientos de versiones del dataset y recalculamos el CAV.
*   **Resultado**: Construimos Intervalos de Confianza (BCa). Si el intervalo es estrecho y la similitud coseno entre todas las versiones es alta (>0.9), confirmamos que el concepto es **Estable**.

## 6. Análisis de Limitaciones y Pruebas Fallidas

En aras de la transparencia científica, es crucial reportar no solo los éxitos sino también las pruebas que **no superaron** los umbrales de rigor establecidos. Estas fallas proporcionan información valiosa sobre los límites del modelo y la complejidad de los conceptos.

### 6.1. Fallo de Convergencia SVM en el Concepto de Control ("Tools")
*   **Observación**: El clasificador SVM para el concepto `Tools` (Herramientas) **no convergió** en ninguna de las capas analizadas (6, 9, 10), incluso tras aumentar las iteraciones y ajustar la regularización.
*   **Datos**: La desviación estándar de la precisión (`cv_accuracy_std`) fue muy alta (~0.09 vs ~0.01 en Time/Place), indicando inestabilidad.
*   **Interpretación Teórica**: Esto sugiere que "Herramientas" es una categoría semántica mucho más difusa y menos linealmente separable en el espacio de activación de GPT-2 que "Tiempo" o "Lugar". Mientras que el tiempo y el lugar tienen marcadores sintácticos y semánticos claros (preposiciones, adverbios), las herramientas se definen funcionalmente, lo cual es más difícil de capturar por un modelo de lenguaje puro sin grounding visual/sensorial.
*   **Acción Correctiva**: Se mantiene `Tools` como un "concepto de control negativo" (baseline), pero se interpreta con cautela su vector, reconociendo que es menos preciso.

### 6.2. Entrelazamiento Conceptual (Time vs Place)
*   **Observación**: La similitud coseno entre los CAVs de `Time` y `Place` superó el umbral de independencia deseado ($<0.3$) en las capas media-tardías.
    *   Capa 9: Similitud `0.538`.
    *   Capa 10: Similitud `0.542`.
*   **Interpretación**: Existe un solapamiento significativo. Esto es consistente con la teoría de la metáfora conceptual (Lakoff & Johnson), donde el tiempo se conceptualiza a often en términos espaciales (e.g., "long time", "ahead of us"). El modelo parece haber aprendido esta correlación estructural.
*   **Riesgo**: Al intervenir sobre "Tiempo", podríamos estar afectando accidentalmente representaciones de "Lugar".
*   **Mitigación**: Se utilizará el método de **Proyección Ortogonal** (Projection) en la fase experimental para intentar aislar el componente exclusivo de tiempo, eliminando la dirección compartida con el espacio.

### 6.3. Sensibilidad del Método "Mean Difference" a Outliers
*   **Observación**: En `layer_6` para `Time`, el SVM logró una precisión del 88%, pero la similitud con Mean Difference fue solo moderada (0.58).
*   **Análisis**: Mean Difference asume que las distribuciones son esféricas y unimodales. La discrepancia sugiere que la distribución de activaciones de "Time" podría ser multimodal (tener varios "clusters" de significado, e.g., tiempo cronológico vs tiempo gramatical).
*   **Decisión**: Priorizar el CAV obtenido por SVM para las intervenciones, ya que su margen de decisión es más robusto ante distribuciones complejas.

### 6.4. Fallo Crítico en Significancia Estadística (Mean Difference vs SVM)
*   **Resultados de la Prueba de Permutación TCAV**:
    *   **Método SVM**: P-values consistentemente < 0.05 (Significativo). El vector SVM es distinguible del azar.
    *   **Método Mean Difference**: P-values > 0.4 (NO Significativo).
*   **Resultados de Estabilidad Bootstrap**:
    *   **Método SVM**: "Moderadamente estable" (Similitud coseno > 0.8 en re-muestreos).
    *   **Método Mean Difference**: "Inestable/Unreliable" (Intervalos de confianza amplios).
*   **Conclusión Metodológica**: Aunque el método de Diferencia de Medias es intuitivo, **no pasó las pruebas de robustez estadística**. Las direcciones varían demasiado dependiendo de los datos exactos y no son estadísticamente mejores que direcciones aleatorias en este espacio de alta dimensionalidad.
*   **Impacto**: Se descarta el uso de vectores "Mean Difference" para la fase final de intervención (Aphasia). **Nos basaremos exclusivamente en vectores SVM validos y triangulados**.

## 7. Métricas de Evaluación de la Intervención (Afasia Inducida)

Para medir el éxito de la "lobotomía conceptual" (intervención), no basta con ver si el modelo falla. Debemos cuantificar *cuánto* daño semántico específico hemos causado. Definimos la siguiente métrica principal:

### 7.1. $\Delta sim$ (Caída de Similitud Semántica)
Esta métrica evalúa la capacidad del modelo para reconocer relaciones semánticas cercanas tras la intervención (Benchmark R2). Una alta degradación sugiere una pérdida de la comprensión del "vecindario" semántico del concepto, análoga a la afasia donde el paciente pierde las conexiones entre palabras relacionadas.

*   **Definición**: Es la diferencia promedio en la similitud coseno entre los embeddings de pares de conceptos relacionados, antes y después de la intervención.
    *   $sim\_cos\_base$: Similitud coseno de los vectores de activación del modelo base (sin modificar).
    *   $sim\_cos\_int$: Similitud coseno de los vectores del modelo intervenido (ablado).
*   **Justificación**: Utilizamos la similitud coseno porque es el estándar fundamental en Procesamiento del Lenguaje Natural (NLP) para medir proximidad semántica en espacios vectoriales. Si el modelo "olvida" el concepto, los vectores de palabras relacionadas (e.g., "hora" y "minuto") deberían volverse ortogonales o aleatorios entre sí.
*   **Fórmula**: 
    $$ \Delta sim = \text{promedio}(sim\_cos\_base - sim\_cos\_int) $$
*   **Interpretación**:
    *   $\Delta sim \approx 0$: La intervención no tuvo efecto (el concepto sigue intacto).
    *   $\Delta sim > 0$ (Alto): Éxito. El modelo ya no puede relacionar los términos semánticamente cercanos. Se ha inducido una "afasia associativa".
