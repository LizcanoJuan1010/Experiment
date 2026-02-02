# Análisis de Robustez de Extracción de CAVs

> **Fuente de los datos:** La información analizada proviene del archivo `cavs/extraction_report.json` generado durante la ejecución de `cav_extraction.py`. No proviene de los archivos CSV de resultados (`results_factorial` o `results_specificity`).

## 0. Fundamentación Metodológica
La metodología se sustenta en el marco TCAV (Testing with Concept Activation Vectors) propuesto por Kim et al. (2018), quienes demostraron la existencia de representaciones lineales de conceptos en espacios latentes. Complementariamente, nos apoyamos en los hallazgos de Bau et al. (2020) sobre la emergencia espontánea de detectores semánticos en unidades individuales. Para la selección taxonómica de los conceptos (Nivel Superordinado), se siguió la teoría de categorización cognitiva de Rosch (1978), priorizando la abstracción funcional sobre la similitud perceptual.

## 1. ¿Fue correcta y funcional? (Funcionalidad)
**VEREDICTO: SÍ, EXCELENTE.**

La métrica principal para saber si el concepto "existe" y fue capturado correctamente es el **SVM Cross-Validation Accuracy** (Precisión de validación cruzada). Si este número es alto, significa que la red distingue perfectamente el concepto.

*   **Time**: ~99% de precisión (Casi perfecto).
*   **Place**: ~98% de precisión.
*   **Tools**: ~94-96% de precisión.

**Conclusión:** Los conceptos son linealmente separables dentro de GPT-2. La extracción es funcional y los vectores son altamente predictivos.

## 2. ¿Es robusto? (Fiabilidad)
Aquí hay matices importantes que debes conocer. La robustez se mide de dos formas:

### A. Consistencia entre Métodos (Mean Diff vs SVM)
¿Apuntan los dos métodos a la misma dirección?
*   **Valores obtenidos**: 0.51 - 0.72 (Similitud Coseno).
*   **Interpretación**: Es una **robustez moderada**.
    *   Un valor de 1.0 sería ideal.
    *   Valores de 0.5-0.7 indican que aunque ambos métodos "funcionan", no están encontrando exactamente el mismo vector. Probablemente el método SVM es más preciso (dada su alta accuracy), mientras que "Mean Difference" es una aproximación más burda.
    *   *Recomendación:* Confía más en los vectores generados por **SVM** (`_svm_layerX.pt`).

### B. Independencia de Conceptos (Entanglement)
¿Se confunden los conceptos entre sí? (Queremos que sea cercano a 0).
*   **Time vs Place**: **0.47** (Media) / **0.30** (SVM).
    *   ⚠️ **Alerta**: Hay cierta superposición. La red relaciona "Tiempo" y "Lugar" (posiblemente porque ambos operan como complementos circunstanciales gramaticales). No son totalmente ortogonales.
*   **Time vs Tools**: ~0.1 - 0.2 (Muy bien, son distintos).
*   **Place vs Tools**: ~0.2 - 0.3 (Bien).

## Resumen Ejecutivo

| Métrica | Resultado | Calidad | Nota |
| :--- | :--- | :--- | :--- |
| **Separabilidad (Accuracy)** | > 94% | 🟢 Excelente | La red entiende los conceptos perfectamente. |
| **Consistencia de Método** | 0.5 - 0.7 | 🟡 Moderada | Usa los vectores SVM, son más precisos. |
| **Independencia (Time/Place)** | 0.3 - 0.47 | 🟠 Cuidado | Existe correlación entre Tiempo y Lugar. |

**Recomendación final:** La extracción fue exitosa. Para tus experimentos posteriores, te recomiendo usar **principalmente los vectores SVM**, ya que mostraron una mejor separación (menor entanglement) y una precisión casi perfecta.
