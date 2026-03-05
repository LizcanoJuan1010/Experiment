# Resultados del Experimento Multi-Capa (Mean Difference)

Resultados del experimento multi-capa usando vectores de diferencia de medias sin normalizar. Sirve como **comparacion** contra el enfoque SVM en [`results_multilayer/`](../results_multilayer/).

## Diseno experimental

Mismo diseno factorial que el experimento SVM (39 condiciones), pero usando `mean(pos) - mean(neg)` en lugar de `clf.coef_[0]`.

## Archivos

| Archivo | Descripcion |
|---------|-------------|
| `results_factorial.csv` | 39 filas: R1, R2, ΔR1, ΔR2 por condicion |
| `results_baseline.json` | R1/R2 base por concepto |
| `extraction_report.json` | Normas de vectores y estadisticas de extraccion |
| `aphasia_test_results.json` | Test cualitativo de generacion de texto |

## Resultados principales

- **Mejor ΔR1**: +0.70 (TIME, multi_all/alpha=4.0) — mas fuerte que SVM (+0.33)
- **Centering**: Sin efecto (matematicamente identico para mean-diff)
- **Amplificacion**: Los 3 controles validaron

## Comparacion SVM vs Mean Diff

| Metrica | Mean Diff | SVM |
|---------|-----------|-----|
| Mejor ΔR1 | **+0.70** | +0.33 |
| Validacion CV | No disponible | 80-91% |
| Cohen's d | No disponible | 3.1-4.9 |
| Rigor estadistico | Bajo | **Alto** |

Mean-diff produce efectos mas fuertes pero sin metricas de validacion. SVM es el enfoque recomendado para la tesis.

## Generado por

- [`experiment_multilayer.py`](../experiment_multilayer.py) (variante mean-diff)
- [`aphasia_test_steering.py`](../aphasia_test_steering.py) (test cualitativo)
