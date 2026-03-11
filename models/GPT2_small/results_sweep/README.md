# Resultados del Experimento Sweep Comprehensivo

Sweep sistematico de ablacion conceptual inspirado en Golden Gate Claude (Anthropic, 2024) y CAA (Turner et al., 2023). Varia tipo de vector, alpha, configuracion de capas y concepto.

**Estado: Diseno completo, pendiente de ejecucion.** Los resultados principales del proyecto se obtuvieron via CCT test y benchmarks BEA, por lo que este sweep es material suplementario opcional.

## Diseno experimental

- **Vectores**: SVM (`clf.coef_[0]`) y Mean-diff (`mean(pos) - mean(neg)`), ambos sin normalizar
- **Alphas**: 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0
- **Configuraciones de capas**: 12 singles + 5 pares + 3 triples + 3 multi = 23
- **Conceptos**: time, place, tools
- **Direccion**: ablacion (restar) + amplificacion (sumar, control)
- **Total**: 1,242 ablacion + 54 amplificacion = **1,296 condiciones**

## Archivos esperados (tras ejecucion)

| Archivo | Descripcion |
|---------|-------------|
| `results_sweep.csv` | 1,296 filas con R1/R2 target + cross-concept por condicion |
| `baselines.json` | R1/R2 baseline por concepto |
| `sweep_report.json` | Resumen: sweet spots, mejores condiciones, efectos por factor |

## Generado por

[`experiment_sweep.py`](../experiment_sweep.py)
