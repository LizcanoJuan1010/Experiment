# Resultados del Experimento Sweep Comprehensivo

Sweep sistematico de ablacion conceptual inspirado en Golden Gate Claude (Anthropic, 2024) y CAA (Turner et al., 2023). Varia tipo de vector, alpha, configuracion de capas y concepto para producir curvas dosis-respuesta e identificar sweet spots.

## Diseno experimental

- **Vectores**: SVM (`clf.coef_[0]`) y Mean-diff (`mean(pos) - mean(neg)`), ambos sin normalizar
- **Alphas**: 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0
- **Configuraciones de capas**: 12 singles + 5 pares + 3 triples + 3 multi = 23
- **Conceptos**: time, place, tools
- **Direccion**: ablacion (restar) + amplificacion (sumar, control)
- **Total**: 1,242 ablacion + 54 amplificacion = **1,296 condiciones**
- **Especificidad cruzada**: cada condicion mide R1/R2 para los 3 conceptos

## Archivos

| Archivo | Descripcion |
|---------|-------------|
| `results_sweep.csv` | 1,296 filas con R1/R2 target + cross-concept por condicion |
| `baselines.json` | R1/R2 baseline por concepto (sin intervencion) |
| `sweep_report.json` | Resumen: sweet spots, mejores condiciones, efectos por factor |

## Figuras (`figures/`)

| Figura | Descripcion |
|--------|-------------|
| F01 | Heatmap ΔR1 por (layer_config x alpha), por concepto y tipo de vector |
| F02 | Curva dosis-respuesta para las mejores configuraciones |
| F03 | Curva dosis-respuesta: singles vs pares vs triples vs multi |
| F04 | Comparacion top-5: SVM vs Mean-diff |
| F05 | Especificidad: ΔR1 target vs dano cruzado |
| F06 | Heatmap por capa individual (12 capas x 9 alphas) |
| F07 | Ablacion vs amplificacion (validacion de direccion) |
| F08 | Sweet spots: ΔR1 target vs dano cruzado |

## Generado por

[`experiment_sweep.py`](../experiment_sweep.py)
