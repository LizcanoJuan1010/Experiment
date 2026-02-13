# Vectores de Steering SVM (sin normalizar)

Contiene vectores de steering basados en SVM para feature steering multi-capa. A diferencia de los CAVs en `cavs/`, estos vectores **NO estan normalizados** — conservan la escala natural del coeficiente SVM (`clf.coef_[0]`), lo cual es esencial para que la ablacion tenga efecto proporcional a la separabilidad real del concepto.

## Fundamento

Siguiendo CAA (Turner et al. 2023), la ablacion conceptual se implementa como:

```
act' = act - alpha * steering_vector
```

Con vectores normalizados (norma=1), alpha necesitaria ser ~100-200 para afectar activaciones de norma ~225. Con vectores crudos (norma natural ~1-4), alpha=1-4 ya produce efectos significativos.

## Contenido (108 archivos .pt)

| Patron | Descripcion |
|--------|-------------|
| `{concepto}_steer_layer{L}.pt` | Vector SVM crudo (sin normalizar) |
| `{concepto}_steer_centered_layer{L}.pt` | Vector SVM entrenado sobre activaciones centradas (mean-centering, RepE) |
| `{concepto}_mean_layer{L}.pt` | Media global de activaciones (necesaria para centering en inference) |

- Conceptos: `time`, `place`, `tools`
- Capas: 0-11 (las 12 capas de GPT-2-small)
- Total: 3 conceptos x 3 tipos x 12 capas = 108 archivos

## Metricas de calidad

| Concepto | CV Accuracy | Cohen's d | Norma SVM | Convergencia |
|----------|-------------|-----------|-----------|--------------|
| Time | 81-91% | 3.4-4.3 (large) | 1.0-4.4 | Si |
| Place | 81-90% | 3.1-3.9 (large) | 1.3-4.5 | Si |
| Tools | 72-82% | 4.1-4.9 (large) | 0.8-3.6 | No (N=100) |

Detalle completo en [`results_multilayer/extraction_report.json`](../results_multilayer/extraction_report.json).

## Generado por

[`experiment_multilayer.py`](../experiment_multilayer.py)

## Referencias

- Kim et al. 2018 — TCAV: SVM-based CAV extraction
- Turner et al. 2023 — CAA: vectores sin normalizar + multi-capa
- Zou et al. 2023 — RepE: mean-centering
