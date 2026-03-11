# Vectores de Steering Mean Difference (sin normalizar)

Vectores de steering calculados como la diferencia de medias entre activaciones positivas y negativas, **sin normalizar**. Misma estructura que [`cavs_steering/`](../cavs_steering/) pero usando el metodo Mean Difference en lugar de SVM.

## Metodo

```
steering_vector = mean(activaciones_positivas) - mean(activaciones_negativas)
```

Este es el metodo CAA (Contrastive Activation Addition, Turner et al. 2023) puro.

## Contenido (108 archivos .pt)

| Patron | Descripcion |
|--------|-------------|
| `{concepto}_steer_layer{L}.pt` | Vector mean-diff crudo |
| `{concepto}_steer_centered_layer{L}.pt` | Vector mean-diff con activaciones centradas |
| `{concepto}_mean_layer{L}.pt` | Media global de activaciones |

- Conceptos: `time`, `place`, `tools`
- Capas: 0-11 (12 capas)

## Rol en el experimento

Sirve como **comparacion directa** contra los vectores SVM en `cavs_steering/`. Los resultados muestran que mean-diff produce efectos de ablacion mas fuertes (deltaR1 hasta +0.70) pero sin la validacion estadistica del SVM (sin CV accuracy ni Cohen's d). Esta comparacion es un hallazgo del experimento: la fuerza bruta de mean-diff vs el rigor de SVM.

Ver [`results_multilayer_meandiff/`](../results_multilayer_meandiff/) para los resultados experimentales.

## Generado por

[`experiment_multilayer.py`](../experiment_multilayer.py) (variante mean-diff)
