# Resultados del Experimento Multi-Capa (SVM Steering)

Resultados del experimento de ablacion multi-capa usando vectores SVM sin normalizar. **Este es el mejor enfoque actual**, con ΔR1 hasta +0.33 (vs +0.07 del experimento original).

## Diseno experimental

- **Vectores**: SVM `clf.coef_[0]` sin normalizar (escala natural)
- **Modos de intervencion**:
  - `single_L3`: solo capa 3
  - `multi_early`: capas 3, 4, 5
  - `multi_all`: capas 0-11 (todas)
- **Alphas**: 1.0, 4.0
- **Centering**: con/sin mean-centering
- **Direccion**: ablacion (restar) + amplificacion (sumar, como control)
- **Total**: 36 condiciones de ablacion + 3 controles de amplificacion = 39

## Archivos

| Archivo | Descripcion |
|---------|-------------|
| `results_factorial.csv` | 39 filas: R1, R2, ΔR1, ΔR2 por condicion |
| `results_baseline.json` | R1/R2 base por concepto |
| `extraction_report.json` | Metricas SVM por capa: CV accuracy, Cohen's d, normas, convergencia |

## Resultados principales

### Mejor ablacion por concepto

| Concepto | Condicion | ΔR1 |
|----------|-----------|-----|
| Place | multi_all / alpha=4.0 / centered | **+0.333** |
| Time | multi_all / alpha=4.0 / raw | **+0.133** |
| Tools | multi_early / alpha=4.0 / centered | **+0.133** |

### Efecto de cada factor

| Factor | Valor | ΔR1 medio |
|--------|-------|-----------|
| Modo | single_L3 | +0.017 |
| | multi_early | +0.056 |
| | **multi_all** | **+0.139** |
| Alpha | 1.0 | +0.033 |
| | **4.0** | **+0.109** |

### Controles de amplificacion (validacion)

| Concepto | ΔR1 | Valido |
|----------|-----|--------|
| Time | -0.333 | Si (R1 sube) |
| Place | -0.100 | Si (R1 sube) |
| Tools | -0.767 | Si (R1 sube) |

Los tres controles validan que los vectores SVM capturan la direccion correcta del concepto.

## Generado por

[`experiment_multilayer.py`](../experiment_multilayer.py)

## Comparacion con experimento original

| Metrica | Original (single-layer, normalizado) | Multi-capa SVM |
|---------|--------------------------------------|----------------|
| Mejor ΔR1 | +0.07 | **+0.333** |
| Factor clave | alpha | modo (multi-capa) |
| Validacion estadistica | Si (CV, Cohen's d) | Si (CV 80-91%, d 3.1-4.9) |
