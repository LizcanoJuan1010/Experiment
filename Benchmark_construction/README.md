# Construccion de Benchmarks R1 y R2

Scripts para construir y validar los benchmarks de evaluacion que miden el impacto de la ablacion conceptual.

## Benchmarks

### R1 — Clasificacion por Hiperonimo

Test de opcion multiple que evalua si el modelo puede clasificar correctamente una palabra en su categoria superordinada.

Formato: *"X is a type of ___"* con 4 opciones (1 correcta + 3 distractores).

- 30 items por concepto (time, place, tools)
- Metrica: accuracy (% de respuestas correctas basado en loss minima)

### R2 — Similitud Semantica

Test de coherencia semantica que mide si el modelo preserva relaciones de similitud entre oraciones.

- 25 pares de parafrasis por concepto
- Metrica: similitud coseno promedio en capa 11

## Archivos

### Scripts

| Archivo | Descripcion |
|---------|-------------|
| `config.py` | Configuracion: conceptos, etiquetas de hiperonimos, plantillas, distractores |
| `build_r1.py` | Genera el benchmark R1 (MCQ) a partir de nodos ConceptNet |
| `build_r2.py` | Genera el benchmark R2 (pares de parafrasis) |
| `validate_benchmarks.py` | Validacion automatica de calidad |
| `__init__.py` | Marcador de paquete Python |

### Output (`output/`)

| Archivo | Descripcion |
|---------|-------------|
| `r1_benchmark.json` | Benchmark R1 generado |
| `r2_benchmark.json` | Benchmark R2 generado |
| `provenance_report.json` | Trazabilidad de datos (origen de cada item) |

## Usado por

[`metrics.py`](../metrics.py) — Carga los benchmarks para evaluar R1 y R2 durante los experimentos.

## Fundamentacion

- Basado en la teoria de categorizacion de Rosch (1978)
- Nodos semanticos de ConceptNet 5.7 (Speer et al. 2017)
