# Construccion de Benchmarks

Scripts para construir y validar los benchmarks de evaluacion que miden el impacto de la ablacion conceptual.

## Benchmarks

### R1 — Clasificacion por Hiperonimo

Test de opcion multiple que evalua si el modelo puede clasificar correctamente una palabra en su categoria superordinada.

Formato: *"X is a type of ___"* con 4 opciones (1 correcta + 3 distractores).

- 30 items por concepto
- Metrica: accuracy (% de respuestas correctas basado en loss minima)

### R2 — Similitud Semantica

Test de coherencia semantica que mide si el modelo preserva relaciones de similitud entre oraciones.

- 25 pares de parafrasis por concepto
- Metrica: similitud coseno promedio en capa 11

### BEA Association

MCQ de asociacion semantica construido automaticamente usando ConceptNet 5.5, WordNet y nodos curados. Distractores de otros conceptos, filtrados para evitar solapamiento semantico.

### BEA Naming

Confrontation naming con distractores intra-categoria. Evalua la capacidad del modelo de nombrar correctamente un concepto dado su descripcion.

### BEA Odd-One-Out

Deteccion de intruso semantico: 3 items relacionados con el concepto + 1 distractor. El modelo debe identificar el item que no pertenece.

## Archivos

### Scripts

| Archivo | Descripcion |
|---------|-------------|
| `config.py` | Configuracion: conceptos, etiquetas, plantillas, distractores |
| `build_r1.py` | Genera el benchmark R1 (MCQ) |
| `build_r2.py` | Genera el benchmark R2 (pares de parafrasis) |
| `build_bea_association.py` | Genera el benchmark BEA association |
| `build_bea_naming.py` | Genera el benchmark BEA naming |
| `build_bea_oddoneout.py` | Genera el benchmark BEA odd-one-out |
| `validate_benchmarks.py` | Validacion automatica de calidad |

### Output (`output/`)

| Archivo | Descripcion |
|---------|-------------|
| `r1_benchmark.json` | Benchmark R1 generado |
| `r2_benchmark.json` | Benchmark R2 generado |
| `bea_association_benchmark.json` | Benchmark BEA association (728 items, 30 conceptos) |
| `bea_naming_benchmark.json` | Benchmark BEA naming |
| `bea_oddoneout_benchmark.json` | Benchmark BEA odd-one-out |
| `provenance_report.json` | Trazabilidad de datos (origen de cada item) |

## Usado por

- [`metrics.py`](../metrics.py) — Carga los benchmarks para evaluar R1 y R2
- [`test/cct_test.py`](../test/cct_test.py) — Usa BEA association para el CCT
- [`test/bea_naming_test.py`](../test/bea_naming_test.py) — Usa BEA naming
- [`test/bea_oddoneout_test.py`](../test/bea_oddoneout_test.py) — Usa BEA odd-one-out
- [`test/bea_synonym_test.py`](../test/bea_synonym_test.py) — Usa datos de conceptos

## Fundamentacion

- Basado en la teoria de categorizacion de Rosch (1978)
- Nodos semanticos de ConceptNet 5.7 (Speer et al. 2017)
- Relaciones taxonomicas de WordNet (Miller, 1995)
