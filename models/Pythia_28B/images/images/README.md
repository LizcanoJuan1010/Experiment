# Imagenes de Prueba (Dominio Visual)

Imagenes de prueba para el test de ablacion conceptual en dominio visual, que evalua si la remocion de conceptos en un modelo de lenguaje afecta la generacion de descripciones de imagenes.

## Contenido

| Imagen | Categoria | Rol |
|--------|-----------|-----|
| `tools/hammer.png` | Herramientas | Target (concepto ablacionado) |
| `cow/image.png` | Animal | Control (concepto no ablacionado) |

## Usado por

[`vision_aphasia_test.py`](../vision_aphasia_test.py) — Test de ablacion en el modelo BLIP (multimodal). Evalua como la remocion del concepto "tools" afecta la descripcion de una imagen de martillo vs una imagen de control.
