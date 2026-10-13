# proyecto_nlp

Proyecto de Procesamiento del Lenguaje Natural.

## Contenido

- `core.py`: funciones principales del proyecto.
- `implementacion_modulo_1.ipynb` … `implementacion_modulo_6.ipynb`: notebooks de cada módulo.
- `testing.ipynb`: notebooks de pruebas.
- `Datasets extra utilizados/`: datasets auxiliares.
- `Explicaciones/`: documentación/explicaciones de cada módulo.

## Dataset procesado

El archivo `processed_dataset/processed_dataset.csv` (~374 MB) **no está en el repositorio**
porque supera el límite de 100 MB por archivo de GitHub. Se distribuye como asset del Release:

- Descarga directa:
  https://github.com/pablosnow26/proyecto_nlp/releases/download/dataset-v1/processed_dataset.csv
- Página del release:
  https://github.com/pablosnow26/proyecto_nlp/releases/tag/dataset-v1

Para usarlo, descárgalo y colócalo en la carpeta `processed_dataset/`:

```bash
mkdir -p processed_dataset
curl -L -o processed_dataset/processed_dataset.csv \
  https://github.com/pablosnow26/proyecto_nlp/releases/download/dataset-v1/processed_dataset.csv
```

También puede regenerarse ejecutando `implementacion_modulo_1.ipynb`.

> SHA-256: `90755bcbc9d307eadeb03dee1d2da8cf95981838e7f1090a13f4f537d3a2284c`
> Líneas: `2433970`
