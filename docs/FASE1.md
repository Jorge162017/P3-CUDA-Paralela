# Avance de la fase 1

## Implementado

- Lector PGM P2/P5 de 8 bits con validación de cabecera, dimensiones y datos.
- Kernel global con un hilo por píxel, ID lineal y límite `>= w * h`.
- Redondeo hacia arriba del número de bloques para cubrir el bloque parcial.
- Tablas trigonométricas y acumulador en memoria global, con `atomicAdd` para votos concurrentes.
- Referencia CPU que comparte las tablas trigonométricas con la GPU y compara todos los bins.
- Validación de argumentos y errores CUDA; vectores CPU y buffers GPU con liberación automática, seguida de `cudaDeviceReset`.
- Notebook Colab con nvcc4jupyter y empaquetado reproducible mediante `make colab-package`.

## Comprobación local realizada

`make cpu-check` compiló con Clang y procesó `data/runway.pgm`:

```text
Imagen: 800 x 600; acumulador: 9000 bins
CPU OK: 138406 pixeles activos, 12456540 votos. GPU NO ejecutada.
```

También se comprobó lectura P2 con comentarios, P5 con píxeles cuyos bytes son espacios o saltos de línea, imagen negra y rechazo de datos truncados, valores fuera de rango, dimensiones inválidas, argumentos ausentes y archivo inexistente. Se verificó el JSON del notebook y el contenido del ZIP.

## Pendiente para cerrar la fase

Ejecutar `notebooks/fase1_colab.ipynb` con GPU y guardar la evidencia descargada en `results/fase1-colab.txt`. La prueba CPU local no demuestra que el kernel CUDA compile o funcione. El notebook verifica la imagen original, una imagen de 21 píxeles (bloque parcial), una imagen negra y entradas inválidas.

La fase 2 agregará CUDA Events, bitácora de diez mediciones e imagen con líneas a color.

## Organización de los commits

Los cambios de esta fase se registran en grupos separados:

1. Organización inicial: `.gitignore`, imagen y enunciado en sus carpetas.
2. Lector PGM: `common/pgm.h` y `common/pgm.cpp`.
3. Versión global: `src/houghBase.cu` y `Makefile`.
4. Colab y documentación: notebook, README y este documento.
5. Después de ejecutar en Colab: evidencia real de GPU en `results/`.

No incluir `PLAN_FASES_LOCAL.md` ni `build/`; ambos están ignorados. No registrar una validación GPU exitosa antes de ejecutarla.
