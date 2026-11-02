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

## Validación GPU completada

La evidencia real está en [results/fase1-colab.txt](../results/fase1-colab.txt): Google Colab con Tesla T4, driver 580.82.07 y CUDA Toolkit 13.0 (nvcc V13.0.88).

- Imagen original de 800 × 600: 1875 bloques de 256 hilos; todos los bins coinciden entre CPU y GPU.
- Imagen de 7 × 3 con píxeles activos: bloque parcial validado, sin diferencias CPU/GPU.
- Imagen negra de 7 × 3: todos los bins coinciden.
- Argumentos ausentes y archivo inexistente: errores esperados registrados.

La primera ejecución completó las pruebas, pero falló al guardar el log porque faltaba la carpeta `results/`. Se corrigió el notebook para crearla y se conservó la evidencia de la sesión. Con estas comprobaciones se cierra la fase 1.

La fase 2 agregará CUDA Events, bitácora de diez mediciones e imagen con líneas a color.

## Organización de los commits

Los cambios de esta fase se registran en grupos separados:

1. Organización inicial: `.gitignore`, imagen y enunciado en sus carpetas.
2. Lector PGM: `common/pgm.h` y `common/pgm.cpp`.
3. Versión global: `src/houghBase.cu` y `Makefile`.
4. Colab y documentación: notebook, README y este documento.
5. Después de ejecutar en Colab: evidencia real de GPU en `results/`.

No incluir `PLAN_FASES_LOCAL.md` ni `build/`; ambos están ignorados. No registrar una validación GPU exitosa antes de ejecutarla.
