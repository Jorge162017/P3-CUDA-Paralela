# Avance del Laboratorio #5: Transformada de Hough con CUDA

**Proyecto 3 – Computación Paralela y Distribuida** · Repositorio: https://github.com/Jorge162017/P3-CUDA-Paralela

## Objetivo

El proyecto consiste en detectar líneas rectas en una imagen mediante la transformada de Hough, ejecutada en una GPU con CUDA, y comparar el tiempo de tres variantes del programa según el tipo de memoria que usan: memoria global, memoria global con memoria constante, y memoria global con constante y compartida. Este documento resume los avances registrados hoy en el repositorio, con la evidencia obtenida en Google Colab.

## Versión base con memoria global

La versión base está completa y validada. Incluye un lector de imágenes PGM (formatos P2 y P5 de 8 bits) con validación de la cabecera y de los datos, y un kernel en el que cada hilo procesa un píxel mediante un identificador lineal, con el límite `>= w * h` y el número de bloques redondeado hacia arriba para cubrir bloques parciales. Las tablas de senos y cosenos y el acumulador viven en memoria global, y los votos concurrentes se suman con `atomicAdd`. El programa valida argumentos y errores de CUDA, libera automáticamente los recursos de CPU y GPU y compara todos los bins del acumulador contra una referencia en CPU que usa las mismas tablas trigonométricas. La compilación usa `--fmad=false` para que la comparación entre CPU y GPU sea reproducible.

En Google Colab, con una Tesla T4 (driver 580.82.07, CUDA 13.0, nvcc V13.0.88), la imagen original de 800 × 600 se procesó con 1875 bloques de 256 hilos y todos los bins coincidieron entre CPU y GPU. También coincidieron una imagen de 7 × 3 con píxeles activos (bloque parcial) y una imagen negra, y el programa rechazó correctamente los argumentos ausentes y los archivos inexistentes. En la imagen original hay 138 406 píxeles activos que emiten 12 456 540 votos sobre 9 000 bins.

## Variante con memoria constante

Se implementó una segunda variante en `src/houghConst.cu`, que conserva la versión global y cambia únicamente la ubicación de las tablas trigonométricas. Los 90 cosenos y 90 senos se declaran como `__constant__` fuera de `main`, se cargan desde la CPU con `cudaMemcpyToSymbol` y el kernel ya no recibe punteros a esas tablas. La razón del cambio es que las tablas son de solo lectura, idénticas para todos los píxeles y pequeñas (720 bytes), y que todos los hilos de un warp leen la misma posición en cada iteración, lo que permite a la caché constante repartir el dato a todo el warp en una sola lectura. La imagen y el acumulador permanecen en memoria global.

En Colab la variante constante y la global coincidieron con la CPU en todos los bins para la imagen original, la imagen de bloque parcial y la imagen negra, y el programa rechazó los argumentos inválidos. Con `cuobjdump -res-usage` se confirmó que el módulo reserva exactamente 720 bytes de memoria constante y que los parámetros del kernel se reducen en 16 bytes, que corresponden a los dos punteros eliminados. 

## Variante con memoria compartida

La tercera variante, en `src/houghShared.cu`, agrega un acumulador por bloque en memoria compartida (`localAcc`, de 90 × 100 enteros, es decir 36 000 bytes). Cada bloque inicializa `localAcc` de forma cooperativa y espera en una primera barrera `__syncthreads()`; luego cada hilo con un píxel válido y distinto de cero vota con `atomicAdd` sobre `localAcc`; una segunda barrera garantiza que todos los votos del bloque estén registrados, y finalmente los hilos suman cooperativamente `localAcc` al acumulador global con `atomicAdd`, solo en los bins que recibieron votos. Los hilos que quedan fuera de la imagen en el último bloque no regresan antes de las barreras: solo se condiciona el procesamiento del píxel, de modo que todos participan en la sincronización. Antes de lanzar el kernel, el programa comprueba que la GPU ofrezca los 36 000 bytes necesarios.

Esta es una variante intermedia, global más compartida: las tablas trigonométricas todavía están en memoria global, y se pasarán a memoria constante al integrar las dos variantes anteriores. La imagen original (1875 bloques de 256 hilos, sin bloque parcial), una imagen de 7 × 3 con 21 píxeles activos, una imagen negra de 7 × 3 y una imagen de 20 × 15 con 300 píxeles (dos bloques, el último con 44 píxeles) coincidieron con la CPU en todos los bins. La Tesla T4 ofrece 49 152 bytes de memoria compartida por bloque frente a los 36 000 requeridos, y los argumentos ausentes y los archivos inexistentes se rechazaron con código de salida distinto de cero. Las imágenes de 21 y 300 píxeles son las que ejercitan las barreras con hilos sin píxel.

## Resultados y alcance

Hoy quedaron validadas, contra la misma referencia de CPU, la versión global, la variante con memoria constante y la variante intermedia con memoria compartida, todas ejecutadas en la misma GPU. No se registran todavía mediciones de tiempo, por lo que no se afirma ninguna mejora de rendimiento. Se identificaron hipótesis para la medición final: la memoria constante podría aportar poca mejora porque en la Tesla T4 las lecturas globales también pasan por cachés y el costo probablemente lo dominan los `atomicAdd` sobre el acumulador; la memoria compartida reduce la contención sobre el acumulador global, pero cada bloque puede volcar hasta 9 000 bins y el uso de 36 000 bytes por bloque podría limitar cuántos bloques residen a la vez en cada multiprocesador. Estas hipótesis se comprobarán con las mediciones.


## Archivos y registros de referencia

Los avances se reflejan en el repositorio.

## Pendientes

Quedan por hacer la medición del kernel con CUDA Events y su registro en `results/`, la imagen con las líneas detectadas dibujadas a color, la integración de las tablas constantes en la variante con memoria compartida para obtener la versión final con las tres memorias, la comparación de tiempos con su análisis y diagramas, y el informe final con la presentación. A