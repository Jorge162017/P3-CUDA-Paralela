# Proyecto 3: Transformada de Hough con CUDA

Proyecto de Computación Paralela y Distribuida para detectar líneas rectas en imágenes y comparar el tiempo de ejecución de tres versiones CUDA:

1. Memoria global.
2. Memoria global y constante.
3. Memoria global, constante y compartida.

## Estado actual

La fase 1 está implementada: lector PGM P2/P5 de 8 bits, kernel con memoria global, validación de argumentos y errores CUDA, comparación exacta CPU/GPU y liberación de recursos. La referencia CPU utiliza las mismas tablas trigonométricas que la GPU. Se desactiva la fusión de operaciones flotantes en la compilación para mantener una comparación reproducible.

**Pendiente:** ejecutar la validación GPU en Colab y guardar la evidencia. La medición con CUDA Events y las imágenes de salida pertenecen a la fase 2; las memorias constante y compartida siguen pendientes.

## Estructura

```text
src/houghBase.cu             Versión global y referencia CPU
common/                     Lector PGM P2/P5 de 8 bits
data/                      Imagen de prueba (runway.pgm)
docs/                       Enunciado y futuro informe
results/                    Mediciones e imágenes generadas
notebooks/fase1_colab.ipynb   Ejecución guiada en Colab
Makefile                    Compilación
```

## Requisitos

- GPU NVIDIA compatible con el CUDA Toolkit instalado.
- Compilador `nvcc`, compilador C++ compatible y `make`.
- Archivos del lector PGM en `common/`.

## Compilar y ejecutar

Desde la raíz del repositorio, en un equipo CUDA:

```sh
make
./build/hough data/runway.pgm
```

Para limpiar los archivos compilados:

```sh
make clean
```

La ejecución correcta debe mostrar `VALIDACION OK: CPU y GPU coinciden en todos los bins.` Un error de argumentos, lectura, CUDA o diferencias entre acumuladores devuelve un código distinto de cero.

## Google Colab sin GPU local

1. Ejecuta `make colab-package` en tu computadora.
2. Abre [el notebook](notebooks/fase1_colab.ipynb) en Google Colab mediante **Subir notebook**.
3. Activa un entorno con GPU y ejecuta las celdas en orden.
4. Cuando se solicite, sube `build/proyecto-colab.zip`.
5. Descarga la evidencia y guárdala en `results/fase1-colab.txt` para un commit posterior.

El notebook instala y carga `nvcc4jupyter`, ejecuta el programa mediante `%%cuda` y verifica el ejecutable con el Makefile. También comprueba una imagen negra, un bloque parcial y entradas inválidas. Consulta la [documentación de nvcc4jupyter](https://nvcc4jupyter.readthedocs.io/en/latest/usage.html).

## Comprobación local sin CUDA

```sh
make cpu-check
```

Solo comprueba lectura de imagen y conteo de votos CPU; **no valida la GPU**. Las tablas y vectores CPU se liberan automáticamente; los buffers GPU usan RAII y el dispositivo se reinicia al terminar.

## Resultados y entrega

- Completar las tres versiones y comparar sus resultados con la CPU.
- Medir el kernel con CUDA Events: al menos 10 mediciones por versión.
- Guardar una imagen PNG o JPG con las líneas detectadas a color.
- Documentar el uso de memoria constante y compartida con explicaciones y diagramas.
- Entregar código, bitácora de tiempos, informe PDF, enlace al repositorio y presentación oral.

Consultar los requisitos completos en [el enunciado](docs/Proyecto%203%20-%20CUDA.pdf).

## Colaboración

Subir avances en commits pequeños, separados y con mensajes claros. Cada integrante debe registrar sus propias contribuciones: no acumular todo el proyecto en uno o dos commits. Actualizar este README cuando cambien los comandos o se complete una funcionalidad.
