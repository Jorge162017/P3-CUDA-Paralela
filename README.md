# Proyecto 3: Transformada de Hough con CUDA

Detección de líneas rectas en imágenes PGM y comparación de tres variantes: memoria global; global + constante; global + constante + compartida.

## Estado

Las tres variantes están integradas en `main` y tienen validación contra CPU en Colab (Tesla T4, CUDA Toolkit 13.0). La comparación final conserva 60 mediciones por variante en la misma sesión, tomadas de los últimos diez lanzamientos de seis rondas de cien.

| Variante | Ejecutable | Tiempo medio del kernel |
| --- | --- | --- |
| Global | `build/hough` | 0.900314 ms |
| Global + constante | `build/hough_const` | 0.861503 ms |
| Global + constante + compartida | `build/hough_final` | 0.736581 ms |

La variante final redujo el tiempo un 18.19 % frente a la global con ese protocolo. La cifra depende del tramo medido; con los cien lanzamientos completos la reducción es aproximadamente 16 %. Ver [análisis y limitaciones](docs/memoria-final.md) y [bitácoras](results/fase6-tres-memorias/final/). No comparar directamente estos tiempos con los de sesiones anteriores.

La versión global tiene [PNG y evidencia](results/fase2-global/). La versión final ahora exporta su propio acumulador y reutiliza el generador de PNG. **Salida final validada en Colab:** [PNG, acumulador y logs](results/salida-final/). Coincidencia con CPU para la imagen original, imagen negra y bloques parciales de 21 y 300 píxeles. **Faltan el informe PDF y la presentación.** Ver [pendientes](docs/PENDIENTES.md).

## Estructura

```text
src/          Variantes global, constante, compartida intermedia y final
common/       Lector PGM y exportación común del acumulador
scripts/      Generador de PNG con líneas seleccionadas
notebooks/    Ejecución y validación en Colab
results/      Bitácoras, CSV, gráficos e imágenes
tests/       Pruebas del generador de imagen
docs/         Enunciado, análisis y material para el informe
```

## Compilación

Requiere GPU NVIDIA, CUDA Toolkit con `nvcc`, compilador C++17 compatible y `make`. Para el dibujo se requiere Python y Pillow.

```sh
make
./build/hough data/runway.pgm
./build/hough_const data/runway.pgm
./build/hough_final data/runway.pgm
```

`build/hough_shared` conserva la variante intermedia global + compartida. La versión final usa `src/houghSharedConst.cu`. Todos los programas comparan el acumulador contra CPU y devuelven error ante diferencias o fallos de CUDA. `make clean` elimina los compilados.

## Salida final con líneas a color

```sh
./build/hough_final data/runway.pgm --runs 1 --output results/salida-final
python3 -m pip install -r requirements.txt
python3 scripts/render_lines.py --image data/runway.pgm --accumulator results/salida-final/tres-memorias-accumulator.json --output results/salida-final/tres-memorias-lines.png
```

El acumulador se exporta después de validar los lanzamientos, fuera del tiempo medido, y conserva dimensiones y tablas trigonométricas exactas. El PNG superpone hasta doce líneas con votos mayores que promedio + dos desviaciones estándar, suprimiendo líneas vecinas. El JSON junto al PNG registra umbral, votos y coordenadas. Se puede ajustar con `--threshold` y `--max-lines`. El radio representativo es el centro del bin, por lo que el dibujo conserva la aproximación de la discretización.

El [notebook de salida final](notebooks/salida_final_colab.ipynb) compila desde `main`, ejecuta la versión final, compara también el acumulador con la evidencia global, prueba imágenes negras y bloques parciales, genera el PNG y descarga `salida-final.zip`. No requiere subir el código manualmente. Esta validación guarda sus archivos en `results/salida-final/` y conserva las bitácoras comparativas previas.

## Mediciones

```sh
./build/hough data/runway.pgm --runs 10 --output results/mi-global
./build/hough_const data/runway.pgm --runs 10 --output results/mi-constante
./build/hough_final data/runway.pgm --runs 10 --output results/mi-final
```

Los programas usan CUDA Events alrededor del kernel, un calentamiento excluido y reinicio del acumulador antes de cada ejecución. Las copias, la validación y la exportación quedan fuera del tiempo. Para comparar versiones en condiciones semejantes, usar [el notebook de las tres memorias](notebooks/fase6_tres_memorias_colab.ipynb), cuyo protocolo y resultados se explican en [memoria-final.md](docs/memoria-final.md).

Otros notebooks: [fase 1](notebooks/fase1_colab.ipynb), [global e imagen](notebooks/fase2_global_colab.ipynb), [constante](notebooks/constante_colab.ipynb), [comparación global/constante](notebooks/fase4_constante_colab.ipynb). El notebook de fase 1 usa un ZIP que puede crearse con `make colab-package`.

## Comprobación local sin GPU

```sh
make cpu-check
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

La primera orden solo ejecuta la referencia CPU. Las pruebas de Python comprueban geometría, selección de picos, PNG y procedencia de la variante. Estas pruebas no sustituyen ejecutar CUDA en Colab.

## Documentación y entrega

- [Enunciado](docs/Proyecto%203%20-%20CUDA.pdf).
- [Memoria constante](docs/memoria-constante.md).
- [Memoria compartida intermedia](docs/memoria-compartida.md).
- [Variante final y comparación](docs/memoria-final.md).
- [Estado de entrega](docs/PENDIENTES.md).

Entregar código, mediciones, informe PDF, enlace al repositorio y presentación ejecutiva. Cada integrante registra sus avances en commits separados, conservando su autoría. Los compilados y el plan local están excluidos de Git.
