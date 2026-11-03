# Fase 2: medición global e imagen con líneas

## Estado

Implementación lista para comprobar en GPU con `notebooks/fase2_global_colab.ipynb`. No se han registrado todavía diez mediciones reales de esta fase ni se atribuyen tiempos a las pruebas locales de CPU.

## Medición del kernel

```sh
make build/hough
./build/hough data/runway.pgm --runs 10 --output results/fase2-global
```

El programa realiza un calentamiento completo, excluido de las diez mediciones. Reserva imagen, tablas y acumulador una vez. Antes de cada ejecución medida pone el acumulador en cero y registra dos CUDA Events en el mismo stream, uno antes y otro después del lanzamiento. Espera el evento final y usa `cudaEventElapsedTime` para obtener milisegundos. El borrado, las transferencias de imagen/tablas, la comparación CPU/GPU, la exportación y el dibujo quedan fuera del intervalo medido.

Cada una de las diez ejecuciones compara sus 9 000 bins con la CPU. Cualquier diferencia o error CUDA hace fallar el programa; los archivos de salida de esa ejecución solo se escriben después de superar todas las validaciones. Los dos eventos se destruyen mediante RAII, igual que los buffers; al final se reinicia el dispositivo. Conviene usar una carpeta nueva para cada sesión si se quieren conservar varias bitácoras.

`global-times.csv` contiene una fila por ejecución: versión, número, milisegundos, GPU, versiones de runtime/driver, dimensiones, bloques, hilos, cantidad de calentamientos y validación. El notebook muestra las diez filas y un resumen con promedio, desviación estándar muestral, mínimo y máximo. `global-validacion.txt` añade salida del programa, información de `nvidia-smi`, versión de `nvcc`, flags y commit ejecutado.

Las variantes de los compañeros deben usar el mismo protocolo y configuración al comparar tiempos. Si cambian GPU o entorno, repetir las tres versiones en la misma sesión. Estos tiempos describen el kernel y no la duración completa del programa.

## Imagen a color

El ejecutable exporta `global-accumulator.json` con el acumulador final y las mismas tablas trigonométricas usadas en las votaciones. El script dibuja sobre la imagen original y genera PNG:

```sh
python3 -m pip install -r requirements.txt
python3 scripts/render_lines.py --image data/runway.pgm \
  --accumulator results/fase2-global/global-accumulator.json \
  --output results/fase2-global/global-lines.png
```

Se eligen bins cuyo peso es estrictamente mayor que `promedio + 2 * desviación estándar poblacional` de los 9 000 contadores. Para evitar líneas muy parecidas se suprimen candidatos vecinos a 4 grados y 2 bins radiales de una línea ya elegida. Se conservan hasta 12 líneas, ordenadas por votos. La unión angular entre 0 y 180 grados se trata invirtiendo el signo del radio. Una imagen sin votos no genera líneas.

El radio dibujado es el centro del intervalo de cada bin. Se intersecta la recta `x*cos(theta) + y*sin(theta) = rho` con los bordes de la imagen; el origen es `(width//2, height//2)` y el eje vertical se invierte al pasar de coordenadas de imagen a cartesianas. La discretización de 100 radios introduce una aproximación: una línea puede quedar desplazada respecto al borde real. No se aplica un detector de bordes adicional a la imagen proporcionada.

El umbral puede cambiarse con `--threshold 4000` y el límite con `--max-lines 8`. El JSON junto al PNG registra el umbral real, la regla de selección, los votos y los extremos de cada línea. Así se puede repetir el dibujo sin volver a medir el kernel. El notebook separa la medición de la celda de dibujo y descarga todo en `fase2-global.zip`, conservando dentro las rutas `results/fase2-global/`.

## Comprobaciones locales

- Referencia CPU compilada y ejecutada sobre `runway.pgm`, con 12 456 540 votos.
- Exportación JSON de CPU y generación del PNG comprobadas localmente; ese archivo se marca `gpu_executed=false` y permanece en `build/`, fuera de Git.
- Pruebas automatizadas de líneas verticales, horizontales y diagonales; imagen negra; umbral estricto; supresión de duplicados, incluida la unión 0/180 grados; color del PNG, conservación del fondo y rechazo de dimensiones inválidas.
- Los tests se ejecutan con `python3 -m unittest discover -s tests -v` después de instalar Pillow.

Estas comprobaciones no validan CUDA Events en GPU. Para cerrar la fase, ejecutar el notebook y agregar a Git el CSV, el log, el resumen, el acumulador, el PNG y el JSON de líneas de la sesión real.

## Referencias

- NVIDIA: [How to Implement Performance Metrics in CUDA C/C++](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/).
- NVIDIA: [CUDA Runtime API — Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html).
