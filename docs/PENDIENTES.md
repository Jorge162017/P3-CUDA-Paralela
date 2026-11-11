# Estado del proyecto antes del informe final

## Completado y respaldado por evidencia

- Versión global, constante y final con las tres memorias integradas en `main`.
- Lectura PGM P2/P5, validación de argumentos y CUDA, acumulación atómica, sincronización y liberación de recursos.
- Validación CPU/GPU y pruebas de bloques parciales de las variantes originales.
- Diez mediciones globales e imagen PNG con doce líneas: `results/fase2-global/`.
- Comparación global/constante de fase 4 y análisis de sus condiciones: `results/fase4-constante/final/`.
- Comparación final en la misma sesión: 60 mediciones reportadas por variante, seis rondas y validación CPU/GPU en cada lanzamiento. Promedios global 0.900314 ms, constante 0.861503 ms y final 0.736581 ms. CSV, logs, gráfica y recursos compilados en `results/fase6-tres-memorias/final/`.
- Explicaciones, diagramas conceptuales y referencias en `memoria-constante.md`, `memoria-compartida.md` y `memoria-final.md`.

## Salida final: completada y validada en GPU

La versión final exporta `tres-memorias-accumulator.json` después de validar el acumulador. La función se comparte con la versión global en `common/hough_output.h`. El dibujo conserva la etiqueta `tres_memorias`.

La ejecución real de `notebooks/salida_final_colab.ipynb` sobre el commit `048085ed7a76c71aad2434cc8472701449cf907a` pasó en Tesla T4 con CUDA Toolkit 13.0. Evidencia en `results/salida-final/`:

- Original 800 × 600, imagen negra y bloques parciales de 21 y 300 píxeles coinciden con CPU.
- Acumulador final idéntico al global previamente registrado: 12 456 540 votos.
- PNG con doce líneas y umbral aproximado de 3541.55 votos, reproducido y comparado píxel por píxel con el exportado de Colab.
- CSV de una ejecución para validar la salida, conservado aparte de las mediciones comparativas de fase 6. Su tiempo aislado no sustituye el promedio del experimento comparativo.

Con esta evidencia se cierra el ajuste de salida. Quedan informe PDF, presentación y revisión de entrega.

## Informe final PDF

Integrar los materiales existentes en un documento con:

- Carátula, índice e introducción según guía UVG.
- Al menos una página de contenido sobre Hough, implementación CUDA y memorias GPU.
- Explicación de coordenadas, constantes y memoria compartida; diagramas de entradas, etapas y salidas.
- Bitácoras (mínimo diez tiempos de cada variante), configuración del equipo, protocolo y comparación de la misma sesión.
- Imagen con líneas a color y justificación del umbral.
- Conclusiones, recomendaciones, limitaciones, al menos tres referencias confiables y apéndice.

Usar los resultados de fase 6 para la comparación principal. Explicar que se reportaron los últimos diez de cien lanzamientos por ronda y que el porcentaje cambia con el tramo. Las mediciones dentro de una ejecución están correlacionadas: no presentar los valores p calculados como prueba concluyente de independencia o generalización a otras GPU/imágenes.

El PDF del enunciado no es el informe final. `avance-lab5.md` tampoco sustituye el informe final.

## Presentación y entrega

- Preparar presentación ejecutiva con participación de los tres integrantes.
- Revisar instrucciones y rúbrica del curso para la exposición.
- Entregar código además del enlace GitHub, bitácoras e informe PDF. Incluir la evidencia de salida final.
- Verificar la entrega en la plataforma del curso; el repositorio no confirma por sí solo una entrega académica.
