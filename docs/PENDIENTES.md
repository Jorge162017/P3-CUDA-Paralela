# Estado del proyecto antes del informe final

## Completado y respaldado por evidencia

- Versión global, constante y final con las tres memorias integradas en `main`.
- Lectura PGM P2/P5, validación de argumentos y CUDA, acumulación atómica, sincronización y liberación de recursos.
- Validación CPU/GPU y pruebas de bloques parciales de las variantes originales.
- Diez mediciones globales e imagen PNG con doce líneas: `results/fase2-global/`.
- Comparación global/constante de fase 4 y análisis de sus condiciones: `results/fase4-constante/final/`.
- Comparación final en la misma sesión: 60 mediciones reportadas por variante, seis rondas y validación CPU/GPU en cada lanzamiento. Promedios global 0.900314 ms, constante 0.861503 ms y final 0.736581 ms. CSV, logs, gráfica y recursos compilados en `results/fase6-tres-memorias/final/`.
- Explicaciones, diagramas conceptuales y referencias en `memoria-constante.md`, `memoria-compartida.md` y `memoria-final.md`.

## Salida final: implementada, falta evidencia GPU del ajuste

La versión final ahora exporta `tres-memorias-accumulator.json` después de validar el acumulador. La misma función de exportación se comparte con la versión global en `common/hough_output.h`. El dibujo usa `scripts/render_lines.py` y conserva la etiqueta de variante `tres_memorias`.

Ejecutar `notebooks/salida_final_colab.ipynb`: comprueba original, imagen negra, bloques parciales de 21 y 300 píxeles y coincidencia con el acumulador global previamente guardado. Genera el PNG y descarga un ZIP con rutas `results/salida-final/`. Incorporar esos archivos al repositorio para cerrar la validación de la nueva salida. Esta ejecución no reemplaza ni mezcla las mediciones comparativas existentes.

Hasta recibir ese ZIP, no afirmar que la nueva exportación ya se ejecutó en GPU. Las comprobaciones locales cubren la serialización y el dibujo, además de la referencia CPU.

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
