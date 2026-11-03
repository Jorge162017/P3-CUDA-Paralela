# Revisión después de integrar las ramas

## Avances integrados en main

- Versión global con lector PGM, manejo de errores, liberación de recursos y validación CPU/GPU.
- Compañero 2: variante global + constante, notebook, diagrama y validación en Tesla T4. Sus commits ya estaban integrados en el main remoto.
- Compañero 3: variante global + compartida, barreras, acumulación atómica por bloque, notebook, explicación, reporte de avance y evidencia en Tesla T4. Integrada preservando sus commits.
- Los registros en `results/` muestran coincidencia CPU/GPU para imagen original, bloques parciales e imagen negra; también prueban entradas inválidas. La revisión local no sustituye esas ejecuciones GPU.

## Pendientes del proyecto final

| Responsable del plan | Trabajo pendiente | Evidencia esperada |
| --- | --- | --- |
| Tú, fase 2 — completada | Diez mediciones globales verificadas: promedio 1.555098 ms, Tesla T4 | `results/fase2-global/global-times.csv` y log |
| Tú, fase 2 — completada | PNG con 12 líneas, umbral aproximado 3541.55 votos, verificado contra acumulador GPU | `results/fase2-global/global-lines.png` y JSON |
| Compañero 2, fase 4 | Medir constante + global al menos 10 veces y comparar con global | CSV, resumen de tiempos y análisis del efecto de memoria constante |
| Compañero 3, completar fase 5 | Combinar tablas constantes con el acumulador compartido | Variante final global + constante + compartida y nueva validación GPU |
| Compañero 3, fase 6 | Medir la variante final al menos 10 veces en iguales condiciones | CSV y tabla o gráfica comparativa de las tres versiones requeridas |
| Los tres; integración compañero 3 | Actualizar diagramas, análisis y conclusiones según tiempos reales | Informe final según guía UVG y al menos 3 referencias |
| Los tres | Exportar informe a PDF y preparar presentación | PDF final y presentación ejecutiva |

La variante global ya incorpora CUDA Events y exporta el acumulador; `scripts/render_lines.py` genera el PNG. El notebook `notebooks/fase2_global_colab.ipynb` ya se ejecutó y su evidencia GPU real está en `results/fase2-global/`. Las otras variantes todavía no incorporan mediciones. La captura `results/evidencia.png` es evidencia de ejecución; no hay una implementación que genere la imagen de detección pedida. La variante compartida actual es intermedia, no la variante final con las tres memorias. Los diagramas de memoria constante y compartida ya existen; falta actualizar el de la variante final y analizar resultados medidos.

## Entrega del avance del laboratorio

`docs/avance-lab5.md` existe, pero no se encontró su PDF exportado en el repositorio. Si ya fue entregado por otra vía, conservar también una copia en `docs/`. El avance del laboratorio y el informe final son entregables distintos.

## Compilación integrada

`make` compila tres ejecutables independientes: `build/hough`, `build/hough_const` y `build/hough_shared`. Ejecutar cada uno con `data/runway.pgm`. El último sigue siendo global + compartida.

Las pruebas GPU nuevas se ejecutan en Colab: este equipo local no tiene NVIDIA. Después de integrar las tres memorias se necesita un nuevo log; los existentes corresponden a las variantes anteriores.
