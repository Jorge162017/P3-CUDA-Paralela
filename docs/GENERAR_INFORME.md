# Editar y regenerar el informe

El texto editable está en `docs/INFORME_FINAL.md` y los nombres/carnés/fecha en `docs/informe-config.json`. El generador produce `docs/INFORME_FINAL.pdf`, con índice y marcadores automáticos. Los diagramas y la gráfica se regeneran en `docs/assets/`.

Desde la raíz:

```sh
python3 -m venv build/informe-venv
build/informe-venv/bin/python -m pip install -r requirements-report.txt
build/informe-venv/bin/python scripts/build_report.py
```

El script usa Arial si está disponible en macOS y DejaVu Sans en otros sistemas. No requiere acceso a red después de instalar dependencias: logo, imagen final, fuente del informe y CSV están en el repositorio.

La gráfica se calcula con los CSV de fase 6. Las tablas del Markdown representan esa misma evidencia y pueden editarse, pero si se cambian las mediciones se deben actualizar también las tablas, el análisis y las conclusiones. Los apéndices incluyen los 180 tiempos reportados de las seis rondas.

La distribución de la carátula y los estilos se editan en `scripts/build_report.py`. El logo proviene del paquete institucional UVG, cuyo origen está registrado en `docs/assets/FUENTES.md`.
