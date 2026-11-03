"""Dibuja en PNG las líneas seleccionadas del acumulador de Hough exportado."""
import argparse
import json
import math
from pathlib import Path
import statistics


def segment(width, height, cosine, sine, rho):
    """Interseca x*cos + y*sin = rho con los cuatro bordes de la imagen."""
    center_x, center_y = width // 2, height // 2
    points = []
    def add(column, row):
        if -1e-6 <= column <= width - 1 + 1e-6 and -1e-6 <= row <= height - 1 + 1e-6:
            point = (max(0, min(width - 1, column)), max(0, min(height - 1, row)))
            if not any(math.dist(point, other) < 1e-6 for other in points):
                points.append(point)
    if abs(sine) > 1e-7:
        for column in (0, width - 1):
            add(column, center_y - (rho - (column - center_x) * cosine) / sine)
    if abs(cosine) > 1e-7:
        for row in (0, height - 1):
            add(center_x + (rho - (center_y - row) * sine) / cosine, row)
    if len(points) < 2:
        return None
    return max(((a, b) for a in points for b in points), key=lambda pair: math.dist(*pair))


def select_lines(data, threshold=None, max_lines=12):
    """Picos por votos y supresión de líneas vecinas, incluida la unión 0°/180°."""
    width, height = data['width'], data['height']
    angles, bins = data['degree_bins'], data['r_bins']
    votes = data['votes']
    if width < 1 or height < 1 or angles < 1 or bins < 1 or len(votes) != angles * bins:
        raise ValueError('Dimensiones del acumulador invalidas')
    if len(data['cosines']) != angles or len(data['sines']) != angles:
        raise ValueError('Tablas trigonometricas invalidas')
    if data['r_scale'] <= 0 or any(v < 0 for v in votes):
        raise ValueError('Escala o votos invalidos')
    if threshold is None:
        threshold = statistics.mean(votes) + 2 * statistics.pstdev(votes)
    if not math.isfinite(threshold) or threshold < 0 or max_lines < 1:
        raise ValueError('Umbral o limite de lineas invalido')
    candidates = sorted((i for i, count in enumerate(votes) if count > threshold),
                        key=lambda i: (-votes[i], i))
    selected = []
    for index in candidates:
        radial_bin, angle_bin = divmod(index, angles)
        # Cada bin representa un intervalo: se dibuja su punto medio.
        rho = (radial_bin + 0.5) * data['r_scale'] - data['r_max']
        cosine, sine = data['cosines'][angle_bin], data['sines'][angle_bin]
        theta = math.atan2(sine, cosine)
        duplicate = False
        for other in selected:
            delta = abs(theta - other['theta_rad'])
            other_rho = other['rho']
            if delta > math.pi / 2:
                delta = math.pi - delta
                other_rho = -other_rho
            if delta <= math.radians(4.01) and abs(rho - other_rho) <= 2.01 * data['r_scale']:
                duplicate = True
                break
        if duplicate:
            continue
        endpoints = segment(width, height, cosine, sine, rho)
        if endpoints is None:
            continue
        selected.append(dict(r_bin=radial_bin, theta_bin=angle_bin, theta_rad=theta,
                             rho=rho, votes=votes[index], endpoints=endpoints))
        if len(selected) >= max_lines:
            break
    return threshold, selected


def render(image_path, accumulator_path, output_path, threshold=None, max_lines=12):
    from PIL import Image, ImageDraw
    data = json.loads(Path(accumulator_path).read_text())
    with Image.open(image_path) as original:
        image = original.convert('RGB')
    if image.size != (data['width'], data['height']):
        raise ValueError('La imagen no coincide con las dimensiones del acumulador')
    cutoff, lines = select_lines(data, threshold, max_lines)
    draw = ImageDraw.Draw(image)
    colors = [(255, 55, 55), (30, 235, 80), (20, 180, 255), (255, 200, 0), (240, 70, 230)]
    for index, line in enumerate(lines):
        draw.line(line['endpoints'], fill=colors[index % len(colors)], width=2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format='PNG')
    details = dict(variant=data['variant'], gpu_executed=data['gpu_executed'],
                   threshold=cutoff, threshold_rule='manual' if threshold is not None else 'mean + 2 * population_std',
                   max_lines=max_lines, nms_angle_degrees=4, nms_radial_bins=2,
                   radial_representation='bin midpoint', lines=lines)
    output_path.with_suffix('.json').write_text(json.dumps(details, indent=2) + '\n')
    print(f"Lineas: {len(lines)}; umbral: {cutoff:.2f} votos; imagen: {output_path}")
    return details


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--accumulator', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--max-lines', type=int, default=12)
    args = parser.parse_args()
    render(args.image, args.accumulator, args.output, args.threshold, args.max_lines)
