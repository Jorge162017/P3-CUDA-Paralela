import json
import math
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from scripts.render_lines import render, segment, select_lines


def accumulator():
    return dict(width=11, height=11, degree_bins=90, r_bins=7,
                r_max=7.0, r_scale=2.0, variant='global', gpu_executed=True,
                cosines=[math.cos(math.radians(2 * i)) for i in range(90)],
                sines=[math.sin(math.radians(2 * i)) for i in range(90)],
                votes=[0] * 630)


class HoughOutputTests(unittest.TestCase):
    def test_vertical_horizontal_and_diagonal_geometry(self):
        for cosine, sine, rho in [(1, 0, 0), (0, 1, 0),
                                  (math.sqrt(0.5), math.sqrt(0.5), 0)]:
            endpoints = segment(11, 11, cosine, sine, rho)
            self.assertIsNotNone(endpoints)
            for col, row in endpoints:
                self.assertAlmostEqual((col - 5) * cosine + (5 - row) * sine, rho)
                self.assertTrue(0 <= col <= 10 and 0 <= row <= 10)
        self.assertIsNone(segment(11, 11, 1, 0, 100))

    def test_black_accumulator_has_no_lines(self):
        threshold, lines = select_lines(accumulator())
        self.assertEqual(threshold, 0)
        self.assertEqual(lines, [])

    def test_threshold_and_neighbor_suppression(self):
        data = accumulator()
        data['votes'][3 * 90] = 100
        data['votes'][3 * 90 + 1] = 99
        data['votes'][3 * 90 + 45] = 80
        threshold, lines = select_lines(data, threshold=80)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]['votes'], 100)
        _, lines = select_lines(data, threshold=79)
        self.assertEqual(len(lines), 2)

    def test_wraparound_uses_opposite_rho(self):
        data = accumulator()
        data['votes'][4 * 90] = 100  # rho +2, theta 0
        data['votes'][2 * 90 + 89] = 99  # rho -2, theta 178
        _, lines = select_lines(data, threshold=10)
        self.assertEqual(len(lines), 1)

    def test_png_contains_expected_color_and_preserves_background(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('L', (11, 11), 25).save(root / 'input.pgm')
            data = accumulator()
            data['votes'][3 * 90] = 100
            (root / 'acc.json').write_text(json.dumps(data))
            details = render(root / 'input.pgm', root / 'acc.json', root / 'out.png', threshold=10)
            self.assertEqual(len(details['lines']), 1)
            with Image.open(root / 'out.png') as image:
                self.assertEqual(image.format, 'PNG')
                self.assertEqual(image.size, (11, 11))
                self.assertEqual(image.getpixel((5, 5)), (255, 55, 55))
                self.assertEqual(image.getpixel((0, 0)), (25, 25, 25))
            self.assertTrue(json.loads((root / 'out.json').read_text())['gpu_executed'])

    def test_final_variant_provenance_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('L', (11, 11), 0).save(root / 'input.pgm')
            data = accumulator()
            data['variant'] = 'tres_memorias'
            (root / 'acc.json').write_text(json.dumps(data))
            details = render(root / 'input.pgm', root / 'acc.json', root / 'final.png')
            self.assertEqual(details['variant'], 'tres_memorias')
            self.assertTrue(details['gpu_executed'])
            self.assertEqual(details['lines'], [])

    def test_reject_malformed_accumulator(self):
        data = accumulator()
        data['votes'] = []
        with self.assertRaises(ValueError):
            select_lines(data)


if __name__ == '__main__':
    unittest.main()
