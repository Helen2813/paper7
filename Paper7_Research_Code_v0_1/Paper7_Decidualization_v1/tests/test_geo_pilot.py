import unittest
import numpy as np
import pandas as pd
from paper7_pilot import parse_matrix_headers, infer_annotation, identifiability_diagnostics, stub


class TestPilot(unittest.TestCase):
    def test_geo_stubs(self):
        self.assertEqual(stub('GSE243039'), 'GSE243nnn')
        self.assertEqual(stub('GSE35287'), 'GSE35nnn')

    def test_geo_parser_and_annotation(self):
        lines = [
            '!Sample_title\t"Normal endometrial stroma, Replicate 2, Day0"\t"Endometriosis stroma, Replicate 1, Day2"',
            '!Sample_geo_accession\t"GSM1"\t"GSM2"',
            '!Sample_characteristics_ch1\t"tissue: endometrium"\t"tissue: endometrium"',
            '!Sample_characteristics_ch1\t"time: day0"\t"time: day2"',
        ]
        result = parse_matrix_headers(lines, 'GSE243039')
        self.assertEqual(result.shape[0], 2)
        controls = infer_annotation(result.iloc[0])
        cases = infer_annotation(result.iloc[1])
        self.assertEqual(controls['group'], 'control')
        self.assertEqual(cases['group'], 'endometriosis')
        self.assertEqual(controls['time_hours'], 0)
        self.assertEqual(cases['time_hours'], 48)

    def test_identifiability_collinear(self):
        # For an exponential f, derivative f'=f: time shift and amplitude are locally inseparable.
        t = np.array([0., 1., 2., 3., 4.])
        f = np.exp(t/2)
        res = identifiability_diagnostics(t, f, smooth=False)
        self.assertGreater(res['abs_corr_f_fprime'], 0.90)

    def test_input_min_points(self):
        with self.assertRaises(ValueError):
            identifiability_diagnostics(np.array([0., 1., 2.]), np.array([0., 1., 2.]))


if __name__ == '__main__':
    unittest.main()
