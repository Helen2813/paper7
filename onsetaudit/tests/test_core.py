"""Regression / safety tests for OnsetAudit. Run: python -m unittest discover ..."""
import csv
import gzip
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Dataset, clean_metadata, table_from_bytes  # noqa: E402


def csv_bytes(rows, sep=','):
    f=io.StringIO(newline='')
    w=csv.writer(f,delimiter=sep);w.writerows(rows)
    return f.getvalue().encode('utf-8')


class OnsetAuditTests(unittest.TestCase):
    def test_synthetic_demo_is_clearly_marked_and_evaluable(self):
        d=Dataset()
        s=d.demonstration()
        self.assertTrue(s['demo'])
        self.assertEqual(s['genes'],450)
        self.assertEqual(s['samples'],72)
        self.assertTrue(s['matrix_loaded'])
        self.assertGreater(len(d.curve('SIM_GENE_001')['series']),1)
        output=d.evaluate()
        self.assertEqual(output['status'],'SYNTHETIC_DEMO_NOT_EVIDENCE')
        self.assertEqual(output['n_active'],48)
        self.assertEqual(output['n_vehicle'],24)
        self.assertEqual(output['vehicle_rejected'],24)
        self.assertFalse('confidence_interval' in output)

    def test_import_and_gene_header_with_unnamed_first_column(self):
        metadata=csv_bytes([
            ['sample_column_name','library','drug','timepoint','replicate'],
            ['DMSO_6h_1','DS009','DMSO','6h','rep_1'],
            ['RX_6h_1','DS009','RX','6h','rep_1'],
            ['DMSO_22h_1','DS017','DMSO','22h','rep_2'],
            ['RX_22h_1','DS017','RX','22h','rep_2'],
        ],'\t')
        rows=[
            ['DMSO_6h_1','RX_6h_1','DMSO_22h_1','RX_22h_1'],
            ['gene01','12','14','18','22'],
            ['gene02','7','11','13','24'],
            ['gene03','1','1','10','5'],
        ]
        matrix=gzip.compress(csv_bytes(rows,'\t'))
        d=Dataset(); d.import_metadata(metadata,'samples.tsv')
        s=d.import_matrix(matrix,'rawcounts.txt.gz','counts')
        self.assertEqual(s['samples'],4)
        self.assertEqual(s['genes'],3)
        self.assertTrue(s['matrix_loaded'])
        self.assertEqual(d.gene_ids,['gene01','gene02','gene03'])
        self.assertEqual(d.counts.shape,(3,4))
        self.assertEqual(d.counts[0,0],12)
        self.assertEqual(d.curve('gene01')['gene'],'gene01')
        self.assertEqual(d.design_audit(6,12)['between_onsets_h'],[])
        with self.assertRaisesRegex(ValueError,'200 gene rows'):
            d.evaluate()

    def test_duplicate_metadata_id_rejected(self):
        source=csv_bytes([
            ['sample','drug','time'],
            ['abc','DMSO','0h'],
            ['abc','RX','6h']
        ])
        with self.assertRaisesRegex(ValueError,'Duplicate sample ID'):
            clean_metadata(source,'metadata.csv')

    def test_sample_mismatch_rejected(self):
        meta=csv_bytes([['sample','drug','time'],['s1','DMSO','6'],['s2','RX','6']])
        mat=csv_bytes([['gene_id','s1','s3'],['g1','4','5']])
        d=Dataset();d.import_metadata(meta,'meta.csv')
        with self.assertRaisesRegex(ValueError,'sample mismatch'):
            d.import_matrix(mat,'expr.csv')

    def test_non_integer_counts_rejected(self):
        meta=csv_bytes([['sample','drug','time'],['s1','DMSO','6'],['s2','RX','6']])
        mat=csv_bytes([['gene_id','s1','s2'],['g1','4.5','5']])
        d=Dataset();d.import_metadata(meta,'meta.csv')
        with self.assertRaisesRegex(ValueError,'nonnegative integers'):
            d.import_matrix(mat,'expr.csv')

    def test_missing_controls_gate(self):
        d=Dataset(); d.demonstration()
        d.meta=d.meta[d.meta.drug!='DMSO'].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError,'Exactly one'):
            d.evaluate()

    def test_invalid_onset_bounds(self):
        d=Dataset();d.demonstration()
        with self.assertRaises(ValueError):
            d.design_audit(12,6)
        result=d.design_audit(0,6)
        self.assertEqual(result['status'],'STRUCTURALLY_AMBIGUOUS_WITH_ARBITRARY_POST_ONSET_SHAPES')
        self.assertEqual(result['between_onsets_h'],[1.,2.,6.])

class ReproducibilityTests(unittest.TestCase):
    def test_fixed_synthetic_demo_repeatability(self):
        first=Dataset();first.demonstration();r1=first.evaluate()
        second=Dataset();second.demonstration();r2=second.evaluate()
        self.assertEqual(r1['predictions'],r2['predictions'])
        self.assertEqual(r1['vehicle_predictions'],r2['vehicle_predictions'])
        self.assertEqual(r1['status'],'SYNTHETIC_DEMO_NOT_EVIDENCE')

if __name__ == "__main__": unittest.main()
