"""Scientific invariants; toy arrays are test fixtures, never observations."""
import unittest
import numpy as np
from phase50_temporal_placebo import comparison_dates, compare

class TemporalAuditTests(unittest.TestCase):
    def values(self,arrays):
        return {str(i):dict(values=np.array(a),weights=np.ones(len(a))) for i,a in enumerate(arrays)}

    def test_no_cross_edition_january_2023(self):
        self.assertNotIn('seasonal',comparison_dates('2023-01'))
        self.assertEqual(comparison_dates('2024-01')['seasonal'],['2023-12','2024-01','2022-12','2023-01'])

    def test_interannual_dates(self):
        self.assertEqual(comparison_dates('2024-06')['interannual'],['2023-06','2024-06','2022-06','2023-06'])
        self.assertNotIn('interannual',comparison_dates('2023-06'))

    def test_missing_is_not_dry(self):
        v=self.values([[0,1,2,1],[2,2,1,1],[1,1,2,1],[1,1,2,1]])
        r=compare(v,['0','1','2','3'],.3,'common_pixels')
        self.assertEqual(r['event_gain_cells'],1)
        self.assertEqual(r['event_loss_cells'],1)
        self.assertEqual(r['event_observed_km2'],3)

    def test_disjoint_observations_cannot_pass_primary(self):
        v=self.values([[1,1,0,0],[2,2,0,0],[0,0,1,1],[0,0,2,2]])
        self.assertIsNone(compare(v,['0','1','2','3'],.3,'common_pixels'))
        self.assertIsNotNone(compare(v,['0','1','2','3'],.3,'pair_specific'))

    def test_both_arms_must_pass(self):
        v=self.values([[1,1,1,1],[2,2,2,2],[1,0,0,0],[2,0,0,0]])
        self.assertIsNone(compare(v,['0','1','2','3'],.3,'pair_specific'))

    def test_shared_pixels_remove_differential_missingness(self):
        v=self.values([[1,1,1,1],[2,1,2,2],[1,1,0,0],[1,2,0,0]])
        r=compare(v,['0','1','2','3'],.3,'common_pixels')
        self.assertEqual(r['event_observed_km2'],r['placebo_observed_km2'])
        self.assertEqual(r['delta_gain_per100'],0)

if __name__=='__main__':unittest.main()
