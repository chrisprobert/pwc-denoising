import unittest
import numpy as np
from mfblp import mfblp_filter

class TestMfblp(unittest.TestCase):

    def test_roll_sum_simple(self):
        x = np.arange(5)
        m = np.zeros(5)
        rs0 = np.array([0, 1, 2, 3, 4])
        rs1 = np.array([1.5, 3, 6, 9, 10.5])

        self.assertTrue((mfblp_filter.roll_sum(x, m, 0) == rs0).all())
        self.assertTrue((mfblp_filter.roll_sum(x, m, 1) == rs1).all())

    def test_roll_sum_diffs(self):
        x = np.array([1., 3., 0., 2., 0., 1., 4.])
        m = np.array([2., 5., 1., 6., 0., 0., 4.])
        rs0 = np.array([1, 2, 1, 4, 0, 1, 0])
        rs1 = np.array([3, 11, 4, 16, 3, 5, 4.5])

        self.assertTrue((mfblp_filter.roll_sum(x, m, 0) == rs0).all())
        self.assertTrue((mfblp_filter.roll_sum(x, m, 1) == rs1).all())

    def test_get_likelihood(self) :
        x = np.array([0.9, 0.91, 0.9, 2.05, 1.9, 3.1, 2.91])
        m1 = np.array([1.1, 1.1, 1.1, 2.1, 2.1, 3.1, 3.1])
        m2 = np.array([2.1, 2.1, 1.3, 1.1, 5.2, 1.1, 3.4])
        
        f = mfblp_filter(w=5, s=np.array([1., 2., 3.]), a=1., b=10.)
        l1 = f.get_likelihood(x, m1)
        l2 = f.get_likelihood(x, m2)

        self.assertTrue(l1.shape == x.shape)
        self.assertTrue((l1 < l2).all())
        self.assertTrue(l1.sum() < l2.sum())

    def test_search_w3(self) :
        x = np.array([10] * 100 + [20] * 100 + [15] * 100).astype(float)
        s = np.array([9.5, 10, 12, 15, 16, 19, 20, 21]).astype(float)
        starts = np.ones(x.shape)
        stepsize = 0.1
        steps = 300

        f = mfblp_filter(w=3, s=s, a=1., b=10.)
        opt_ms = f.search(x, starts, stepsize, steps)
        self.assertTrue((np.abs(opt_ms - x) < 0.1).all())

    def test_search_w5(self) :
        x = np.array([12] * 100 + [20] * 100 + [15] * 100).astype(float)
        s = np.array([9.5, 10, 12, 15, 16, 19, 20, 21]).astype(float)
        starts = np.ones(x.shape)
        stepsize = 0.1
        steps = 300

        f = mfblp_filter(w=5, s=s, a=.1, b=10.)
        opt_ms = f.search(x, starts, stepsize, steps)
        self.assertTrue((np.abs(opt_ms - x) < 0.1).all())

    def test_search_w7(self) :
        x = np.array([12] * 100 + [19] * 100 + [15] * 100).astype(float)
        s = np.array([9.5, 10, 12, 15, 16, 19, 20, 21]).astype(float)
        starts = np.ones(x.shape)
        stepsize = 0.1
        steps = 300

        f = mfblp_filter(w=7, s=s, a=.1, b=10.)
        opt_ms = f.search(x, starts, stepsize, steps)
        self.assertTrue((np.abs(opt_ms - x) < 0.1).all())

    def test_search_noisy_uniform(self) :
        x = np.array([12] * 100 + [19] * 100 + [15] * 100).astype(float)
        np.random.seed(42)
        x = x + np.random.uniform(low=-0.5, high=0.5)
        s = np.array([9.5, 10, 12, 15, 16, 19, 20, 21]).astype(float)
        starts = np.ones(x.shape)
        stepsize = 0.1
        steps = 300

        f = mfblp_filter(w=11, s=s, a=.1, b=10.)
        opt_ms = f.search(x, starts, stepsize, steps)
        self.assertTrue((np.abs(opt_ms - x) < 2.).all())

    def test_search_noisy_gaussian(self) :
        x = np.array([12] * 100 + [19] * 100 + [15] * 100).astype(float)
        np.random.seed(42)
        x = x + np.random.normal(0, .5, size=x.shape)
        s = np.array([9.5, 10, 12, 15, 16, 19, 20, 21]).astype(float)
        starts = np.ones(x.shape)
        stepsize = 0.1
        steps = 300

        f = mfblp_filter(w=15, s=s, a=.5, b=.5)
        opt_ms = f.search(x, starts, stepsize, steps)
        self.assertTrue((np.abs(opt_ms - x) < 3.).all())

    def test_apply(self) :
        x = np.array([12] * 100 + [20] * 100 + [15] * 100).astype(float)
        s = np.array([9.5, 10, 12, 15, 16, 19, 20, 21]).astype(float)

        f = mfblp_filter(w=5, s=s, a=.1, b=1.)
        m = f.apply(x)
        self.assertTrue((np.abs(m - x) < 0.1).all())

    def test_apply_noisy_uniform(self) :
        x = np.array([12] * 100 + [20] * 100 + [15] * 100).astype(float)
        s = np.array([9.5, 10, 12, 15, 16, 19, 20, 21]).astype(float)

        np.random.seed(42)
        x = x + np.random.uniform(low=-0.5, high=0.5)

        f = mfblp_filter(w=7, s=s, a=.1, b=.1)
        m = f.apply(x)
        self.assertTrue((np.abs(m - x) < 0.1).all())

    def test_apply_noisy_gaussian(self) :
        x = np.array([12] * 100 + [20] * 100 + [15] * 100).astype(float)
        s = np.array([10, 12, 15, 17, 20]).astype(float)

        np.random.seed(42)
        #x = x + np.random.normal(0, .1, size=x.shape)

        f = mfblp_filter(w=9, s=s, a=.1, b=10.)
        m = f.apply(x)
        self.assertTrue((np.abs(m - x) > 0.5).nonzero()[0].shape[0] < 5)

    def test_apply_restrict(self) :
        x = np.array([12] * 100 + [20] * 100 + [15] * 100).astype(float)
        s = np.array([9.5, 10, 12, 15, 16, 19, 20, 21]).astype(float)

        f = mfblp_filter(w=5, s=s, a=.1, b=1.)
        m = f.apply_restrict(x)
        self.assertTrue((np.abs(m - x) > 0.1).nonzero()[0].shape[0] < 3)
  

if __name__ == '__main__':
    unittest.main()