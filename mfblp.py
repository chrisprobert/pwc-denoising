from __future__ import division

import numpy as np
from scipy.ndimage.filters import median_filter

class mfblp_filter(object) :

    def __init__(self, w, s, a, b) :
        '''
        parameters:
            w - int; must be odd; window width
            s - ndarray; laplace prior locations
            a - float; strength of MAE vs. LP terms
            b - float; spread of laplace prior term
        '''
        if (w % 2) != 1 :
            raise ValueError('window width parameter w must be odd')
        self.hw = w // 2 # half window width
        self.s = s.flatten()
        self.a = a
        self.b = b

    def apply(self, x, numiter=200) :
        '''
        Apply the filter by searching entire range of S and M_{i}. Note that
        apply_restrict() is significantly faster at the same MAE.

        paramters:
            x - ndarray; sparse signal to be filtered
            numiter - int (optional); maximum number of search iterations

        returns :
            m - ndarray; best filtered signal
        '''
        steps = numiter // 2
        maxv = max(x.max(), self.s.max())
        minv = min(x.min(), self.s.min())
        first_stepsize = (maxv - minv) / steps
        first_starts = np.ones(x.shape) * minv
        second_stepsize = (first_stepsize * 2) / steps

        # first round, course search
        m = self.search(x, first_starts, first_stepsize, steps)
        # second round, fine search
        m = self.search(x, m - first_stepsize, second_stepsize, steps)
        return m
    
    def apply_restrict(self, x, numiter=20) :
        '''
            Applies the filter, but only searches output points between the median
            of each window and the closest point in s. This search space is 
            smaller than the full range in apply(), and covers the expected
            filter output for any valid self.a mixture paramter. If the signal is
            equidistant from two points in S, selection is performed according to
            numpy.argmin, which defaults to the value with the lowest index (so,
            if S is sorted, default behavior for equidistant S points is to
            select the lesser value). On benchmarking for the same MAE, this is
            around 10X faster than the apply() method (but compartive speed
            depends on density of points in S relative to M).

            paramters:
                x - ndarray; sparse signal to be filtered
                numiter - int (optional); maximum number of search iterations

            returns :
                m - ndarray; best filtered signal
        '''
        medians = median_filter(x, self.hw * 2)
        median_dists = np.abs(medians[:,None] - self.s)
        closest_s_indexes = np.argmin(median_dists, axis=1)
        closest_s = self.s[closest_s_indexes]
        
        starts = np.zeros(x.shape)
        closest_s_smaller = closest_s < medians
        starts[closest_s_smaller] = closest_s[closest_s_smaller]
        starts[closest_s_smaller == False] = medians[closest_s_smaller == False]

        steps = numiter // 2
        stepsizes = np.abs(closest_s - medians)
        first_stepsize = stepsizes / steps
        second_stepsize = (first_stepsize * 2) / steps

        # first round, course search
        m = self.search(x, starts, first_stepsize, steps)
        # second round, fine search
        m = self.search(x, m - first_stepsize, second_stepsize, steps)
        return m

    @classmethod
    def roll_sum(cls, x, m, hw) :
        '''return an array of the cummulative absolute errors for the point
        estimate m of x_{w} for each window [x-hw, x+hw]'''
        sums = sum((np.abs(np.roll(x, i) - m) for i in xrange(-hw, hw + 1)))

        # continuity correction for edge cases - remove error terms from opposite
        # end, and scale error based on window size
        for i in xrange(hw) :
            mfac = (2 * hw + 1) / (hw + i + 1)
            sums[i] = sum(np.abs(x[:i+hw+1] - m[i])) * mfac
            sums[-i-1] = sum(np.abs(x[-i-hw-1:] - m[-i-1])) * mfac

        return sums

    def get_likelihood(self, x, m) :
        '''return the negative log likelihood for mae-blp for estimate m
        with window size w'''
        difs = self.a * self.roll_sum(x, m, self.hw)
        assert(difs.shape == x.shape)
        prior = np.log(np.exp(-1.0 * self.b * np.abs(m[:,None] - self.s)).sum(axis=1))
        assert(prior.shape == x.shape)
        return difs - prior

    def search(self, x, starts, stepsize, steps) :
        '''return optimal m from grid search given starting position and stepsize'''
        opt_ms = starts.copy()
        opt_ls = self.get_likelihood(x, starts)
        assert(opt_ms.shape == x.shape)
        assert(opt_ls.shape == x.shape)
        for iter in xrange(steps) :
            starts += stepsize
            ls = self.get_likelihood(x, starts)
            replace = ls < opt_ls
            opt_ms[replace] = starts[replace]
            opt_ls[replace] = ls[replace]
        return opt_ms
