'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """
        self.xt = None

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  []

        # num samples: M
        M = X_bar.shape[0]

        # get r, random num between 0, M-1
        sampler_r = np.random.uniform(0, 1 / M)

        # get the importance weights and normalize them
        Wt = X_bar[:, -1]
        Wt = Wt / Wt.sum()
        
        # get the first importance weight
        C = Wt[0]

        idx = 0

        # go through each particle
        for m in range(M):
            # get the current sample position
            U = sampler_r + (m) * (1/M)

            # while current position is greater than importance weight
            while U > C:
                # increment index and add new weight to C
                idx += 1
                C += Wt[idx]

            # add the particle to the resampled array
            X_bar_resampled.append(X_bar[idx])
        
        X_bar_resampled = np.array(X_bar_resampled)
        return X_bar_resampled
