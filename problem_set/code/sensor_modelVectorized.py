'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 0.1
        self._z_short = 0.01
        self._z_max = 0.01
        self._z_rand = 100

        self._sigma_hit = 70
        self._lambda_short = 0.01

        self._max_range = 1000
        self._min_probability = 0.35
        self._subsampling = 2

        self._norm_wts = 1.0

        self.rows = occupancy_map.shape[0]
        self.cols = occupancy_map.shape[1]
        
        self.map = occupancy_map
    
    def p_hit(self, z_tkstar_arr, z_tk_arr):
        mask_idx1 = np.where(z_tk_arr > self._max_range)
        mask_idx2 = np.where(z_tk_arr < 0)
        mask_idx = np.hstack((mask_idx1, mask_idx2))
        phit = np.exp(-(z_tk_arr - z_tkstar_arr)**2 / (2 * (self._sigma_hit ** 2)))
        phit = phit / np.sqrt(2 * math.pi * (self._sigma_hit ** 2))
        phit[mask_idx] = 0 
        return phit
    
    def p_short(self, z_tkstar_arr, z_tk_arr):
        mask_idx1 = np.where(z_tk_arr > z_tkstar_arr)
        mask_idx2 = np.where(z_tk_arr < 0)
        mask_idx = np.hstack((mask_idx1, mask_idx2))
        eta = 1 / (1 - np.exp(-self._lambda_short * z_tkstar_arr))
        pshort = eta * self._lambda_short * np.exp(-self._lambda_short * z_tk_arr)
        pshort[mask_idx] = 0
        return pshort
    
    def p_max(self, z_tk_arr):
        idx = np.where(z_tk_arr == self._max_range)
        pmax = np.zeros(z_tk_arr.shape)
        pmax[idx] = 1.0
        return pmax
    
    def p_rand(self, z_tk_arr):
        prand = np.ones(z_tk_arr.shape)
        prand *= 1/self._max_range
        idx = z_tk_arr[z_tk_arr < 0]
        if len(idx):
            prand[idx] = 0.0
        idx = z_tk_arr[z_tk_arr >= self._max_range]
        if len(idx):
            prand[idx] = 0.0
        return prand

    def dist(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def rayCasting(self, x_t1, k, x_cord_laser, y_cord_laser):

        # net ray angle -> sum of bot angle and ray angle
        ray_angle = x_t1[-1] + math.radians(k)

        # initialize the initial and final positions of the laser as the initial coords
        init_coord_x, init_coord_y = x_cord_laser, y_cord_laser
        fin_coord_x, fin_coord_y = x_cord_laser, y_cord_laser

        while 0 < fin_coord_x < self.cols and 0 < fin_coord_y < self.rows and abs(self.map[fin_coord_y, fin_coord_x]) < self._min_probability:
            # update init coordinates, move in the direction of the beam
            init_coord_x += self._subsampling * np.cos(ray_angle)
            init_coord_y += self._subsampling * np.sin(ray_angle)
            # update final coordinates
            fin_coord_x, fin_coord_y = int(round(init_coord_x)), int(round(init_coord_y))
            if self.dist(x_cord_laser, y_cord_laser, fin_coord_x, fin_coord_y) * 10 > self._max_range:
                return self._max_range
        # compute the distance between the initial and the final position along the ray and multiply grid width
        return self.dist(x_cord_laser, y_cord_laser, fin_coord_x, fin_coord_y) * 10


    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        prob_zt1 = 0.0

        x_cord_bot, y_cord_bot, theta_bot = x_t1
        # check if the x coordinate is within range
        temp_x = min(int(x_cord_bot / 10), self.cols)
        temp_y = min(int(y_cord_bot / 10), self.rows)

        # check if the cell is occupied or if the measurnment is -1, if yes return 0 
        current_cell_val = self.map[temp_y, temp_x] 

        if current_cell_val == -1 or current_cell_val > self._min_probability:
            return 1e-100

        # get the laser coordinates
        x_cord_laser = x_cord_bot + 25.0 * np.cos(theta_bot)
        y_cord_laser = y_cord_bot + 25.0 * np.sin(theta_bot)

        # normalize and restrict to grid
        x_cord_laser = min(int(round(x_cord_laser / 10)), self.cols)
        y_cord_laser = min(int(round(y_cord_laser / 10)), self.rows)

        # get number of rays
        numRays = z_t1_arr.shape[0]

        z_tkstar_arr = []
        z_tk_arr = []

        # go over each ray
        for k in range(-int(numRays/2), int(numRays/2), 15):
            # get z*, using ray casting
            z_tkstar_arr.append(self.rayCasting(x_t1, k, x_cord_laser, y_cord_laser))
            # get the measured value of range finder
            z_tk_arr.append(z_t1_arr[k + int(numRays/2)])

        z_tkstar_arr = np.array(z_tkstar_arr)
        z_tk_arr = np.array(z_tk_arr)

        # compute the net probability from all 4 probabilities
        ph = self._z_hit * self.p_hit(z_tkstar_arr, z_tk_arr)
        ps = self._z_short * self.p_short(z_tkstar_arr, z_tk_arr)
        pm = self._z_max * self.p_max(z_tk_arr)
        pr = self._z_rand * self.p_rand(z_tk_arr)
        pNet =  ph + ps + pm + pr
        
        pNet = pNet[pNet > 0]
        # update net probability
        prob_zt1 = pNet.sum()

        return math.exp(prob_zt1)
