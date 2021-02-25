'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.0001
        self._alpha2 = 0.0001
        self._alpha3 = 0.01
        self._alpha4 = 0.01

    # triangular distribution
    # def sample(self, b2):
    #     b    = np.sqrt(b2)
    #     val  = np.sqrt(6)/2
    #     val *= (np.random.uniform(0, b) + np.random.uniform(0, b))
    #     return val

    def warp(self, theta):
        if theta > math.pi:
            theta -= math.pi
        if theta < -math.pi:
            theta += math.pi
        return theta

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """

        # get the odometer readings at t and t-1
        xt_1, yt_1, thetat_1  = u_t1[0], u_t1[1], u_t1[2]
        x_t0_, yt_0, thetat_0 = u_t0[0], u_t0[1], u_t0[2]
        x0, y0, theta0        = x_t0[0], x_t0[1], x_t0[2]

        # get the rotation and translation
        delRot_1 = np.arctan2((yt_1 - yt_0), (xt_1 -  x_t0_)) - thetat_0
        delRot_2 = thetat_1 - thetat_0 - delRot_1
        # print(delRot_1, delRot_2)
        delTrans = np.sqrt((xt_1 - x_t0_)**2 + (yt_1 - yt_0)**2)

        # create some sampling noise
        b1 = self._alpha1 * (delRot_1 ** 2) + self._alpha2 * (delTrans ** 2)
        b2 = self._alpha3 * (delTrans ** 2) + self._alpha4 * (delRot_1 ** 2) + self._alpha4 * (delRot_2 ** 2)
        b3 = self._alpha1 * (delRot_2 ** 2) + self._alpha2 * (delTrans ** 2)

        # add some noise to the original estimate
        delRoth_1 = delRot_1 - np.random.normal(0, np.sqrt(b1))
        delTransh = delTrans - np.random.normal(0, np.sqrt(b2))
        delRoth_2 = delRot_2 - np.random.normal(0, np.sqrt(b3))

        # get the new state estimate
        xhat = x0 + delTransh * np.cos(theta0 + delRoth_1)
        yhat = y0 + delTransh * np.sin(theta0 + delRoth_1)
        thetah = theta0 + delRoth_1 + delRoth_2

        x_t = np.array([xhat, yhat, thetah])
        
        return x_t
