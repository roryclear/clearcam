# https://github.com/noahcao/OC_SORT
from __future__ import absolute_import, division

from copy import deepcopy
from math import log
import sys
import numpy as np
from numpy import dot, zeros, eye, isscalar

class KalmanFilterNew(object):
    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))        # state
        self.P = eye(dim_x)               # uncertainty covariance
        self.Q = eye(dim_x)               # process uncertainty
        self.B = None                     # control transition matrix
        self.F = eye(dim_x)               # state transition matrix
        self.H = zeros((dim_z, dim_x))    # measurement function
        self.R = eye(dim_z)               # measurement uncertainty
        self._alpha_sq = 1.               # fading memory control
        self.M = np.zeros((dim_x, dim_z)) # process-measurement cross correlation
        self.z = np.array([[None]*self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z)) # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty
        self.SI = np.zeros((dim_z, dim_z)) # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # keep all observations 
        self.history_obs = []

        self.inv = np.linalg.inv

        self.attr_saved = None
        self.observed = False 

    def predict(self):
        self.x = dot(self.F, self.x)
        self.P = self._alpha_sq * dot(dot(self.F, self.P), self.F.T) + self.Q

    def unfreeze(self):
        occur = np.array([d is None for d in self.history_obs], dtype=int)
        indices = np.where(occur == 0)[0]
        index1, index2 = indices[-2], indices[-1]
        x1, y1, s1, r1 = self.history_obs[index1]
        x2, y2, s2, r2 = self.history_obs[index2]
        w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
        w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)
        time_gap = index2 - index1
        t = np.arange(1, time_gap + 1)
        x = x1 + t * (x2 - x1) / time_gap
        y = y1 + t * (y2 - y1) / time_gap
        w = w1 + t * (w2 - w1) / time_gap
        h = h1 + t * (h2 - h1) / time_gap
        s = w * h
        r = w / h
        boxes = np.stack([x, y, s, r], axis=1).reshape(-1, 4, 1)
        self.__dict__ = self.attr_saved
        for i in range(time_gap):
            self.update2(boxes[i])
            if i < time_gap - 1:
                self.x = self.F @ self.x
                self.P = self._alpha_sq * (self.F @ self.P @ self.F.T) + self.Q


    def update2(self, z):
        self.history_obs.append(z)
        self.observed = True
        R = self.R
        H = self.H
        self.y = z - dot(H, self.x)
        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        self.K = dot(PHT, self.SI)
        self.x = self.x + dot(self.K, self.y)
        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

    def update(self, z, R=None, H=None):
        self.history_obs.append(z)
        if z is None:
            if self.observed:
                self.attr_saved = deepcopy(self.__dict__)
            self.observed = False 
            self.z = np.array([[None]*self.dim_z]).T
            self.y = zeros((self.dim_z, 1))
            return
        if not self.observed and self.attr_saved: self.unfreeze()
        self.observed = True
        R = self.R
        H = self.H
        self.y = z - dot(H, self.x)
        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        self.K = dot(PHT, self.SI)
        self.x = self.x + dot(self.K, self.y)
        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)
        