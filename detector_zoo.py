#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Oct 2018 @author: Rui ZHAO
The following code builds the classical detectors used in the experiment sections (sec 3.3, 3.4 and 3.5) of our paper:
R Zhao, Z Shi, Z Zou, Z Zhang, Ensemble-Based Cascaded Constrained Energy Minimization for Hyperspectral Target Detection. Remote Sensing 2019.

"""

import numpy as np
from utils import Detector


class Detectors(Detector):

    def __init__(self):
        Detector.__init__(self)

    def cem(self):
        size = self.img.shape   # get the size of image matrix
        R = np.dot(self.img, self.img.T/size[1])   # R = X*X'/size(X,2);
        w = np.dot(np.linalg.inv(R), self.tgt)  # w = (R+lamda*eye(size(X,1)))\d ;
        result = np.dot(w.T, self.img).T  # y=w'* X;
        return result

    def ace(self):
        size = self.img.shape
        img_mean = np.mean(self.img, axis=1)[:, np.newaxis]
        img0 = self.data.img-img_mean.dot(np.ones((1, size[1])))
        R = img0.dot(img0.T)/size[1]
        y0 = (self.tgt-img_mean).T.dot(np.linalg.inv(R)).dot(img0)**2
        y1 = (self.tgt-img_mean).T.dot(np.linalg.inv(R)).dot(self.tgt-img_mean)
        y2 = (img0.T.dot(np.linalg.inv(R))*(img0.T)).sum(axis=1)[:, np.newaxis]
        result = y0/(y1*y2).T
        return result.T

    def mf(self):
        size = self.img.shape
        a = np.mean(self.img)
        k = (self.img-a).dot((self.img-a).T)/size[1]
        w = np.linalg.inv(k).dot(self.tgt-a)
        result = w.T.dot(self.img-a)
        return result.T

    def sid(self):
        size = self.img.shape
        result = np.zeros((1, size[1]))
        for i in range(size[1]):
            pi = (self.img[:, i]/(self.img[:, i].sum())).reshape(-1, 1)+1e-20
            di = self.tgt/(self.tgt.sum())+1e-20
            sxd = (pi*np.log(abs(pi/di))).sum()
            sdx = (di*np.log(abs(di/pi))).sum()
            result[:, i] = 1/(sxd + sdx)/size[1]
        return result.T

    def sam(self):
        size = self.img.shape
        ld = np.sqrt(self.tgt.T.dot(self.tgt))
        result = np.zeros((1, size[1]))
        for i in range(size[1]):
            x = self.img[:, i]
            lx = np.sqrt(x.T.dot(x))
            cos_angle = x.dot(self.tgt)/(lx*ld)
            angle = np.arccos(cos_angle)
            result[:, i] = 1/abs(angle)
        return result.T

    def detect(self, img_data):
        self.load_data(img_data)
        return {'CEM': self.cem, 'ACE': self.ace, 'MF': self.mf, 'SID': self.sid, 'SAM': self.sam}
