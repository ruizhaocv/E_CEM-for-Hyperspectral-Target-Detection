#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Oct 2018 @author: Rui ZHAO
The following code reproduces the experiments of the sections (sec 3.3, 3.4 and 3.5) of our paper:
R Zhao, Z Shi, Z Zou, Z Zhang, Ensemble-Based Cascaded Constrained Energy Minimization for Hyperspectral Target Detection. Remote Sensing 2019.

"""

import numpy as np
from utils import Detector


class Detectors(Detector):

    def __init__(self):
        Detector.__init__(self)

    def cem(self):
        # Basic implementation of the Constrained Energy Minimization (CEM) detector
        # Farrand, William H., and Joseph C. Harsanyi. "Mapping the distribution of mine tailings 
        # in the Coeur d'Alene River Valley, Idaho, through the use of a constrained energy minimization 
        # technique." Remote Sensing of Environment 59, no. 1 (1997): 64-76.
        size = self.img.shape   # get the size of image matrix
        R = np.dot(self.img, self.img.T/size[1])   # R = X*X'/size(X,2);
        w = np.dot(np.linalg.inv(R), self.tgt)  # w = (R+lamda*eye(size(X,1)))\d ;
        result = np.dot(w.T, self.img).T  # y=w'* X;
        return result

    def ace(self):
        # Basic implementation of the Adaptive Coherence/Cosine Estimator (ACE)
        # Manolakis, Dimitris, David Marden, and Gary A. Shaw. "Hyperspectral image processing for 
        # automatic target detection applications." Lincoln laboratory journal 14, no. 1 (2003): 79-116.
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
        # Basic implementation of the Matched Filter (MF)
        # Manolakis, Dimitris, Ronald Lockwood, Thomas Cooley, and John Jacobson. "Is there a best hyperspectral 
        # detection algorithm?." In Algorithms and technologies for multispectral, hyperspectral, and ultraspectral 
        # imagery XV, vol. 7334, p. 733402. International Society for Optics and Photonics, 2009.
        size = self.img.shape
        a = np.mean(self.img)
        k = (self.img-a).dot((self.img-a).T)/size[1]
        w = np.linalg.inv(k).dot(self.tgt-a)
        result = w.T.dot(self.img-a)
        return result.T

    def sid(self):
        # Basic implementation of the Spectral Information Divergence (SID) detector
        # Chang, Chein-I. "An information-theoretic approach to spectral variability, similarity, and discrimination 
        # for hyperspectral image analysis." IEEE Transactions on information theory 46, no. 5 (2000): 1927-1932.
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
        # Basic implementation of the Spectral Angle Mapper (SAM)
        # Kruse, Fred A., A. B. Lefkoff, J. W. Boardman, K. B. Heidebrecht, A. T. Shapiro, P. J. Barloon, 
        # and A. F. H. Goetz. "The spectral image processing system (SIPS)—interactive visualization and analysis 
        # of imaging spectrometer data." Remote sensing of environment 44, no. 2-3 (1993): 145-163.
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
