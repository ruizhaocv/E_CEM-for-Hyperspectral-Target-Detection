#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Oct 2018 @author: Rui ZHAO
The following code builds the E-CEM detector of our paper:
R Zhao, Z Shi, Z Zou, Z Zhang, Ensemble-Based Cascaded Constrained Energy Minimization for Hyperspectral Target Detection. Remote Sensing 2019.

"""

import numpy as np
import random
import multiprocessing
from utils import Detector, dual_sigmoid


class ECEM(Detector):

    def __init__(self):
        Detector.__init__(self)
        self.windowsize = [1/4, 2/4, 3/4, 4/4]  # window size
        self.num_layer = 10  # the number of detection layers
        self.num_cem = 6  # the number of CEMs per layer
        self.Lambda = 1e-6  # the regularization coefficient
        self.show_proc = True  # show the process or not

    def parmset(self, **parm):
        self.windowsize = parm['windowsize']  # parameters
        self.num_layer = parm['num_layer']
        self.num_cem = parm['num_cem']
        self.Lambda = parm['Lambda']
        self.show_proc = parm['show_proc']

    def setlambda(self):
        switch = {
            'san': 1e-6,
            'san_noise': 6e-2,
            'syn_noise': 5e-3,
            'cup': 1e-1
        }
        if self.name in switch:
            return switch[self.name]
        else:
            return 1e-10

    def cem(self, img, tgt):
        size = img.shape   # get the size of image matrix
        lamda = random.uniform(self.Lambda/(1+self.Lambda), self.Lambda)  # random regularization coefficient
        R = np.dot(img, img.T/size[1])   # R = X*X'/size(X,2);
        w = np.dot(np.linalg.inv((R+lamda*np.identity(size[0]))), tgt)  # w = (R+lamda*eye(size(X,1)))\d ;
        result = np.dot(w.T, img)  # y=w'* X;
        return result

    def ms_scanning_unit(self, winowsize):
        d = self.img.shape[0]
        winlen = int(d*winowsize**2)
        size = self.imgt.shape  # get the size of image matrix
        result = np.zeros(shape=(int((size[0]-winlen+1)/2+1), size[1]))
        pos = 0
        if self.show_proc: print('Multi_Scale Scanning: size of window: %d' % winlen)
        for i in range(0, size[0]-winlen+1, 2):  # multi-scale scanning
            imgt_tmp = self.imgt[i:i+winlen-1, :]
            result[pos, :] = self.cem(imgt_tmp, imgt_tmp[:, -1])
            pos += 1
        return result

    def cascade_detection(self, mssimg):   # defult parameter configuration
        size = mssimg.shape
        result_forest = np.zeros(shape=(self.num_cem, size[1]))
        for i_layer in range(self.num_layer):
            if self.show_proc: print('Cascaded Detection layer: %d' % i_layer)  # show the process of cascade detection
            for i_num in range(self.num_cem):
                result_forest[i_num,:] = self.cem(mssimg, mssimg[:, -1])
            weights = dual_sigmoid(np.mean(result_forest, axis=0))  # sigmoid nonlinear function
            mssimg = mssimg*weights
        result = result_forest[:, 0:-1]
        return result

    def detect(self, img_data, pool_num=4):
        self.load_data(img_data)
        self.imgt = np.hstack((self.img, self.tgt))
        p = multiprocessing.Pool(pool_num)  # multiprocessing
        results = p.map(self.ms_scanning_unit, self.windowsize)  # Multi_Scale Scanning
        p.close()
        p.join()
        mssimg = np.concatenate(results, axis=0)
        cadeimg = self.cascade_detection(mssimg)  # Cascaded Detection
        result = np.mean(cadeimg, axis=0)[:self.imgt.shape[1]].reshape(-1, 1)
        return result
