#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Feb 2019 @author: Rui ZHAO
The following code corresponds to the experiment sections (sec 3.3, 3.4 and 3.5) of our paper:
R Zhao, Z Shi, Z Zou, Z Zhang, Ensemble-Based Cascaded Constrained Energy Minimization for Hyperspectral Target Detection. Remote Sensing 2019.

"""


from utils import Data
from e_cem import ECEM
from detector_zoo import Detectors as Other_detectors


class Exp(object):

    def __init__(self):
        self.data = []

    def san(self):
        self.data = Data('hyperspectral_data//san.mat')  # load data
        ecem = ECEM()
        ecem.parmset(**{'windowsize': [1 / 4, 2 / 4, 3 / 4, 4 / 4],  # window size
                        'num_layer': 10,  # the number of detection layers
                        'num_cem': 6,  # the number of CEMs per layer
                        'Lambda': 1e-6,  # the regularization coefficient
                        'show_proc': True})  # show the process or not
        return ecem

    def san_noise(self, snr=20):
        self.data = Data('hyperspectral_data//san.mat', snr)  # load data with white Gaussian noise (SNR=20 or 25)
        ecem = ECEM()
        ecem.parmset(**{'windowsize': [1 / 4, 2 / 4, 3 / 4, 4 / 4],  # window size
                        'num_layer': 10,  # the number of detection layers
                        'num_cem': 6,  # the number of CEMs per layer
                        'Lambda': 6e-2,  # the regularization coefficient
                        'show_proc': True})  # show the process or not
        return ecem

    def syn_noise(self, snr=20):
        ecem = ECEM()
        ecem.parmset(**{'windowsize': [1 / 4, 2 / 4, 3 / 4, 4 / 4],  # window size
                        'num_layer': 10,  # the number of detection layers
                        'num_cem': 6,  # the number of CEMs per layer
                        'Lambda': 5e-3,  # the regularization coefficient
                        'show_proc': True})  # show the process or not
        return ecem

    def cup(self):
        self.data = Data('hyperspectral_data//cup.mat')  # load data with noise
        ecem = ECEM()
        ecem.parmset(**{'windowsize': [1 / 4, 2 / 4, 3 / 4, 4 / 4],  # window size
                        'num_layer': 10,  # the number of detection layers
                        'num_cem': 6,  # the number of CEMs per layer
                        'Lambda': 1e-1,  # the regularization coefficient
                        'show_proc': True})  # show the process or not
        return ecem


def main():
    exp = Exp()
    ecem = exp.san()  # chose the experiment: san, san_noise, syn_noise or cup
    results = []
    names = []
    for name, detector in Other_detectors().detect(exp.data).items():  # dectection
        print('detector:' + name)
        results.append(detector())
        names.append(name)
    print('detector:' + 'E-CEM')
    results.append(ecem.detect(exp.data, pool_num=4))  # dectection
    names.append('E-CEM')
    ecem.show(results, names)  # show


if __name__ == '__main__':
    main()
