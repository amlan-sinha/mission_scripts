import sys, os

sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding/debug')
sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding')

import pydylan
import numpy as np

from lineplot import LinePlot
import matplotlib.pyplot as plt
from support import html_colors

strongorange, babyblue = html_colors['strong orange'], html_colors['baby blue']

days_to_seconds, seconds_to_days = 86400., 1/86400.


def p1_initial_guess():
    return np.array([2.43850607e+01, 4.50000000e-01, 0.00000000e+00, 4.67521084e+00,
       0.00000000e+00, 1.31539122e-03, 0.00000000e+00, 6.14024654e+00,
       2.24797318e-03, 3.84438363e+00, 3.14245844e+00, 5.53767614e-01,
       2.36002242e+00, 0.00000000e+00, 0.00000000e+00, 6.28318531e+00,
       5.84514646e+00, 0.00000000e+00, 5.43243259e+00, 3.44559780e+00,
       0.00000000e+00, 0.00000000e+00, 2.48393842e+00, 0.00000000e+00,
       0.00000000e+00, 3.43105873e+00, 0.00000000e+00, 6.28318531e+00,
       6.28318531e+00, 0.00000000e+00, 3.68707373e+00, 3.90244526e-01,
       0.00000000e+00, 9.33580618e-01, 2.96914496e+00, 6.73469192e-04,
       4.21969158e+00, 0.00000000e+00, 1.00000000e+00, 3.36110946e+00,
       0.00000000e+00, 0.00000000e+00, 3.25279477e-02, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 6.28318531e+00, 6.78748668e-02,
       0.00000000e+00, 5.97435595e+00, 0.00000000e+00, 6.28318531e+00,
       9.71157342e-01, 0.00000000e+00, 9.46990008e-01, 0.00000000e+00,
       0.00000000e+00, 9.49280953e-01, 0.00000000e+00, 0.00000000e+00,
       4.35038739e+00, 5.42859402e+00, 0.00000000e+00, 1.97674985e+00,
       6.28318531e+00, 4.02913259e-01, 2.62088004e+00, 6.28318531e+00,
       0.00000000e+00, 6.28318531e+00, 6.28318531e+00, 0.00000000e+00,
       3.86339513e+00, 5.42245223e+00, 0.00000000e+00, 6.28318531e+00,
       3.14659678e+00, 0.00000000e+00, 2.35414532e+00, 0.00000000e+00,
       0.00000000e+00, 1.15367418e+00, 2.54576806e+00, 0.00000000e+00,
       2.00497573e-01, 3.92492495e+00, 0.00000000e+00, 6.28318531e+00,
       3.17891882e+00, 0.00000000e+00, 5.51580516e+00, 6.28318531e+00,
       0.00000000e+00, 4.56941715e+00, 0.00000000e+00, 0.00000000e+00,
       4.70101290e+00, 1.75023556e+00, 0.00000000e+00, 6.28318531e+00,
       0.00000000e+00, 0.00000000e+00, 6.28318531e+00, 5.74138179e+00,
       0.00000000e+00, 0.00000000e+00, 6.28318531e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.24289861e+00,
       5.05330231e-01, 0.00000000e+00, 2.18992600e+00, 0.00000000e+00,
       0.00000000e+00, 2.62399111e+00, 6.28318531e+00, 0.00000000e+00,
       6.12832149e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.81560283e-01,
       0.00000000e+00, 4.37758002e+00, 6.28318531e+00, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 0.00000000e+00, 4.85275581e+00,
       6.28318531e+00, 0.00000000e+00, 0.00000000e+00, 1.95181812e+00,
       0.00000000e+00, 6.28318531e+00, 0.00000000e+00, 0.00000000e+00,
       2.86372704e+00, 4.44064259e+00, 0.00000000e+00, 5.78316821e+00,
       3.94798445e+00, 0.00000000e+00, 6.28318531e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       6.15770404e-01, 6.28318531e+00, 0.00000000e+00, 4.36420296e-01,
       6.28318531e+00, 7.03709657e-01, 0.00000000e+00, 2.62580805e+00,
       0.00000000e+00, 6.28318531e+00, 3.98780092e+00, 0.00000000e+00,
       5.01016532e+00, 6.28318531e+00, 9.99622135e-01, 4.63987460e+00,
       6.28318531e+00, 0.00000000e+00, 0.00000000e+00, 5.68951140e+00,
       0.00000000e+00, 4.62176443e+00, 0.00000000e+00, 0.00000000e+00,
       3.84587184e+00, 6.28318531e+00, 0.00000000e+00, 6.28318531e+00,
       0.00000000e+00, 0.00000000e+00, 5.62514293e+00, 3.66149353e+00,
       0.00000000e+00, 0.00000000e+00, 3.14161474e+00, 9.99781964e-01,
       0.00000000e+00, 4.33355793e+00, 0.00000000e+00, 1.27272243e-01,
       0.00000000e+00, 0.00000000e+00, 7.26918346e-01, 5.93957093e-02,
       0.00000000e+00, 0.00000000e+00, 6.28318531e+00, 0.00000000e+00,
       0.00000000e+00, 2.49572033e+00, 0.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 0.00000000e+00, 6.28318531e+00, 6.28318531e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       4.78457390e+00, 4.90784252e+00, 0.00000000e+00, 3.61739073e+00,
       5.01349296e+00, 0.00000000e+00, 2.33128659e+00, 5.25841071e+00,
       0.00000000e+00, 2.15591288e+00, 6.28318531e+00, 6.61953640e-04,
       2.62018453e+00, 0.00000000e+00, 0.00000000e+00, 5.27015102e+00,
       0.00000000e+00, 0.00000000e+00, 1.66442810e+00, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 6.28318531e+00, 0.00000000e+00,
       6.28318531e+00, 0.00000000e+00, 0.00000000e+00, 7.19974223e-01,
       0.00000000e+00, 0.00000000e+00, 4.27936056e+00, 3.04231461e+00,
       0.00000000e+00, 2.96103759e+00, 8.18774451e-02, 0.00000000e+00,
       4.91826451e+00, 6.28318531e+00, 0.00000000e+00, 5.24522123e+00,
       0.00000000e+00, 0.00000000e+00, 6.28318531e+00, 2.69852383e+00,
       0.00000000e+00, 0.00000000e+00, 1.66372209e-02, 0.00000000e+00,
       3.79570907e+00, 4.49274528e+00, 0.00000000e+00, 1.24983080e+00,
       1.52265772e+00, 0.00000000e+00, 0.00000000e+00, 4.14531399e+00,
       0.00000000e+00, 6.28318531e+00, 3.37916690e+00, 0.00000000e+00,
       4.95680144e+00, 3.91424939e+00, 0.00000000e+00, 4.88348633e+00,
       6.28318531e+00, 0.00000000e+00, 0.00000000e+00, 5.82255269e+00,
       0.00000000e+00, 1.15898016e+00, 2.44347500e-01, 0.00000000e+00,
       4.77985475e+00, 0.00000000e+00, 0.00000000e+00, 6.28318531e+00,
       0.00000000e+00, 0.00000000e+00, 2.21230969e+00, 2.00141906e+00,
       0.00000000e+00, 2.82735917e+00, 0.00000000e+00, 0.00000000e+00,
       6.21082164e+00, 6.28318531e+00, 8.41291971e-01, 2.46979961e+00])


def p2_initial_guess():
    return np.array([3.04212271e+00, 0.00000000e+00, 9.91939757e-01, 4.68212434e-02,
       2.53829630e-13, 1.00000000e+00, 7.00806117e-02, 6.25974581e+00,
       1.00000000e+00, 4.52960416e-02, 4.94141266e-02, 1.00000000e+00,
       1.63150877e-02, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       1.00000000e+00, 5.70963448e+00, 0.00000000e+00, 1.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       6.31742649e-01, 0.00000000e+00, 8.41351603e-02, 1.00000000e+00,
       8.90575917e-01, 1.40427203e+00, 0.00000000e+00, 0.00000000e+00,
       4.52307892e-02, 1.00000000e+00, 0.00000000e+00, 6.28318531e+00,
       1.00000000e+00, 0.00000000e+00, 6.28318531e+00, 1.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 4.64172067e+00,
       6.28318531e+00, 1.00000000e+00, 0.00000000e+00, 6.28318531e+00,
       8.19608521e-01, 4.49629456e+00, 6.28318531e+00, 1.00000000e+00,
       0.00000000e+00, 4.13480351e-02, 1.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 9.14484813e-01, 0.00000000e+00, 6.28318531e+00,
       1.00000000e+00, 5.36193898e+00, 0.00000000e+00, 1.00000000e+00,
       9.91074682e-02, 4.21824426e+00, 1.00000000e+00, 0.00000000e+00,
       2.23212642e-01, 1.00000000e+00, 6.28318531e+00, 0.00000000e+00,
       3.35092055e-01, 0.00000000e+00, 6.28318531e+00, 0.00000000e+00,
       3.35828106e+00, 2.58495914e+00, 5.26234377e-01, 4.13117802e+00,
       4.09028939e-01, 2.36137207e-02, 0.00000000e+00, 5.04518406e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 0.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 1.00000000e+00, 8.05553044e-01, 3.28111499e+00,
       1.00000000e+00, 4.63037421e+00, 6.28318531e+00, 7.86811114e-01,
       0.00000000e+00, 9.88566994e-01, 1.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 8.84188556e-15, 4.45095350e+00, 6.28318531e+00,
       9.97454963e-01, 4.20732771e+00, 6.28318531e+00, 9.47603530e-01,
       3.70142680e+00, 6.28318531e+00, 1.00000000e+00, 1.61193286e+00,
       0.00000000e+00, 1.00000000e+00, 1.49994516e-01, 0.00000000e+00,
       2.43523637e-01, 0.00000000e+00, 6.28318531e+00, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 0.00000000e+00, 1.93054062e+00,
       0.00000000e+00, 0.00000000e+00, 6.28318531e+00, 3.56787586e+00,
       1.00000000e+00, 0.00000000e+00, 6.28318531e+00, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 0.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 0.00000000e+00, 0.00000000e+00, 6.28318531e+00,
       3.71371504e-02, 2.42234362e+00])


def p3_initial_guess():
    return np.array([2.32998330e+01, 2.00000000e+00, 1.36727831e+00, 0.00000000e+00,
       3.78321540e+00, 0.00000000e+00, 3.84147250e+00, 6.28318531e+00,
       1.00000000e+00, 3.68166793e+00, 6.28318531e+00, 1.00000000e+00,
       3.51364607e+00, 0.00000000e+00, 7.38998569e-01, 0.00000000e+00,
       6.58566220e-01, 0.00000000e+00, 3.21042709e+00, 6.28318531e+00,
       0.00000000e+00, 3.16468452e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 3.54356310e+00, 0.00000000e+00, 0.00000000e+00,
       4.30818470e-01, 0.00000000e+00, 6.03385645e+00, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 0.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 2.46284052e-01, 0.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 4.89620580e+00, 5.30812919e+00, 0.00000000e+00,
       1.53553482e+00, 6.28318531e+00, 0.00000000e+00, 6.28318531e+00,
       4.99429454e+00, 0.00000000e+00, 0.00000000e+00, 6.28318531e+00,
       0.00000000e+00, 6.28318531e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 0.00000000e+00, 2.59333192e+00,
       8.11751272e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 6.28318531e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       5.11295446e+00, 6.28318531e+00, 0.00000000e+00, 0.00000000e+00,
       2.56195263e+00, 0.00000000e+00, 5.33508983e+00, 2.23417341e+00,
       0.00000000e+00, 0.00000000e+00, 6.28318531e+00, 0.00000000e+00,
       6.28318531e+00, 4.71181589e+00, 0.00000000e+00, 0.00000000e+00,
       5.95870877e+00, 0.00000000e+00, 1.40400020e+00, 6.28318531e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 0.00000000e+00, 0.00000000e+00, 3.04782231e+00,
       6.28318531e+00, 0.00000000e+00, 1.23464485e+00, 0.00000000e+00,
       0.00000000e+00, 4.07319989e+00, 4.93405731e+00, 0.00000000e+00,
       1.06108328e+00, 0.00000000e+00, 0.00000000e+00, 6.28318531e+00,
       6.28318531e+00, 0.00000000e+00, 1.91876045e+00, 6.28318531e+00,
       0.00000000e+00, 3.60524169e+00, 6.28318531e+00, 0.00000000e+00,
       0.00000000e+00, 1.73801388e+00, 0.00000000e+00, 1.57554865e+00,
       6.28318531e+00, 0.00000000e+00, 1.13510037e-02, 5.82158402e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       2.79293782e-01, 6.25653892e+00, 0.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 0.00000000e+00, 3.13371104e+00, 3.08161736e+00,
       0.00000000e+00, 0.00000000e+00, 6.28318531e+00, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 0.00000000e+00, 5.23526232e+00,
       3.14155515e+00, 1.00000000e+00, 4.42295716e+00, 6.28318531e+00,
       0.00000000e+00, 2.34430302e+00, 0.00000000e+00, 1.00000000e+00,
       6.01396159e+00, 3.28175517e+00, 0.00000000e+00, 2.63242570e+00,
       6.28318531e+00, 6.49414751e-01, 0.00000000e+00, 4.57075196e+00,
       0.00000000e+00, 1.03194314e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 2.08485617e-01, 0.00000000e+00, 1.41474514e+00,
       0.00000000e+00, 0.00000000e+00, 6.28318531e+00, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 4.23314195e+00, 0.00000000e+00,
       4.01241105e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       2.10344883e-01, 0.00000000e+00, 6.28318531e+00, 2.82028145e+00,
       0.00000000e+00, 4.61243757e+00, 6.28318531e+00, 0.00000000e+00,
       0.00000000e+00, 2.77731082e+00, 0.00000000e+00, 1.81084600e+00,
       0.00000000e+00, 0.00000000e+00, 4.54652845e+00, 3.88133077e+00,
       0.00000000e+00, 6.28318531e+00, 6.28318531e+00, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 0.00000000e+00, 3.26783871e+00,
       5.46544249e+00, 0.00000000e+00, 6.28318531e+00, 6.28318531e+00,
       0.00000000e+00, 6.28318531e+00, 3.61702574e+00, 0.00000000e+00,
       0.00000000e+00, 1.26713248e-01, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 5.92835353e-01, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 6.28318531e+00, 0.00000000e+00,
       6.28318531e+00, 0.00000000e+00, 0.00000000e+00, 1.29965392e+00,
       0.00000000e+00, 0.00000000e+00, 4.54811776e+00, 3.12934587e+00,
       0.00000000e+00, 6.28318531e+00, 5.80032435e+00, 0.00000000e+00,
       0.00000000e+00, 3.05673885e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.87551851e+00,
       0.00000000e+00, 0.00000000e+00, 5.35108640e+00, 0.00000000e+00,
       4.11563749e+00, 0.00000000e+00, 0.00000000e+00, 6.28318531e+00,
       4.50554302e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 3.17840302e+00, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 0.00000000e+00, 6.87100847e-01,
       0.00000000e+00, 0.00000000e+00, 5.80918069e+00, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 5.86208027e+00, 0.00000000e+00,
       6.28318531e+00, 6.28318531e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 1.36305981e+00, 6.28318531e+00,
       0.00000000e+00, 6.28318531e+00, 3.89922876e+00, 0.00000000e+00,
       6.28318531e+00, 3.17574164e+00, 0.00000000e+00, 2.39452362e+00])


def convert_to_days(eom: pydylan.eom.CR3BP, time_in_TU: float) -> float:

    return time_in_TU * eom.TU / 86400


def output_control_to_screen(control):
    print('\nLength of control vector:', len(control), flush=True)
    print('\nThe control vector:', flush=True)
    for entry in control:
        print('{},'.format(entry), flush=True)


def get_throttle_history(control_states):
    assert isinstance(control_states, np.ndarray)
    assert control_states.ndim == 2
    assert control_states.shape[1] == 3

    throttle_history = np.zeros(control_states.shape[0],)
    for i, control in enumerate(control_states):
        throttle_history[i] = np.linalg.norm(control)

    return throttle_history


def generate_and_plot_manifold(orbit: pydylan.periodic_orbit, plot: LinePlot, coordinates=(0, 1), manifold_time=10., manifold_direction=pydylan.enum.PerturbationDirection.StableLeft):

    orbit_times = np.linspace(0., orbit.orbit_period, 50)

    StableDirections = [pydylan.enum.PerturbationDirection.StableLeft, pydylan.enum.PerturbationDirection.StableRight]
    defcolor=html_colors['baby blue'] if manifold_direction in StableDirections else html_colors['strong orange']

    for orbit_time in orbit_times:
        manifold_arc = orbit.generate_manifold_arc(orbit_time, manifold_time, manifold_direction)
        plot.plot(xdata=manifold_arc.mani_states[:, coordinates[0]], ydata=manifold_arc.mani_states[:, coordinates[1]], color=defcolor, alpha=0.2)

    return plot


def periodic_orbit_continuation(periodic_orbit: pydylan.periodic_orbit, desired_value: float) -> pydylan.periodic_orbit:

    continuation_settings = pydylan.periodic_orbit.ContinuationParameterInfo()
    if type(periodic_orbit)==pydylan.periodic_orbit.Lyapunov:
        continuation_settings.parameter_t = pydylan.enum.ParameterType.energy
    else:
        continuation_settings.parameter_t = pydylan.enum.ParameterType.single_state
        continuation_settings.state_index = 2
    continuation_settings.desired_value = desired_value
    continuation_settings.step_size = 1e-4
    continuation_settings.min_step_size = 1e-10
    periodic_orbit.set_continuation_information(continuation_settings, iLimit=1000)
    result = periodic_orbit.solve_for_orbit()

    return periodic_orbit


def solve_for_first_phase(eom: pydylan.eom, thruster_parameters: pydylan.ThrustParameters, L1L: pydylan.periodic_orbit.Lyapunov) -> (np.ndarray, np.ndarray):

    L1_unstable_manifold_arc = L1L.generate_manifold_arc(L1L.orbit_period, 4.5, pydylan.enum.PerturbationDirection.UnstableLeft)    
    left_boundary_condition = pydylan.FixedBoundaryCondition(L1_unstable_manifold_arc.get_end_state())

    L1_stable_manifold_arc = L1L.generate_manifold_arc(L1L.orbit_period, 4.5, pydylan.enum.PerturbationDirection.StableLeft)
    right_boundary_condition = pydylan.FixedBoundaryCondition(L1_stable_manifold_arc.get_end_state())

    snopt_options = pydylan.SNOPT_options_structure()
    snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
    snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible
    snopt_options.quiet_SNOPT = False
    snopt_options.time_limit = 1.

    mbh_options = pydylan.MBH_options_structure()
    mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
    mbh_options.max_step_size = 0.75
    mbh_options.quiet_MBH = False
    mbh_options.time_limit = 1.

    phase_options = pydylan.phase_options_structure()
    phase_options.number_of_segments = 100
    phase_options.minimum_initial_coast_time = 0.45
    phase_options.maximum_initial_coast_time = 0.45
    phase_options.minimum_final_coast_time = 0.
    phase_options.maximum_final_coast_time = 1.
    phase_options.minimum_shooting_time = 0.
    phase_options.maximum_shooting_time = 25.
    phase_options.match_point_position_constraint_tolerance = 1E-6
    phase_options.match_point_velocity_constraint_tolerance = 1E-6
    phase_options.match_point_mass_constraint_tolerance = 1E-4
    phase_options.control_coordinate_transcription = pydylan.enum.polar

    mission = pydylan.Mission(eom, left_boundary_condition, right_boundary_condition, pydylan.enum.mbh)
    mission.add_phase_options(phase_options)
    mission.set_thruster_parameters(thruster_parameters)
    mission.add_control_initial_guess(p1_initial_guess())
    mission.optimize(snopt_options, mbh_options)

    output_control_to_screen(mission.get_control_state())

    assert mission.is_best_solution_feasible()
    np.save("./Plots_For_NG_Meeting_2022/CLtour_p1_init.npy", mission.get_control_state())
    np.save("./Plots_For_NG_Meeting_2022/CLtour_p1_feasible_control_solutions.npy", mission.get_all_feasible_control_solutions())

    result = mission.evaluate_and_return_solution(mission.get_control_state(), forward_backward_shooting=True)

    time = result.time
    states = result.states

    return time, states, L1_unstable_manifold_arc, L1_stable_manifold_arc


def solve_for_second_phase(eom: pydylan.eom, thruster_parameters: pydylan.ThrustParameters, L1L: pydylan.periodic_orbit.Lyapunov, L2L: pydylan.periodic_orbit.Lyapunov) -> (np.ndarray, np.ndarray):

    L1_manifold_arc = L1L.generate_manifold_arc(L1L.orbit_period, 3.5, pydylan.enum.PerturbationDirection.UnstableRight)    
    left_boundary_condition = pydylan.FixedBoundaryCondition(L1_manifold_arc.get_end_state())

    L2_manifold_arc = L2L.generate_manifold_arc(L2L.orbit_period, 4., pydylan.enum.PerturbationDirection.StableLeft)
    right_boundary_condition = pydylan.FixedBoundaryCondition(L2_manifold_arc.get_end_state())

    snopt_options = pydylan.SNOPT_options_structure()
    snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
    snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible
    snopt_options.quiet_SNOPT = False
    snopt_options.time_limit = 1.

    mbh_options = pydylan.MBH_options_structure()
    mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
    mbh_options.max_step_size = 0.75
    mbh_options.quiet_MBH = False
    mbh_options.time_limit = 1.

    phase_options = pydylan.phase_options_structure()
    phase_options.number_of_segments = 50
    phase_options.minimum_initial_coast_time = 0.
    phase_options.maximum_initial_coast_time = 1.
    phase_options.minimum_final_coast_time = 0.
    phase_options.maximum_final_coast_time = 1.
    phase_options.minimum_shooting_time = 0.
    phase_options.maximum_shooting_time = 10.
    phase_options.match_point_position_constraint_tolerance = 1E-6
    phase_options.match_point_velocity_constraint_tolerance = 1E-6
    phase_options.match_point_mass_constraint_tolerance = 1E-4
    phase_options.control_coordinate_transcription = pydylan.enum.polar

    mission = pydylan.Mission(eom, left_boundary_condition, right_boundary_condition, pydylan.enum.mbh)
    mission.add_phase_options(phase_options)
    mission.set_thruster_parameters(thruster_parameters)
    mission.add_control_initial_guess(p2_initial_guess())
    mission.optimize(snopt_options, mbh_options)

    output_control_to_screen(mission.get_control_state())

    assert mission.is_best_solution_feasible()
    np.save("./Plots_For_NG_Meeting_2022/CLtour_p2_init.npy", mission.get_control_state())
    np.save("./Plots_For_NG_Meeting_2022/CLtour_p2_feasible_control_solutions.npy", mission.get_all_feasible_control_solutions())

    result = mission.evaluate_and_return_solution(mission.get_control_state(), forward_backward_shooting=True)

    time = result.time
    states = result.states

    return time, states, L1_manifold_arc, L2_manifold_arc
    

def solve_for_third_phase(eom: pydylan.eom, thruster_parameters: pydylan.ThrustParameters, L2L: pydylan.periodic_orbit.Lyapunov) -> (np.ndarray, np.ndarray):

    L2_unstable_manifold_arc = L2L.generate_manifold_arc(L2L.orbit_period, 5., pydylan.enum.PerturbationDirection.UnstableRight)    
    left_boundary_condition = pydylan.FixedBoundaryCondition(L2_unstable_manifold_arc.get_end_state())

    L2_stable_manifold_arc = L2L.generate_manifold_arc(L2L.orbit_period, 5., pydylan.enum.PerturbationDirection.StableRight)
    right_boundary_condition = pydylan.FixedBoundaryCondition(L2_stable_manifold_arc.get_end_state())

    snopt_options = pydylan.SNOPT_options_structure()
    snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
    snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible
    snopt_options.quiet_SNOPT = False
    snopt_options.time_limit = 1.

    mbh_options = pydylan.MBH_options_structure()
    mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
    mbh_options.max_step_size = 0.75
    mbh_options.quiet_MBH = False
    mbh_options.time_limit = 1.

    phase_options = pydylan.phase_options_structure()
    phase_options.number_of_segments = 100
    phase_options.minimum_initial_coast_time = 2.
    phase_options.maximum_initial_coast_time = 2.
    phase_options.minimum_final_coast_time = 0.
    phase_options.maximum_final_coast_time = 2.
    phase_options.minimum_shooting_time = 0.
    phase_options.maximum_shooting_time = 25.
    phase_options.match_point_position_constraint_tolerance = 1E-6
    phase_options.match_point_velocity_constraint_tolerance = 1E-6
    phase_options.match_point_mass_constraint_tolerance = 1E-4
    phase_options.control_coordinate_transcription = pydylan.enum.polar

    mission = pydylan.Mission(eom, left_boundary_condition, right_boundary_condition, pydylan.enum.mbh)
    mission.add_phase_options(phase_options)
    mission.set_thruster_parameters(thruster_parameters)
    mission.add_control_initial_guess(p3_initial_guess())
    mission.optimize(snopt_options, mbh_options)

    output_control_to_screen(mission.get_control_state())

    assert mission.is_best_solution_feasible()
    np.save("./Plots_For_NG_Meeting_2022/CLtour_p3_init.npy", mission.get_control_state())
    np.save("./Plots_For_NG_Meeting_2022/CLtour_p3_feasible_control_solutions.npy", mission.get_all_feasible_control_solutions())

    result = mission.evaluate_and_return_solution(mission.get_control_state(), forward_backward_shooting=True)

    time = result.time
    states = result.states

    return time, states, L2_unstable_manifold_arc, L2_stable_manifold_arc


def solve():

    earth = pydylan.Body("Earth")
    moon = pydylan.Body("Moon")
    cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)

    # L1 Lyapunov
    libration_point_information_L1 = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L1)
    desired_orbit_energy_L1 = libration_point_information_L1[1] + 100E-4
    lyapunov_L1 = pydylan.periodic_orbit.Lyapunov(cr3bp, pydylan.enum.LibrationPoint.L1, desired_orbit_energy_L1)
    lyapunov_L1 = periodic_orbit_continuation(lyapunov_L1, desired_orbit_energy_L1)

    # L2 Lyapunov
    libration_point_information_L2 = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L2)
    desired_orbit_energy_L2 = desired_orbit_energy_L1
    lyapunov_L2 = pydylan.periodic_orbit.Lyapunov(cr3bp, pydylan.enum.LibrationPoint.L2, desired_orbit_energy_L2)
    lyapunov_L2 = periodic_orbit_continuation(lyapunov_L2, desired_orbit_energy_L2)

    thruster_parameters = pydylan.ThrustParameters(fuel_mass=2.5, dry_mass=12.5, Isp=2500., thrust=1.5E-3)

    ''' Phase I: EM L1 Lyapunov to EM L1 Lyapunov with Earth Flyby'''
    time_p1, states_p1, p1_L1_unstable_mani_arc, p1_L1_stable_mani_arc = solve_for_first_phase(cr3bp, thruster_parameters, lyapunov_L1)
    time_p1 = convert_to_days(cr3bp, time_p1)

    ''' Phase II: EM L1 Lyapunov to EM L2 Lyapunov with Moon Flyby'''
    thruster_parameters.fuel_mass = states_p1[-1, 6]
    time_p2, states_p2, p2_L1_unstable_mani_arc, p2_L2_stable_mani_arc = solve_for_second_phase(cr3bp, thruster_parameters, lyapunov_L1, lyapunov_L2)
    time_p2 = convert_to_days(cr3bp, time_p2)

    ''' Phase III: EM L2 Lyapunov to EM L2 Lyapunov with Earth Flyby'''
    thruster_parameters.fuel_mass = states_p2[-1, 6]
    time_p3, states_p3, p3_L1_unstable_mani_arc, p3_L2_stable_mani_arc = solve_for_third_phase(cr3bp, thruster_parameters, lyapunov_L2)
    time_p3 = convert_to_days(cr3bp, time_p3)

    # Pack the result into a dictionary
    result_dict = {
            'p1_times': time_p1, 
            'p2_times': time_p2, 
            'p3_times': time_p3,
            'p1_states': states_p1,
            'p2_states': states_p2,
            'p3_states': states_p3,
            'L1_Lyapunov': lyapunov_L1,
            'L1_point_information': libration_point_information_L1,
            'L2_Lyapunov': lyapunov_L2,
            'L2_point_information': libration_point_information_L2,
            'p1_L1_unstable_mani_arc': p1_L1_unstable_mani_arc,
            'p1_L1_stable_mani_arc': p1_L1_stable_mani_arc,
            'p2_L1_unstable_mani_arc': p2_L1_unstable_mani_arc,
            'p2_L2_stable_mani_arc': p2_L2_stable_mani_arc,
            'p3_L1_unstable_mani_arc': p3_L1_unstable_mani_arc,
            'p3_L2_stable_mani_arc': p3_L2_stable_mani_arc}

    return result_dict


def plot_phase(result_dict, phase, pathname):

    # Unpack the result into a dictionary

    time_p1, states_p1 = result_dict.get('p1_times'), result_dict.get('p1_states')
    time_p2, states_p2 = result_dict.get('p2_times'), result_dict.get('p2_states')
    time_p3, states_p3 = result_dict.get('p3_times'), result_dict.get('p3_states')
    lyapunov_L1, lyapunov_L2 = result_dict.get('L1_Lyapunov'), result_dict.get('L2_Lyapunov')
    libration_point_information_L1, libration_point_information_L2 = result_dict.get('L1_point_information'), result_dict.get('L2_point_information')
    p1_L1_unstable_mani_arc, p1_L1_stable_mani_arc = result_dict.get('p1_L1_unstable_mani_arc'), result_dict.get('p1_L1_stable_mani_arc')
    p2_L1_unstable_mani_arc, p2_L2_stable_mani_arc = result_dict.get('p2_L1_unstable_mani_arc'), result_dict.get('p2_L2_stable_mani_arc')
    p3_L1_unstable_mani_arc, p3_L2_stable_mani_arc = result_dict.get('p3_L1_unstable_mani_arc'), result_dict.get('p3_L2_stable_mani_arc')

    p = LinePlot()
    p.grid()
    p.set_xlabel(xlabel_in=r'X (DU)', fontsize=15)
    p.set_ylabel(ylabel_in=r'Y (DU)', fontsize=15)
    # p.set_title(title_in='Low Thrust Cislunar Tour', fontsize=20)

    p = generate_and_plot_manifold(orbit=lyapunov_L1, plot=p, coordinates=(0, 1), manifold_time=10., manifold_direction=pydylan.enum.PerturbationDirection.UnstableLeft)
    p = generate_and_plot_manifold(orbit=lyapunov_L1, plot=p, coordinates=(0, 1), manifold_time=10., manifold_direction=pydylan.enum.PerturbationDirection.StableLeft)

    p = generate_and_plot_manifold(orbit=lyapunov_L1, plot=p, coordinates=(0, 1), manifold_time=5., manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight)
    p = generate_and_plot_manifold(orbit=lyapunov_L2, plot=p, coordinates=(0, 1), manifold_time=7., manifold_direction=pydylan.enum.PerturbationDirection.StableLeft)

    p = generate_and_plot_manifold(orbit=lyapunov_L2, plot=p, coordinates=(0, 1), manifold_time=7., manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight)
    p = generate_and_plot_manifold(orbit=lyapunov_L2, plot=p, coordinates=(0, 1), manifold_time=7., manifold_direction=pydylan.enum.PerturbationDirection.StableRight)

    p.plot(xdata=states_p1[:, 0], ydata=states_p1[:, 1], color='black', linewidth=1)
    p.plot(xdata=states_p2[:, 0], ydata=states_p2[:, 1], color='black', linewidth=1)

    p.plot(xdata=p1_L1_unstable_mani_arc.mani_states[:, 0], ydata=p1_L1_unstable_mani_arc.mani_states[:, 1], color='black', linewidth=1)
    p.plot(xdata=p1_L1_stable_mani_arc.mani_states[:, 0], ydata=p1_L1_stable_mani_arc.mani_states[:, 1], color='black', linewidth=1)

    p.plot(xdata=p2_L1_unstable_mani_arc.mani_states[:, 0], ydata=p2_L1_unstable_mani_arc.mani_states[:, 1], color='black', linewidth=1)
    p.plot(xdata=p2_L2_stable_mani_arc.mani_states[:, 0], ydata=p2_L2_stable_mani_arc.mani_states[:, 1], color='black', linewidth=1)

    if phase==0:
        p.plot(xdata=states_p3[:, 0], ydata=states_p3[:, 1], color='black', linewidth=1)
        p.plot(xdata=p3_L1_unstable_mani_arc.mani_states[:, 0], ydata=p3_L1_unstable_mani_arc.mani_states[:, 1], color='black', linewidth=1)
        p.plot(xdata=p3_L2_stable_mani_arc.mani_states[:, 0], ydata=p3_L2_stable_mani_arc.mani_states[:, 1], color='black', linewidth=1)
        p.save_figure(pathname+'CLtour_full_xy.png', dpi=100)
        
        t = LinePlot()
        t.grid()
        t.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        t.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
        t.plot(xdata=time_p1, ydata=get_throttle_history(states_p1[:, -3:]), color='black')
        t.plot(xdata=time_p1[-1] + time_p2, ydata=get_throttle_history(states_p2[:, -3:]), color='black')
        t.plot(xdata=time_p1[-1] + time_p2[-1] + time_p3, ydata=get_throttle_history(states_p3[:, -3:]), color='black')
        t.save_figure(pathname+'CLtour_full_throttle.png', dpi=100)

        m = LinePlot()
        m.grid()
        m.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        m.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
        m.plot(xdata=time_p1[-1] + time_p2[-1] + time_p3, ydata=states_p3[:, 6], color='black')
        m.plot(xdata=time_p1, ydata=states_p1[:, 6], color='black')
        m.plot(xdata=time_p1[-1] + time_p2, ydata=states_p2[:, 6], color='black')
        m.save_figure(pathname+'CLtour_full_mass.png', dpi=100)
    elif phase==1:
        p.set_axis('equal', [-0.75, libration_point_information_L2[0][0], -0.8, 0.8])
        p.save_figure(pathname+'CLtour_earth_moon_xy.png', dpi=100)

        t = LinePlot()
        t.grid()
        t.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        t.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
        t.plot(xdata=time_p1, ydata=get_throttle_history(states_p1[:, -3:]), color='black')
        t.plot(xdata=time_p1[-1] + time_p2, ydata=get_throttle_history(states_p2[:, -3:]), color='black')
        t.save_figure(pathname+'CLtour_earth_moon_throttle.png', dpi=100)

        m = LinePlot()
        m.grid()
        m.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        m.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
        m.plot(xdata=time_p1, ydata=states_p1[:, 6], color='black')
        m.plot(xdata=time_p1[-1] + time_p2, ydata=states_p2[:, 6], color='black')
        m.save_figure(pathname+'CLtour_earth_moon_mass.png', dpi=100)
    elif phase==10:
        p.set_axis('equal', [-0.75, libration_point_information_L1[0][0], -0.8, 0.8])
        p.save_figure(pathname+'CLtour_earth_xy.png', dpi=100)
        t = LinePlot()

        t.grid()
        t.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        t.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
        t.plot(xdata=time_p1, ydata=get_throttle_history(states_p1[:, -3:]), color='black')
        t.save_figure(pathname+'CLtour_earth_throttle.png', dpi=100)

        m = LinePlot()
        m.grid()
        m.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        m.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
        m.plot(xdata=time_p1, ydata=states_p1[:, 6], color='black')
        m.save_figure(pathname+'CLtour_earth_mass.png', dpi=100)
    elif phase==11:
        p.set_axis('equal', [libration_point_information_L1[0][0], libration_point_information_L2[0][0], -0.15, 0.15])
        p.save_figure(pathname+'CLtour_moon_xy.png', dpi=100)
        t = LinePlot()

        t.grid()
        t.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        t.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
        t.plot(xdata=time_p1[-1] + time_p2, ydata=get_throttle_history(states_p2[:, -3:]), color='black')
        t.save_figure(pathname+'CLtour_moon_throttle.png', dpi=100)

        m = LinePlot()
        m.grid()
        m.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        m.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
        m.plot(xdata=time_p1[-1] + time_p2, ydata=states_p2[:, 6], color='black')
        m.save_figure(pathname+'CLtour_moon_mass.png', dpi=100)
    elif phase==2:
        p.plot(xdata=states_p3[:, 0], ydata=states_p3[:, 1], color='black', linewidth=1)
        p.plot(xdata=p3_L1_unstable_mani_arc.mani_states[:, 0], ydata=p3_L1_unstable_mani_arc.mani_states[:, 1], color='black', linewidth=1)
        p.plot(xdata=p3_L2_stable_mani_arc.mani_states[:, 0], ydata=p3_L2_stable_mani_arc.mani_states[:, 1], color='black', linewidth=1)
        p.save_figure(pathname+'CLtour_last_xy.png', dpi=100)

        t = LinePlot()
        t.grid()
        t.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        t.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
        t.plot(xdata=time_p1[-1] + time_p2[-1] + time_p3, ydata=get_throttle_history(states_p3[:, -3:]), color='black')
        t.save_figure(pathname+'CLtour_last_throttle.png', dpi=100)

        m = LinePlot()
        m.grid()
        m.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        m.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
        m.plot(xdata=time_p1[-1] + time_p2[-1] + time_p3, ydata=states_p3[:, 6], color='black')
        m.save_figure(pathname+'CLtour_last_mass.png', dpi=100)
    
    plt.show()


if __name__ == '__main__':

    try:
        os.mkdir('./Plots_For_NG_Meeting_2022')
    except FileExistsError:
        print("Directory already exists")

    result_dict = solve()

    plot_phase(result_dict, 1, pathname='./Plots_For_NG_Meeting_2022/') # first+second phase
    plot_phase(result_dict, 10, pathname='./Plots_For_NG_Meeting_2022/') # first phase
    plot_phase(result_dict, 11, pathname='./Plots_For_NG_Meeting_2022/') # second phase
    plot_phase(result_dict, 2, pathname='./Plots_For_NG_Meeting_2022/') # last phase
    plot_phase(result_dict, 0, pathname='./Plots_For_NG_Meeting_2022/') # full tour
