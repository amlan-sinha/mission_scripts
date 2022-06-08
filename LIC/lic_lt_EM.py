import sys, os

sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding/debug')
sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding')

import pydylan
import numpy as np
import pandas as pd
from GSFC_reference_waypoints import get_post_deployment_epoch_in_MJS_and_state_in_EJ2000, get_post_flyby_epoch_in_MJS_and_state_in_EJ2000, get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000

import matplotlib.pyplot as plt
from lineplot import LinePlot
from support import html_colors

days_to_seconds, seconds_to_days = 86400., 1/86400.

strongorange, babyblue = html_colors['strong orange'], html_colors['baby blue']


# Feasible initial condition with Fixed+Fixed+Fixed BC (Phase I)
def p1_initial_guess_fixed_BC():

    return np.array([4.82482866e+05, 0.00000000e+00, 0.00000000e+00, 2.12122534e+00,
       0.00000000e+00, 1.00000000e+00, 6.07653656e+00, 1.11879809e+00,
       0.00000000e+00, 2.11802598e+00, 0.00000000e+00, 7.83263956e-01,
       2.02256518e+00, 0.00000000e+00, 0.00000000e+00, 5.60337716e+00,
       4.16489261e+00, 6.35773888e-01, 3.05833137e+00, 2.28978456e-02,
       8.14424692e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       1.28106227e+00, 5.61211936e-01, 0.00000000e+00, 4.79030298e+00,
       4.13385240e-01, 0.00000000e+00, 2.79601855e+00, 0.00000000e+00,
       0.00000000e+00, 1.49145524e+00])


# Feasible initial condition with Fixed+Advecting+CR3BP BC (Phase I)
def p1_initial_guess_other_BC():

    return np.array([4.58207756e+05, 0.00000000e+00, 0.00000000e+00, 2.20652068e+00,
       5.78721592e-01, 9.93006840e-01, 5.64018641e-01, 4.36369403e+00,
       3.26264811e-04, 1.55274588e+00, 1.80192466e+00, 5.90662747e-02,
       1.25368338e+00, 4.94129234e-01, 2.55762536e-01, 0.00000000e+00,
       5.82655813e+00, 0.00000000e+00, 2.30905729e-03, 1.26645898e-01,
       6.55104327e-01, 2.50085433e+00, 1.79397487e-02, 8.20500953e-01,
       4.65398931e+00, 3.35721722e-03, 9.92983271e-01, 3.46900620e+00,
       1.52371811e-03, 8.00528095e-01, 0.00000000e+00, 1.86400730e+00,
       3.04324362e-02, 1.48608297e+00])


# Feasible initial condition with Fixed+Fixed+Fixed BC (Phase II)
def p2_initial_guess_fixed_BC():

    return np.array([1.55321644e+07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 3.31176642e+00, 1.53476270e-01,
       4.31881546e-01, 3.58472450e+00, 6.28318531e+00, 0.00000000e+00,
       0.00000000e+00, 2.74356732e-01, 0.00000000e+00, 3.11545962e+00,
       4.21640735e-02, 0.00000000e+00, 3.01288387e+00, 6.12038633e+00,
       0.00000000e+00, 3.00441996e+00, 0.00000000e+00, 0.00000000e+00,
       2.72460484e+00, 0.00000000e+00, 0.00000000e+00, 5.77078982e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 8.83623250e-01, 0.00000000e+00,
       2.93633473e+00, 0.00000000e+00, 0.00000000e+00, 2.62078060e+00,
       0.00000000e+00, 8.17831228e-05, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 3.47523440e+00, 0.00000000e+00,
       2.87645280e+00, 0.00000000e+00, 2.18525422e-01, 3.07880918e+00,
       0.00000000e+00, 0.00000000e+00, 2.21180172e+00, 0.00000000e+00,
       0.00000000e+00, 3.90603991e+00, 0.00000000e+00, 0.00000000e+00,
       5.19212712e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 2.53932329e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 3.82477885e+00, 0.00000000e+00, 7.79324461e-02,
       1.03608660e+00, 0.00000000e+00, 2.86732174e+00, 1.93584645e-01,
       0.00000000e+00, 3.32816828e+00, 5.50209134e+00, 0.00000000e+00,
       0.00000000e+00, 4.57984539e+00, 0.00000000e+00, 2.04832076e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.22469141e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       5.63346317e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 1.11783536e+00, 1.02126008e+00,
       0.00000000e+00, 4.34557867e+00, 9.77248373e-02, 0.00000000e+00,
       4.19938268e+00, 5.10852155e+00, 0.00000000e+00, 2.30897960e+00,
       0.00000000e+00, 0.00000000e+00, 1.79754329e+00, 0.00000000e+00,
       0.00000000e+00, 1.09750408e+00, 5.96554815e+00, 0.00000000e+00,
       2.27295269e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       5.86794155e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 8.43436826e-01, 1.74818672e+00, 0.00000000e+00,
       0.00000000e+00, 1.10239327e-01, 0.00000000e+00, 2.04557243e+00,
       7.14601708e-01, 0.00000000e+00, 2.29849165e+00, 4.48138455e-01,
       0.00000000e+00, 2.86347222e-01, 2.82560778e+00, 0.00000000e+00,
       3.76832971e+00, 0.00000000e+00, 0.00000000e+00, 2.51838723e+00,
       7.97438786e-02, 0.00000000e+00, 2.61833232e+00, 1.84883131e+00,
       0.00000000e+00, 3.12157206e-01, 0.00000000e+00, 6.68973453e-01,
       2.13666202e-01, 0.00000000e+00, 0.00000000e+00, 1.43358210e-01,
       0.00000000e+00, 0.00000000e+00, 4.53705882e-01, 8.67207276e-01,
       0.00000000e+00, 1.33227763e-01, 2.46776898e+00, 0.00000000e+00,
       0.00000000e+00, 8.22415582e-01, 0.00000000e+00, 2.65495957e+00,
       3.61722281e+00, 0.00000000e+00, 1.22150111e-01, 4.74982097e+00,
       0.00000000e+00, 6.06260144e+00, 0.00000000e+00, 0.00000000e+00,
       7.22597860e-01, 0.00000000e+00, 0.00000000e+00, 1.10662018e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 6.28318531e+00, 4.14100545e+00, 0.00000000e+00,
       4.96831691e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 2.70136048e+00, 0.00000000e+00,
       0.00000000e+00, 2.89943993e+00, 2.06154940e-01, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       7.71559078e-01, 0.00000000e+00, 2.06537533e+00, 3.47743051e-01,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       4.51695481e+00, 5.74340924e+00, 0.00000000e+00, 2.42571038e-01,
       0.00000000e+00, 0.00000000e+00, 1.13449090e+00, 6.01250123e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       3.82176612e+00, 0.00000000e+00, 2.33293675e+00, 2.25236031e-01,
       0.00000000e+00, 0.00000000e+00, 3.86639164e+00, 1.92516289e-01,
       4.30944655e+00, 2.10383365e+00, 0.00000000e+00, 1.64300889e+00,
       0.00000000e+00, 0.00000000e+00, 2.69135866e+00, 5.65001504e+00,
       1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       5.30136911e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       3.70950261e+00, 1.00000000e+00, 0.00000000e+00, 3.66277745e+00,
       1.00000000e+00, 5.92541155e+00, 3.59771280e+00, 1.00000000e+00,
       2.21140080e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       3.54583106e+00, 1.00000000e+00, 5.22324758e+00, 5.70338593e+00,
       0.00000000e+00, 0.00000000e+00, 8.78039376e-01, 0.00000000e+00,
       2.78598152e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
       1.66943803e+00, 0.00000000e+00, 4.88542915e-01, 0.00000000e+00,
       0.00000000e+00, 2.76546950e+00, 0.00000000e+00, 1.00000000e+00,
       0.00000000e+00, 3.22421619e+00, 1.00000000e+00, 0.00000000e+00,
       1.06878252e+00, 0.00000000e+00, 5.86857113e+00, 3.12086465e+00,
       1.00000000e+00, 0.00000000e+00, 3.50740117e+00, 0.00000000e+00,
       7.42419017e-02, 0.00000000e+00, 0.00000000e+00, 1.39256306e+00])


# Feasible initial condition with Fixed+Advecting+CR3BP BC (Phase II)
def p2_initial_guess_other_BC():

    return np.array([1.53713367e+07, 0.00000000e+00, 0.00000000e+00, 5.49651486e+00,
       4.75236791e+00, 8.44944955e-01, 3.36314909e+00, 4.04634546e+00,
       0.00000000e+00, 4.01651710e+00, 0.00000000e+00, 9.03645050e-01,
       3.52523359e+00, 2.24786652e+00, 0.00000000e+00, 1.94622940e+00,
       2.25203668e+00, 7.05566356e-01, 0.00000000e+00, 4.66594410e-02,
       9.16053848e-01, 3.30506351e+00, 2.37352333e-01, 5.91208018e-02,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.60869381e-01,
       3.20676676e+00, 9.39449021e-01, 0.00000000e+00, 2.01458470e-02,
       6.56465332e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       2.64061674e+00, 1.89095460e-02, 7.03905292e-01, 0.00000000e+00,
       1.46943596e+00, 0.00000000e+00, 3.90612511e+00, 5.41543305e+00,
       8.01796312e-01, 0.00000000e+00, 2.50834983e-01, 6.58115874e-01,
       3.63855521e-01, 3.16941603e+00, 7.94815445e-01, 3.02839962e-02,
       5.54274223e-01, 8.75834172e-01, 6.61830652e-03, 3.06907610e+00,
       8.37102447e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       5.20012289e-03, 3.18251735e+00, 1.17115391e-01, 1.35504012e+00,
       0.00000000e+00, 0.00000000e+00, 8.82118703e-01, 0.00000000e+00,
       0.00000000e+00, 3.65803893e+00, 6.15852929e+00, 9.94968030e-01,
       0.00000000e+00, 1.98646063e+00, 4.74766614e-02, 3.39476306e+00,
       4.10742893e-03, 1.00000000e+00, 1.92401592e+00, 1.02856161e+00,
       1.75685724e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 1.55207377e+00, 3.09965867e-03, 4.29242305e-01,
       2.29533889e-01, 6.78687169e-01, 1.96849928e+00, 0.00000000e+00,
       2.74545529e-02, 2.70108645e+00, 0.00000000e+00, 1.00000000e+00,
       1.44840762e-02, 3.86434208e+00, 9.32010632e-01, 5.98981139e-01,
       0.00000000e+00, 6.58213500e-01, 4.24674094e+00, 2.18227128e+00,
       1.62739303e-03, 2.41202521e+00, 5.66511263e+00, 8.81570394e-01,
       3.00672426e+00, 3.02852404e-01, 9.60642509e-01, 0.00000000e+00,
       1.31752764e+00, 0.00000000e+00, 2.94327587e+00, 0.00000000e+00,
       7.97505359e-02, 2.54450352e+00, 2.99524801e-01, 9.59528022e-01,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.20598779e+00,
       3.62879429e-01, 7.92949195e-01, 0.00000000e+00, 4.78120108e+00,
       9.25179325e-04, 5.27448086e+00, 5.52465699e-02, 2.53885188e-03,
       1.70166367e+00, 1.03479478e+00, 7.41582075e-01, 3.30370419e+00,
       1.81954500e+00, 5.88131422e-01, 2.92143649e-01, 5.23669512e+00,
       1.97643191e-04, 5.29242264e+00, 9.54371805e-01, 3.43906652e-04,
       5.39469513e+00, 0.00000000e+00, 0.00000000e+00, 1.18848184e-01,
       3.18746476e+00, 9.56361421e-01, 0.00000000e+00, 3.31626155e+00,
       8.96648167e-01, 4.28998453e+00, 0.00000000e+00, 4.29081523e-04,
       4.67678980e+00, 4.22726478e+00, 1.95873994e-02, 7.16716812e-01,
       3.73650536e-02, 6.70220215e-02, 3.17196239e+00, 6.95087686e-01,
       5.68415979e-01, 3.58185570e+00, 8.35771627e-01, 6.53113216e-01,
       4.63862054e+00, 2.76784660e+00, 8.92924064e-01, 0.00000000e+00,
       5.80803255e+00, 1.20176382e-02, 5.39290882e+00, 2.03161790e+00,
       7.38860366e-01, 0.00000000e+00, 2.19687425e+00, 9.18225826e-01,
       9.07944172e-01, 1.74573378e+00, 1.63100491e-02, 1.79520351e+00,
       6.24997962e+00, 7.95107196e-01, 2.13850700e-01, 1.50501417e-01,
       2.19959174e-01, 6.01389960e+00, 2.50673053e+00, 2.38263979e-02,
       3.95491771e+00, 5.02937769e+00, 9.16265922e-01, 8.07712999e-02,
       4.19960574e-02, 9.89507119e-02, 2.15122132e+00, 1.67168233e-01,
       7.31328449e-01, 1.29137126e+00, 1.65373853e+00, 9.38028679e-01,
       4.85685548e+00, 2.22707599e-01, 1.32394496e-01, 4.88640663e+00,
       0.00000000e+00, 4.25495391e-02, 3.01492251e+00, 0.00000000e+00,
       8.02490514e-01, 0.00000000e+00, 0.00000000e+00, 2.52000952e-02,
       1.33697659e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 7.13963254e-02, 2.01507324e+00, 0.00000000e+00,
       9.70486548e-03, 1.74016770e-01, 4.26971280e+00, 2.10881352e-01,
       2.13131778e-01, 3.26133467e+00, 7.51558808e-01, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 2.00903059e-01, 0.00000000e+00,
       0.00000000e+00, 6.51142265e-01, 9.33998711e-01, 0.00000000e+00,
       5.43177164e+00, 1.75136906e+00, 2.00605873e-01, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 3.17424286e+00, 9.10271295e-04,
       8.18618689e-01, 0.00000000e+00, 1.05861845e-03, 8.48835824e-01,
       2.77436494e+00, 1.69012639e-03, 1.00000000e+00, 4.08327953e-01,
       2.69520561e+00, 1.26150965e-01, 0.00000000e+00, 1.45821537e+00,
       8.01500349e-01, 2.46346039e-03, 2.81530122e+00, 1.44470997e-01,
       6.28318531e+00, 3.12508857e+00, 8.36043929e-01, 5.80341812e+00,
       3.56511061e+00, 6.70044897e-01, 0.00000000e+00, 2.18225944e+00,
       6.22640063e-01, 6.66762567e-02, 2.75944632e+00, 7.36819826e-01,
       3.23243071e+00, 6.28318531e+00, 9.75430558e-01, 0.00000000e+00,
       8.68345563e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 6.06898229e+00, 2.68664662e+00, 1.81094182e-01,
       1.00716645e-01, 2.12021233e-01, 7.27530761e-03, 0.00000000e+00,
       2.83009949e+00, 8.01915466e-01, 2.48774358e+00, 1.76358299e-01,
       3.41223721e-01, 1.35843087e-01, 2.76147886e-01, 6.83584267e-02,
       0.00000000e+00, 2.25131351e+00, 2.38358712e-01, 1.10045184e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00])


def get_throttle_history(control_states):
    assert isinstance(control_states, np.ndarray)
    assert control_states.ndim == 2
    assert control_states.shape[1] == 3

    throttle_history = np.zeros(control_states.shape[0],)
    for i, control in enumerate(control_states):
        throttle_history[i] = np.linalg.norm(control)

    return throttle_history


def output_control_to_screen(control):
    print('\nLength of control vector:', len(control), flush=True)
    print('\nThe control vector:', flush=True)
    for entry in control:
        print('{},'.format(entry), flush=True)


# integrate natural dynamics forward in time
def get_plots_of_phase_integration(body: pydylan.Body, eom: pydylan.eom.Ephemeris_nBP) -> (LinePlot, LinePlot):

    rk54 = pydylan.integrators.RK54()
    rk54.set_eom(eom)

    p = LinePlot()
    p.grid()
    p.set_xlabel(xlabel_in=r'X (km)', fontsize=15)
    p.set_ylabel(ylabel_in=r'Y (km)', fontsize=15)
    # p.set_title(title_in='EJ2000 MJS: {}'.format(epoch), fontsize=20)

    q = LinePlot()
    q.grid()
    q.set_xlabel(xlabel_in=r'X (km)', fontsize=15)
    q.set_ylabel(ylabel_in=r'Z (km)', fontsize=15)
    # q.set_title(title_in='EJ2000 MJS: {}'.format(epoch), fontsize=20)

    epoch, state = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000() # at deployment
    if body.name=='Moon': epoch, state = get_rv_state_relative_to_moon(epoch, state)

    p.plot(xdata=np.array([state[0], ]), ydata=np.array([state[1], ]), color='red', marker='x', markersize=7, linewidth=2)
    q.plot(xdata=np.array([state[0], ]), ydata=np.array([state[2], ]), color='red', marker='x', markersize=7, linewidth=2)

    post_flyby_epoch, state = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000() # after flyby around earth
    if body.name=='Moon': post_flyby_epoch, state = get_rv_state_relative_to_moon(post_flyby_epoch, state)
    
    p.plot(xdata=np.array([state[0], ]), ydata=np.array([state[1], ]), color='red', marker='x', markersize=7, linewidth=2)
    q.plot(xdata=np.array([state[0], ]), ydata=np.array([state[2], ]), color='red', marker='x', markersize=7, linewidth=2)

    NRHO_insertion_epoch, state = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000() # right before NRHO insertion
    if body.name=='Moon': NRHO_insertion_epoch, state = get_rv_state_relative_to_moon(NRHO_insertion_epoch, state)

    p.plot(xdata=np.array([state[0], ]), ydata=np.array([state[1], ]), color='red', marker='x', markersize=7, linewidth=2)
    q.plot(xdata=np.array([state[0], ]), ydata=np.array([state[2], ]), color='red', marker='x', markersize=7, linewidth=2)

    number_of_steps = 5000
    number_of_spline_points = int(number_of_steps / 5)
    start_time, end_time = epoch, NRHO_insertion_epoch + 7 * days_to_seconds
    time_points = np.linspace(start_time, end_time, num=number_of_steps)

    splined_moon = pydylan.Body("Moon", True, number_of_spline_points, start_time, end_time)

    if body==pydylan.Body("Earth"):
        moon_states = np.zeros((len(time_points), 6))
        for i, time in enumerate(time_points):
            moon_states[i] = splined_moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(time)

        p.plot(xdata=moon_states[:, 0], ydata=moon_states[:, 1], color='black', alpha=0.3, linestyle='--')
        q.plot(xdata=moon_states[:, 0], ydata=moon_states[:, 2], color='black', alpha=0.3, linestyle='--')
    else:
        earth_states = np.zeros((len(time_points), 6))
        for i, time in enumerate(time_points):
            earth_states[i] = - splined_moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(time)
        
        p.plot(xdata=earth_states[:, 0], ydata=earth_states[:, 1], color='black', alpha=0.3, linestyle='--')
        q.plot(xdata=earth_states[:, 0], ydata=earth_states[:, 2], color='black', alpha=0.3, linestyle='--')
    
    return p, q


# generate periodic orbits around unstable libration points
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


# generate CR3BP boundary conditions
def generate_CR3BP_boundary_condition():

    earth = pydylan.Body("Earth")
    moon = pydylan.Body("Moon")
    cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)

    libration_point_information_L2 = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L2)
    desired_orbit_energy_L2 = libration_point_information_L2[1] + 100E-4
    lyapunov_L2 = pydylan.periodic_orbit.Lyapunov(cr3bp, pydylan.enum.LibrationPoint.L2, desired_orbit_energy_L2)
    lyapunov_L2 = periodic_orbit_continuation(lyapunov_L2, desired_orbit_energy_L2)

    L2_manifold_arc = lyapunov_L2.generate_manifold_arc(lyapunov_L2.orbit_period, 4., pydylan.enum.PerturbationDirection.StableRight)

    return pydylan.CR3BPStateBoundaryCondition(earth, moon, L2_manifold_arc.get_end_state(), 1E3, 1E-2)


# convert rv state relative to moon to rv state relative to earth
def get_rv_state_relative_to_earth(epoch: float, states: np.array) -> (float, np.array):

    moon = pydylan.Body("Moon")

    if states.ndim==2:
        relative_state = np.copy(states)
        for i, ti in enumerate(epoch):
            relative_state[i, :6] = states[i, :6] + moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(epoch[i])
        return epoch, relative_state
    else:
        return epoch, states + moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(epoch)


# convert rv state relative to earth to rv state relative to moon
def get_rv_state_relative_to_moon(epoch: float, states: np.array) -> (float, np.array):

    moon = pydylan.Body("Moon")

    if states.ndim==2:
        relative_state = np.copy(states)
        for i, ti in enumerate(epoch):
            relative_state[i, :6] = states[i, :6] - moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(epoch[i])
        return epoch, relative_state
    else:
        return epoch, states - moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(epoch)


# convert rv state relative to coe state
def get_coe_state_from_rv_state(body: pydylan.Body, states: np.array) -> np.array:

    s2bp = pydylan.eom.S2BP(body)

    if states.ndim==2:
        coe_states = np.copy(states)
        for i, si in enumerate(states):
            tmp = s2bp.rv2coe(si[:3], si[3:6])
            coe_states[i, :6] = np.array([tmp.sma, tmp.ecc, tmp.inc, tmp.ran, tmp.ape, tmp.tru])
        return coe_states
    else:
        coe_states = s2bp.rv2coe(states[:3], states[3:6])
        return np.array([coe_states.sma, coe_states.ecc, coe_states.inc, coe_states.ran, coe_states.ape, coe_states.tru])


# convert coe state relative to rv state
def get_rv_state_from_coe_state(body: pydylan.Body, state: np.array) -> np.array:

    s2bp = pydylan.eom.S2BP(body)

    if state.ndim==2:
        coe_states = np.copy(state)
        for i, si in enumerate(state):
            sma, ecc, inc, lan, ape, tru = si[0], si[1], si[2], si[3], si[4], si[5]
            pos, vel = s2bp.coe2rv(sma, ecc, inc, lan, ape, tru)
            coe_states[i, :6] = np.concatenate((pos,vel))
        return coe_states
    else:
        sma, ecc, inc, lan, ape, tru = state[0], state[1], state[2], state[3], state[4], state[5]
        pos, vel = s2bp.coe2rv(sma, ecc, inc, lan, ape, tru)
        return np.concatenate((pos, vel))


# compute the relative error between two coe states
# TODO Need to change to compute a weighted sum
def compute_error_in_state(current_state: np.array, target_state: np.array, w: np.array):

    w = w / np.linalg.norm(w)
    err = w * (current_state - target_state)
    err = np.dot(err, err) / np.dot(target_state, target_state)

    return np.linalg.norm(err)


def read_qlaw_output(qlawhistory):

    # requires qlaw output to be written to "./qlaw_history.csv"
    qlawhistory = pd.read_csv(qlawhistory, names=["epoch", "sma", "ecc", "inc", "ape", "lan", "mass", "tru", "u0", "u1", "u2"])
    epoch, sma, ecc, inc, ape, lan, tru, mass, u0, u1, u2 = qlawhistory.epoch, qlawhistory.sma, qlawhistory.ecc, qlawhistory.inc, qlawhistory.ape, qlawhistory.lan, qlawhistory.tru, qlawhistory.mass, qlawhistory.u0, qlawhistory.u1, qlawhistory.u2

    epoch, sma, ecc, inc, ape, lan, tru, mass, u0, u1, u2 = epoch.to_numpy(), sma.to_numpy(), ecc.to_numpy(), inc.to_numpy(), ape.to_numpy(), lan.to_numpy(), tru.to_numpy() % (2 * np.pi), mass.to_numpy() - thruster_parameters.dry_mass, u0.to_numpy(), u1.to_numpy(), u2.to_numpy() 
    
    states = np.vstack((sma, ecc, inc, ape, lan, tru, mass, u0, u1, u2)).T
    states = get_rv_state_from_coe_state(pydylan.Body("Moon"), states)

    return epoch, states


# obtain the trajectory from the NRHO_ORBIT to a LOW_LUNAR_ORBIT using a qlaw method
def solve_qlaw(initial_epoch, initial_state, thruster_parameters):

    rad2deg =  180 / np.pi

    qlawhistory = "./qlaw_history.csv"
    if os.path.exists(qlawhistory): os.remove(qlawhistory)

    # used later for the total allowable integration time for qlaw
    t1, _ = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()
    t2, _ = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000()

    moon = pydylan.Body("Moon")
    initial_state = get_coe_state_from_rv_state(moon, initial_state)
    
    moon = pydylan.qlaw.body(moon.mu, moon.radius)
    coe_i = pydylan.qlaw.coe(initial_state[0], initial_state[1], initial_state[2] * rad2deg, initial_state[3] * rad2deg, initial_state[4] * rad2deg, initial_state[5] * rad2deg)
    coe_f = pydylan.qlaw.coe(4287, initial_state[1], 90)

    sc = pydylan.qlaw.thruster(thruster_parameters.thrust, thruster_parameters.Isp, thruster_parameters.fuel_mass + thruster_parameters.dry_mass)

    settings = pydylan.qlaw.QLaw_Parameters()
    
    settings.tol_sma = 10.
    settings.tol_inc = 1
    settings.tol_ecc = 0.1
    settings.Wsma = 0.45
    settings.Wecc = 0.1
    settings.Winc = 0.45
    settings.Wape = 0.0
    settings.Wlan = 0.0
    settings.Wp = 0.0
    settings.tld_limit = 2 * (t2 - t1)
    settings.stepsize = 1E-3
    settings.t0 = initial_epoch
    # settings.m_petro = 8.772
    # settings.n_petro = 4.209
    # settings.r_petro = 2.502
    # settings.k_petro = 7.378    
    # settings.b_petro = 0.01
    # settings.minimum_periapse = 100

    qlawsolver = pydylan.qlaw.QLaw5_Solver(moon, coe_i, coe_f, sc, settings)

    print(f"INITIAL COE -> sma: {initial_state[0]} km, ecc: {initial_state[1]}, inc: {initial_state[2] * rad2deg}, arg_per: {initial_state[3] * rad2deg}, lon_asc: {initial_state[4] * rad2deg}, tru_ano: {(initial_state[5] * rad2deg) % 360}")

    final_state = qlawsolver.solve()

    sma = final_state[0] / (1 - final_state[1] * final_state[1]) * moon.DU
    ecc = final_state[1]
    inc = final_state[2] * 180 / np.pi

    print(f"FINAL COE -> sma: {sma} km, ecc: {ecc}, inc: {final_state[2] * rad2deg}, arg_per: {final_state[3] * rad2deg}, lon_asc: {final_state[4] * rad2deg}, tru_ano: {(final_state[6] * rad2deg) % 360}")

    print(f"TOTAL TIME TAKEN DURING QLAW: {final_state[6] * moon.TU * seconds_to_days} ")

    assert(qlawsolver.status == pydylan.qlaw.QLaw5_Solver_Status.converged)
    assert(abs(sma - coe_f.sma) < settings.tol_sma)
    assert(abs(ecc - coe_f.ecc) < settings.tol_ecc)
    assert(abs(inc - coe_f.inc) < settings.tol_inc)

    epoch, states = read_qlaw_output(qlawhistory)

    return epoch, states


# obtain the trajectory from the NRHO_INSERTION_POINT to a NRHO_ORBIT by thrusting in the velocity direction
def solve_spiral(eom: pydylan.eom, thruster_parameters:pydylan.ThrustParameters, initial_epoch, initial_state, verbose=True) -> (np.ndarray, np.ndarray):

    moon = pydylan.Body("Moon")
    outward_spiral = False

    initial_epoch, initial_state = get_rv_state_relative_to_moon(initial_epoch, initial_state)

    current_epoch, current_state = initial_epoch, initial_state
    target_coe_state = np.array([ 4287 * 2.0 , 0.5714, np.pi / 2, 0., 0., 0.])
    
    error_weight = np.array([0.45, 0.1, 0.45, 0., 0., 0.])
    current_coe_state = get_coe_state_from_rv_state(moon, current_state)
    current_error = compute_error_in_state(current_coe_state, target_coe_state, error_weight)

    # First, compute the time-of-flight to reach within a tolerance of the desired coe state
    time_of_flight, dt, niter = 0, 100, 0
    while current_error > 5E-2 and niter<1E6:

        if ((niter % 100) == 0 and verbose): print(f"Iteration: {niter}, current error: {current_error}")

        time_of_flight = current_epoch + dt

        spiral_onto_NRHO = pydylan.phases.lowthrust_spiral(eom, current_state, thruster_parameters)
        spiral_onto_NRHO.evaluate(time_of_flight, outward_spiral, current_epoch, 10)
        current_epoch, current_state = time_of_flight, spiral_onto_NRHO.get_final_states(False)

        current_coe_state = get_coe_state_from_rv_state(moon, current_state)
        current_error = compute_error_in_state(current_coe_state, target_coe_state, error_weight)

        niter += 1

    # Then, recompute the spiral with the time-of-flight
    spiral_onto_NRHO = pydylan.phases.lowthrust_spiral(eom, initial_state, thruster_parameters)
    spiral_onto_NRHO.evaluate(time_of_flight, outward_spiral, initial_epoch, 10)

    return spiral_onto_NRHO.get_time(), spiral_onto_NRHO.get_states()


# obtain the trajectory from the POST_DEPLOYMENT_STATE to NRHO_INSERTION_STATE with Fixed+Fixed+Fixed BCs
def solve_transfer_fixed_BC(eom: pydylan.eom, thruster_parameters: pydylan.ThrustParameters) -> (np.ndarray, np.ndarray):

    initial_epoch, initial_state = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()
    middle_epoch, middle_state = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000()
    target_epoch, target_state = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000()

    left_boundary_condition = pydylan.FixedBoundaryCondition(initial_state)
    middle_boundary_condition = pydylan.FixedBoundaryCondition(middle_state)
    right_boundary_condition = pydylan.FixedBoundaryCondition(target_state)

    snopt_options = pydylan.SNOPT_options_structure()
    snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
    snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible
    snopt_options.quiet_SNOPT = False
    snopt_options.time_limit = 1.

    mbh_options = pydylan.MBH_options_structure()
    mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
    mbh_options.max_step_size = 1.
    mbh_options.quiet_MBH = False
    mbh_options.time_limit = 1.

    phase_options_1 = pydylan.phase_options_structure()
    phase_options_1.number_of_segments = 10

    phase_options_1.earliest_initial_date_in_MJS = initial_epoch
    phase_options_1.minimum_initial_coast_time = 0.
    phase_options_1.maximum_initial_coast_time = 0.
    phase_options_1.minimum_final_coast_time = 0.
    phase_options_1.maximum_final_coast_time = 0.
    phase_options_1.minimum_shooting_time = middle_epoch - initial_epoch
    phase_options_1.maximum_shooting_time = middle_epoch - initial_epoch

    phase_options_1.match_point_position_constraint_tolerance = 1E-3
    phase_options_1.match_point_velocity_constraint_tolerance = 1E-3
    phase_options_1.match_point_mass_constraint_tolerance = 1E-3
    phase_options_1.control_coordinate_transcription = pydylan.enum.polar

    phase_options_2 = pydylan.phase_options_structure()
    phase_options_2.number_of_segments = 100
    
    phase_options_2.earliest_initial_date_in_MJS = middle_epoch
    phase_options_2.minimum_initial_coast_time = 0.
    phase_options_2.maximum_initial_coast_time = 0.
    phase_options_2.minimum_final_coast_time = 0.
    phase_options_2.maximum_final_coast_time = 0.
    phase_options_2.minimum_shooting_time = target_epoch - middle_epoch
    phase_options_2.maximum_shooting_time = target_epoch - middle_epoch

    phase_options_2.match_point_position_constraint_tolerance = 1E-3
    phase_options_2.match_point_velocity_constraint_tolerance = 1E-3
    phase_options_2.match_point_mass_constraint_tolerance = 1E-3
    phase_options_2.control_coordinate_transcription = pydylan.enum.polar

    mission = pydylan.Mission(eom, left_boundary_condition, middle_boundary_condition, pydylan.enum.mbh)
    mission.set_thruster_parameters(thruster_parameters)
    mission.add_phase_options(phase_options_1)
    mission.add_control_initial_guess(p1_initial_guess_fixed_BC())
    mission.add_boundary_condition(right_boundary_condition)
    mission.add_phase_options(phase_options_2)
    mission.add_control_initial_guess(p2_initial_guess_fixed_BC())
    mission.optimize(snopt_options, mbh_options)

    output_control_to_screen(mission.get_control_state())
    
    assert mission.is_best_solution_feasible()
    np.save("./Plots_for_LIC/lic_lt_feasible_control_solutions_fixed_BC.npy", mission.get_all_feasible_control_solutions())

    result = mission.evaluate_and_return_solution(mission.get_control_state(), forward_backward_shooting=True)

    time = result.time
    states = result.states

    return time, states


# obtain the trajectory from the POST_DEPLOYMENT_STATE to NRHO_INSERTION_STATE with Fixed+Advecting+CR3BP BCs
def solve_transfer_other_BC(eom, thruster_parameters) -> (np.ndarray, np.ndarray):

    initial_epoch, initial_state = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()
    middle_epoch, middle_state = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000()
    target_epoch, target_state = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000()

    left_boundary_condition = pydylan.FixedBoundaryCondition(initial_state)
    middle_boundary_condition = pydylan.AdvectingBoundaryCondition(middle_state, middle_epoch, eom)
    right_boundary_condition = generate_CR3BP_boundary_condition()

    snopt_options = pydylan.SNOPT_options_structure()
    snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
    snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible
    snopt_options.quiet_SNOPT = False
    snopt_options.time_limit = 1 #60 * 15.

    mbh_options = pydylan.MBH_options_structure()
    mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
    # mbh_options.max_step_size = 1.
    mbh_options.quiet_MBH = False
    mbh_options.time_limit = 1 #60 * 60.

    phase_options_1 = pydylan.phase_options_structure()
    phase_options_1.number_of_segments = 10

    phase_options_1.earliest_initial_date_in_MJS = initial_epoch
    phase_options_1.minimum_initial_coast_time = 0.
    phase_options_1.maximum_initial_coast_time = 0.
    phase_options_1.minimum_final_coast_time = 0.
    phase_options_1.maximum_final_coast_time = 0.
    phase_options_1.minimum_shooting_time = 2 * (middle_epoch - initial_epoch) / 3
    phase_options_1.maximum_shooting_time = 4 * (middle_epoch - initial_epoch) / 3

    phase_options_1.match_point_position_constraint_tolerance = 1E+3
    phase_options_1.match_point_velocity_constraint_tolerance = 1E-2
    phase_options_1.match_point_mass_constraint_tolerance = 1E-3
    phase_options_1.control_coordinate_transcription = pydylan.enum.polar

    phase_options_2 = pydylan.phase_options_structure()
    phase_options_2.number_of_segments = 100
    
    phase_options_2.earliest_initial_date_in_MJS = middle_epoch
    phase_options_2.minimum_initial_coast_time = 0.
    phase_options_2.maximum_initial_coast_time = 0.
    phase_options_2.minimum_final_coast_time = 0.
    phase_options_2.maximum_final_coast_time = 0.
    phase_options_2.minimum_shooting_time = target_epoch - middle_epoch - (middle_epoch - initial_epoch) / 3
    phase_options_2.maximum_shooting_time = target_epoch - middle_epoch + (middle_epoch - initial_epoch) / 3

    phase_options_2.match_point_position_constraint_tolerance = 1E-4
    phase_options_2.match_point_velocity_constraint_tolerance = 1E-4
    phase_options_2.match_point_mass_constraint_tolerance = 1E-3
    phase_options_2.control_coordinate_transcription = pydylan.enum.polar

    mission = pydylan.Mission(eom, left_boundary_condition, middle_boundary_condition, pydylan.enum.mbh)
    mission.set_thruster_parameters(thruster_parameters)
    mission.add_phase_options(phase_options_1)
    mission.add_control_initial_guess(p1_initial_guess_other_BC())
    mission.add_boundary_condition(right_boundary_condition)
    mission.add_phase_options(phase_options_2)
    mission.add_control_initial_guess(p2_initial_guess_other_BC())
    mission.optimize(snopt_options, mbh_options)

    output_control_to_screen(mission.get_control_state())
    
    assert mission.is_best_solution_feasible()
    np.save("./Plots_for_LIC/lic_lt_feasible_control_solutions_other_BC.npy", mission.get_all_feasible_control_solutions())

    result = mission.evaluate_and_return_solution(mission.get_control_state(), forward_backward_shooting=True)

    time = result.time
    states = result.states

    return time, states


if __name__ == '__main__':

    zoomed_in = True # For visuals
    
    # Load SPICE kernels
    kernels = pydylan.spice.load_spice()

    earth = pydylan.Body("Earth")
    moon = pydylan.Body("Moon")
    eom_earth = pydylan.eom.Ephemeris_nBP(earth)
    eom_earth.add_secondary_body(moon)
    eom_moon = pydylan.eom.Ephemeris_nBP(moon)
    eom_moon.add_secondary_body(earth)

    initial_epoch, _ = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()

    thruster_parameters = pydylan.ThrustParameters(fuel_mass=1.5, dry_mass=12.5, Isp=2500., thrust=1.5E-3)

    time_transfer, states_transfer = solve_transfer_fixed_BC(eom_earth, thruster_parameters)

    thruster_parameters.fuel_mass = states_transfer[-1,6]
    time_spiral, states_spiral = solve_spiral(eom_moon, thruster_parameters, time_transfer[-1], states_transfer[-1, :6])

    thruster_parameters.fuel_mass = states_spiral[-1,6]
    time_qlaw, states_qlaw = solve_qlaw(time_spiral[-1], states_spiral[-1, :6], thruster_parameters)

    _, states_transfer_rel_moon = get_rv_state_relative_to_moon(time_transfer, states_transfer)
    _, states_spiral_rel_earth = get_rv_state_relative_to_earth(time_spiral, states_spiral)
    _, states_qlaw_rel_earth = get_rv_state_relative_to_earth(time_qlaw, states_qlaw)

    p, q = get_plots_of_phase_integration(earth, eom_earth)
    p.plot(xdata=states_transfer[:, 0], ydata=states_transfer[:, 1], color='black', linewidth=1)
    q.plot(xdata=states_transfer[:, 0], ydata=states_transfer[:, 2], color='black', linewidth=1)
    p.plot(xdata=states_spiral_rel_earth[:, 0], ydata=states_spiral_rel_earth[:, 1], color=strongorange, linewidth=1, alpha=0.9)
    q.plot(xdata=states_spiral_rel_earth[:, 0], ydata=states_spiral_rel_earth[:, 2], color=strongorange, linewidth=1, alpha=0.9)
    p.plot(xdata=states_qlaw_rel_earth[:, 0], ydata=states_qlaw_rel_earth[:, 1], color=babyblue, linewidth=1, alpha=0.8)
    q.plot(xdata=states_qlaw_rel_earth[:, 0], ydata=states_qlaw_rel_earth[:, 2], color=babyblue, linewidth=1, alpha=0.8)

    # # Mark the end of the spiral phase
    # p.plot(xdata=np.array([states_spiral_rel_earth[-1, 0], ]), ydata=np.array([states_spiral_rel_earth[-1, 1], ]), color='blue', marker='x', markersize=10, linewidth=2)
    # q.plot(xdata=np.array([states_spiral_rel_earth[-1, 0], ]), ydata=np.array([states_spiral_rel_earth[-1, 2], ]), color='blue', marker='x', markersize=10, linewidth=2)
    # # Mark the beginning of the qlaw phase
    # p.plot(xdata=np.array([states_qlaw_rel_earth[0, 0], ]), ydata=np.array([states_qlaw_rel_earth[0, 1], ]), color='green', marker='x', markersize=10, linewidth=2)
    # q.plot(xdata=np.array([states_qlaw_rel_earth[0, 0], ]), ydata=np.array([states_qlaw_rel_earth[0, 2], ]), color='green', marker='x', markersize=10, linewidth=2)

    m, n = get_plots_of_phase_integration(moon, eom_moon)
    m.plot(xdata=states_transfer_rel_moon[:, 0], ydata=states_transfer_rel_moon[:, 1], color='black', linewidth=1)
    n.plot(xdata=states_transfer_rel_moon[:, 0], ydata=states_transfer_rel_moon[:, 2], color='black', linewidth=1)
    m.plot(xdata=states_spiral[:, 0], ydata=states_spiral[:, 1], color=strongorange, linewidth=1, alpha=0.9)
    n.plot(xdata=states_spiral[:, 0], ydata=states_spiral[:, 2], color=strongorange, linewidth=1, alpha=0.9)
    m.plot(xdata=states_qlaw[:, 0], ydata=states_qlaw[:, 1], color=babyblue, linewidth=1, alpha=0.7)
    n.plot(xdata=states_qlaw[:, 0], ydata=states_qlaw[:, 2], color=babyblue, linewidth=1, alpha=0.7)

    # # Mark the end of the spiral phase
    # m.plot(xdata=np.array([states_spiral[-1, 0], ]), ydata=np.array([states_spiral[-1, 1], ]), color='blue', marker='x', markersize=10, linewidth=2)
    # n.plot(xdata=np.array([states_spiral[-1, 0], ]), ydata=np.array([states_spiral[-1, 2], ]), color='blue', marker='x', markersize=10, linewidth=2)
    # # Mark the beginning of the qlaw phase
    # m.plot(xdata=np.array([states_qlaw[0, 0], ]), ydata=np.array([states_qlaw[0, 1], ]), color='green', marker='x', markersize=10, linewidth=2)
    # n.plot(xdata=np.array([states_qlaw[0, 0], ]), ydata=np.array([states_qlaw[0, 2], ]), color='green', marker='x', markersize=10, linewidth=2)

    if zoomed_in==True:
        m.set_axis('tight', [-12000, 12000, -20000, 50000])
        n.set_axis('tight', [-50000, 50000, -60000, 25000])

    d1, _ = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()
    d1 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d1)
    d2, _ = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000()
    d2 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d2)
    d3, _ = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000()
    d3 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d3)
    d4 = time_spiral[-1]
    d4 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d4)
    d5 = time_qlaw[-1]
    d5 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d5)

    time_transfer, time_spiral, time_qlaw = (time_transfer - initial_epoch) * seconds_to_days, (time_spiral - initial_epoch) * seconds_to_days, (time_qlaw - initial_epoch) * seconds_to_days
    t = LinePlot()
    t.grid()
    t.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    t.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
    t.plot(xdata=time_transfer, ydata=get_throttle_history(states_transfer[:, -3:]), color='black', linewidth=2)

    t_spiral = LinePlot()
    t_spiral.grid()
    t_spiral.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    t_spiral.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
    t_spiral.plot(xdata=time_spiral, ydata=get_throttle_history(states_spiral[:, -3:]), color=strongorange, linewidth=2)

    t_qlaw = LinePlot()
    t_qlaw.grid()
    t_qlaw.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    t_qlaw.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
    t_qlaw.plot(xdata=time_qlaw, ydata=get_throttle_history(states_qlaw[:, -3:]), color=babyblue, linewidth=2)

    m = LinePlot()
    m.grid()
    m.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    m.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
    m.plot(xdata=time_transfer, ydata=states_transfer[:, 6], color='black', linewidth=2)

    m_spiral = LinePlot()
    m_spiral.grid()
    m_spiral.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    m_spiral.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
    m_spiral.plot(xdata=time_spiral, ydata=states_spiral[:, 6], color=strongorange, linewidth=2)

    m_qlaw = LinePlot()
    m_qlaw.grid()
    m_qlaw.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    m_qlaw.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
    m_qlaw.plot(xdata=time_qlaw, ydata=states_qlaw[:, 6], color=babyblue, linewidth=2)

    fig, axs = plt.subplots(3, 2, sharex=True)
    axs[0, 0].plot(time_transfer, states_transfer[:, 0])
    axs[0, 0].plot(time_spiral, states_spiral_rel_earth[:, 0], color=strongorange)
    axs[0, 0].plot(time_qlaw, states_qlaw_rel_earth[:, 0], color=babyblue)
    axs[0, 0].set_title('X-coordinate')
    axs[0, 0].grid()
    axs[1, 0].plot(time_transfer, states_transfer[:, 1])
    axs[1, 0].plot(time_spiral, states_spiral_rel_earth[:, 1], color=strongorange)
    axs[1, 0].plot(time_qlaw, states_qlaw_rel_earth[:, 1], color=babyblue)
    axs[1, 0].set_title('Y-coordinate')
    axs[1, 0].grid()
    axs[2, 0].plot(time_transfer, states_transfer[:, 2])
    axs[2, 0].plot(time_spiral, states_spiral_rel_earth[:, 2], color=strongorange)
    axs[2, 0].plot(time_qlaw, states_qlaw_rel_earth[:, 2], color=babyblue)
    axs[2, 0].set_title('Z-coordinate')
    axs[2, 0].grid()
    axs[0, 1].plot(time_transfer, states_transfer[:, 3])
    axs[0, 1].plot(time_spiral, states_spiral_rel_earth[:, 3], color=strongorange)
    axs[0, 1].plot(time_qlaw, states_qlaw_rel_earth[:, 3], color=babyblue)
    axs[0, 1].set_title('VX-coordinate')
    axs[0, 1].grid()
    axs[1, 1].plot(time_transfer, states_transfer[:, 4])
    axs[1, 1].plot(time_spiral, states_spiral_rel_earth[:, 4], color=strongorange)
    axs[1, 1].plot(time_qlaw, states_qlaw_rel_earth[:, 4], color=babyblue)
    axs[1, 1].set_title('VY-coordinate')
    axs[1, 1].grid()
    axs[2, 1].plot(time_transfer, states_transfer[:, 5])
    axs[2, 1].plot(time_spiral, states_spiral_rel_earth[:, 5], color=strongorange)
    axs[2, 1].plot(time_qlaw, states_qlaw_rel_earth[:, 5], color=babyblue)
    axs[2, 1].set_title('VZ-coordinate')
    axs[2, 1].grid()

    print('\n Transfer time: {} (days)'.format(time_transfer[-1]))
    print('\n Spiral time: {} (days)'.format(time_spiral[-1] - time_transfer[-1]))
    print('\n QLaw time: {} (days)'.format(time_qlaw[-1] - time_spiral[-1]))

    print('\n DEPLOYMENT: ', d1.year, '-', d1.month, '-', d1.day, '|', d1.hour, ':', d1.minute, ':', d1.second)
    print('\n POST-FLYBY: ', d2.year, '-', d2.month, '-', d2.day, '|', d2.hour, ':', d2.minute, ':', d2.second)
    print('\n NRHO-INSERTION: ', d3.year, '-', d3.month, '-', d3.day, '|', d3.hour, ':', d3.minute, ':', d3.second)
    print('\n END OF SPIRAL: ', d4.year, '-', d4.month, '-', d4.day, '|', d4.hour, ':', d4.minute, ':', d4.second)
    print('\n END OF QLAW: ', d5.year, '-', d5.month, '-', d5.day, '|', d5.hour, ':', d5.minute, ':', d5.second)

    print('\n Mass Remaining at the end of NRHO-INSERTION', states_transfer[-1, 6], ' kg')
    print('\n Mass Remaining at the end of SPIRAL', states_spiral[-1, 6], ' kg')
    print('\n Mass Remaining at the end of QLAW', states_qlaw[-1, 6], ' kg')

    plt.show()

    # Unload SPICE kernels
    pydylan.spice.unload_spice(kernels)
