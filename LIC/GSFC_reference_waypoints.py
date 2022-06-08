import sys, os

sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding/debug')
sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding')

import pydylan
from numpy import ndarray, array


def get_post_deployment_epoch_in_JD_and_state_in_EJ2000() -> (float, ndarray):
    """ Julian Date epoch and position (km), velocity (km/s) in the Earth J2000 frame"""
    epoch = 2459487.4348728
    state = array([-113733.5263697117800000,
                   58562.5333572497690000,
                   51414.3169829583060000,
                   -1.8345926622363595,
                   0.4319560623021563,
                   0.4251972981399922])

    return epoch, state


def get_post_deployment_epoch_in_MJS_and_state_in_EJ2000() -> (float, ndarray):
    """ MJS epoch and position (km), velocity (km/s) in the Earth J2000 frame"""
    epoch, state = get_post_deployment_epoch_in_JD_and_state_in_EJ2000()
    epoch_MJS = pydylan.epoch.convert_JD_to_relative_MJS(epoch)

    return epoch_MJS, state


def get_post_flyby_epoch_in_JD_and_state_in_EJ2000() -> (float, ndarray):
    """ Julian Date epoch and position (km), velocity (km/s) in the Earth J2000 frame"""
    epoch = 2459493.01916523
    state = array([-281493.2578310815700000,
                   -8597.5191897981294000,
                   24463.1335824873280000,
                   1.0163750929632283,
                   -1.0264253284952656,
                   -0.5420657946153973])

    return epoch, state


def get_post_flyby_epoch_in_MJS_and_state_in_EJ2000() -> (float, ndarray):
    """ MJS epoch and position (km), velocity (km/s) in the Earth J2000 frame"""
    epoch, state = get_post_flyby_epoch_in_JD_and_state_in_EJ2000()
    epoch_MJS = pydylan.epoch.convert_JD_to_relative_MJS(epoch)

    return epoch_MJS, state


def get_NRHO_insertion_epoch_in_JD_and_state_in_EJ2000() -> (float, ndarray):
    """ Julian Date epoch and position (km), velocity (km/s) in the Earth J2000 frame"""
    epoch = 2459672.78958612
    state = array([334996.3583319024200000,
                   271357.4125142795400000,
                   35861.5301774528560000,
                   -0.5712115877087332,
                   0.5995965476128268,
                   0.4304749341519304])

    return epoch, state


def get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000() -> (float, ndarray):
    """ Julian Date epoch and position (km), velocity (km/s) in the Earth J2000 frame"""
    epoch, state = get_NRHO_insertion_epoch_in_JD_and_state_in_EJ2000()
    epoch_MJS = pydylan.epoch.convert_JD_to_relative_MJS(epoch)

    return epoch_MJS, state