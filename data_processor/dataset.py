import pickle
import numpy as np
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import normalize, minmax_scale
import json
import os
import datetime


def main():
    file_name_list = [
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_0_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_1_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_2_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_3_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_4_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_5_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_6_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_7_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_8_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_9_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_10_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_11_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_12_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_13_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_14_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_15_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_16_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_17_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_200_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_201_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_202_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_203_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_204_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_205_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_206_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_207_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_208_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_209_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-07/seed_210_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-08/seed_18_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-08/seed_19_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-08/seed_211_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-08/seed_212_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-08/seed_213_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-08/seed_214_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-08/seed_215_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-08/seed_216_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-08/seed_217_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-08/seed_218_sample_num_100_mode_uniform_control_time_mass.pkl",
        "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-08/seed_219_sample_num_100_mode_uniform_control_time_mass.pkl",
    ]

    data_list = []
    for file in file_name_list:
        f = open(file, 'rb')
        data_list += pickle.load(f)

    print(len(data_list))

    num_feasible = 0
    final_control_solution_list = []
    raw_snopt_control_evaluations_list = []
    snopt_inform_list = []

    # Get all feasible data ####################################################################################################
    # for data in data_list:
    #     if data["feasibility"]:
    #         num_feasible += 1
    #         final_control_solution_list.append(data["results.control"])
    #         raw_snopt_control_evaluations_list.append(data["snopt_control_evaluations"])
    #         snopt_inform_list.append(data["snopt_inform"])
    #
    # print("feasible solution number is", num_feasible)
    # # print("final control solution is")
    # # print(final_control_solution_list)
    #
    # # filter raw snopt control evaluation
    # snopt_control_evaluations_list = []
    # num_raw_snopt_control_evaluation = 0
    # num_file = 0
    # for raw_snopt_control_evaluations in raw_snopt_control_evaluations_list:
    #     snopt_control_evaluations_list.append(filter_raw_snopt_control_evaluation(raw_snopt_control_evaluations))
    #     num_raw_snopt_control_evaluation += np.shape(raw_snopt_control_evaluations)[0]
    #     print("before filtering, ", np.shape(raw_snopt_control_evaluations)[0])
    #     print("after filtering, ", np.shape(filter_raw_snopt_control_evaluation(raw_snopt_control_evaluations))[0])
    #     num_file += 1
    #     if num_file % 10 == 0:
    #         print(num_file)
    #
    # for raw_snopt_control_evaluations in raw_snopt_control_evaluations_list:
    #     num_raw_snopt_control_evaluation += np.shape(raw_snopt_control_evaluations)[0]
    #
    # print("raw feasible snopt control evaluation has size of ", num_raw_snopt_control_evaluation)
    #
    # num_filtered_snopt_control_evaluation = 0
    # for snopt_control_evaluations in snopt_control_evaluations_list:
    #     num_filtered_snopt_control_evaluation += np.shape(snopt_control_evaluations)[0]
    #
    # print("filtered feasible snopt control evaluation has size of ", num_filtered_snopt_control_evaluation)

    # Get all local optimal data ####################################################################################################
    snopt_inform_dict = {}
    num_feasible = 0
    for data in data_list:
        if data["feasibility"]:
            snopt_inform_list.append(data["snopt_inform"])
            num_feasible += 1
    print("feasible solution number is", num_feasible)

    for inform_num in snopt_inform_list:
        if "{:d}".format(inform_num) in snopt_inform_dict:
            snopt_inform_dict["{:d}".format(inform_num)] += 1
        else:
            snopt_inform_dict["{:d}".format(inform_num)] = 1

    print(snopt_inform_list)
    print(snopt_inform_dict)

    for data in data_list:
        if data["feasibility"] and data["snopt_inform"] == 1:
            final_control_solution_list.append(data["results.control"])
            raw_snopt_control_evaluations_list.append(data["snopt_control_evaluations"])

    # filter raw snopt control evaluation
    snopt_control_evaluations_list = []
    num_raw_snopt_control_evaluation = 0
    num_file = 0
    for raw_snopt_control_evaluations in raw_snopt_control_evaluations_list:
        snopt_control_evaluations_list.append(filter_raw_snopt_control_evaluation(raw_snopt_control_evaluations))
        num_raw_snopt_control_evaluation += np.shape(raw_snopt_control_evaluations)[0]
        # print("before filtering, ", np.shape(raw_snopt_control_evaluations)[0])
        # print("after filtering, ", np.shape(filter_raw_snopt_control_evaluation(raw_snopt_control_evaluations))[0])
        num_file += 1
        if num_file % 10 == 0:
            print(num_file)

    for raw_snopt_control_evaluations in raw_snopt_control_evaluations_list:
        num_raw_snopt_control_evaluation += np.shape(raw_snopt_control_evaluations)[0]

    print("raw optimal snopt control evaluation has size of ", num_raw_snopt_control_evaluation)

    num_filtered_snopt_control_evaluation = 0
    for snopt_control_evaluations in snopt_control_evaluations_list:
        num_filtered_snopt_control_evaluation += np.shape(snopt_control_evaluations)[0]

    print("filtered optimal snopt control evaluation has size of ", num_filtered_snopt_control_evaluation)

def filter_raw_snopt_control_evaluation(raw_eval):
    distance_matrix = get_distance_matrix(raw_eval, raw_eval)
    # print(np.shape(distance_matrix)[0])

    index_set = []
    for i in range(np.shape(distance_matrix)[0]):
        if i == 0:
            index_set.append(i)
            continue
        else:
            if np.min(distance_matrix[:i, i]) < 1:
                continue
            else:
                index_set.append(i)

    # print(len(index_set))
    # print(index_set)

    return raw_eval[index_set, :]


def get_distance_matrix(A, B, squared=False):
    ## adpted from https://www.dabblingbadger.com/blog/2020/2/27/implementing-euclidean-distance-matrix-calculations-from-scratch-in-python
    """
    Compute all pairwise distances between vectors in A and B.

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.

    See also
    --------
    A more generalized version of the distance matrix is available from
    scipy (https://www.scipy.org) using scipy.spatial.distance_matrix,
    which also gives a choice for p-norm.
    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A * A).sum(axis=1).reshape((M, 1)) * np.ones(shape=(1, N))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(M, 1))
    D_squared = A_dots + B_dots - 2 * A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared


if __name__ == "__main__":
    main()
