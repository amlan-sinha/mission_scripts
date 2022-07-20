import pickle
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import normalize, minmax_scale
import json
import os
import datetime

def main():
    # uniform sampling
    # file_name_list = [
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-17/seed_0_sample_num_200_mode_uniform_control_time_mass.pkl",
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-17/seed_1_sample_num_200_mode_uniform_control_time_mass.pkl",
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-17/seed_2_sample_num_200_mode_uniform_control_time_mass.pkl",
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-17/seed_3_sample_num_200_mode_uniform_control_time_mass.pkl",
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-18/seed_4_sample_num_200_mode_uniform_control_time_mass.pkl",
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-18/seed_5_sample_num_200_mode_uniform_control_time_mass.pkl",
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-18/seed_6_sample_num_200_mode_uniform_control_time_mass.pkl",
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-18/seed_7_sample_num_200_mode_uniform_control_time_mass.pkl",
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-18/seed_8_sample_num_200_mode_uniform_control_time_mass.pkl",
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-18/seed_9_sample_num_200_mode_uniform_control_time_mass.pkl"]

    # # independent truncated norm
    # file_name_list = [
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-20/seed_0_sample_num_200_mode_independent_truncated_norm_control_time_mass.pkl",
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-20/seed_1_sample_num_200_mode_independent_truncated_norm_control_time_mass.pkl",
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-20/seed_2_sample_num_200_mode_independent_truncated_norm_control_time_mass.pkl",
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-20/seed_3_sample_num_200_mode_independent_truncated_norm_control_time_mass.pkl",
    #     "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-20/seed_4_sample_num_200_mode_independent_truncated_norm_control_time_mass.pkl"
    # ]

    file_name_list = ["/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-07-13/seed_0_sample_num_10_mode_uniform_control_time_mass_start_bdry_6.48423370092_end_bdry_8.0_thrust_0.9.pkl"]

    data_list = []
    for file in file_name_list:
        f = open(file, 'rb')
        data_list += pickle.load(f)

    print(len(data_list))

    num_total = len(file_name_list) * 200
    num_segments = 20

    num_feasible = 0
    mass = []
    shooting_time = []
    init_coast_time = []
    final_coast_time = []
    control_list = []
    all_variables = []

    for data in data_list:
        if data["feasibility"]:
        # if data["feasibility"] and data["results.control"][-1] >= 398.27: # TODO, manually set all good samples
            num_feasible += 1
            variable = data["results.control"]
            mass.append(variable[-1])
            shooting_time.append(variable[0])
            init_coast_time.append(variable[1])
            final_coast_time.append(variable[2])

            # Control, the second half need to reverse
            current_control = variable[3: 11 * 3]
            for i in range(10):
                index = 20 - i
                current_control = np.append(current_control, variable[index * 3: (index + 1) * 3])
            control_list.append([current_control])

            # control_list.append([variable[3: -1]])

            all_variables.append(variable)

    mass = np.asarray(mass)
    shooting_time = np.asarray(shooting_time)
    init_coast_time = np.asarray(init_coast_time)
    final_coast_time = np.asarray(final_coast_time)
    control = np.asarray(control_list).reshape(num_feasible, num_segments, 3)
    all_variables = np.asarray(all_variables)
    print(np.shape(control))
    print(np.shape(all_variables))

    print("feasible solution number is {:d}, ratio is {:.2f}".format(num_feasible, num_feasible / num_total))
    print("mass average is {:.2f}, median is {:.2f}, max is {:.2f}, min is {:.2f}, std is {:.2f}".format(
        np.mean(mass), np.median(mass), np.max(mass), np.min(mass), np.std(mass)))
    print("shooting_time average is {:.2f}, median is {:.2f}, max is {:.2f}, min is {:.2f}, std is {:.2f}".format(
        np.mean(shooting_time),
        np.median(shooting_time),
        np.max(shooting_time),
        np.min(shooting_time),
        np.std(shooting_time)))
    print("init_coast_time average is {:.2f}, median is {:.2f}, max is {:.2f}, min is {:.2f}, std is {:.2f}".format(
        np.mean(init_coast_time),
        np.median(init_coast_time),
        np.max(init_coast_time),
        np.min(init_coast_time),
        np.std(init_coast_time)))
    print(
        "final_coast_time average is {:.2f}, median is {:.2f}, max is {:.2f}, min is {:.2f}, std is {:.2f}".format(
            np.mean(final_coast_time),
            np.median(final_coast_time),
            np.max(final_coast_time),
            np.min(final_coast_time),
            np.std(final_coast_time)))

    radius = np.squeeze(control[:, :, -1])
    # print(np.shape(radius))
    print("the mean and the std of the control radius over 20 timesteps")
    print(np.mean(radius, axis=0))
    print(np.std(radius, axis=0))
    print("the max and min of the control radius over 20 timesteps")
    print(np.max(radius, axis=0))
    print(np.min(radius, axis=0))

    # Plot
    # Shooting time histogram
    fig, ax = plt.subplots()
    ax.hist(shooting_time)
    ax.set_xlim(0, 10)
    ax.set_title("histogram for shooting time")
    ax.set_xlabel("time")
    ax.set_ylabel("number")

    # init time histogram
    fig, ax = plt.subplots()
    ax.hist(init_coast_time)
    ax.set_xlim(0, 10)
    ax.set_title("histogram for initial coast time")
    ax.set_xlabel("time")
    ax.set_ylabel("number")

    # final time histogram
    fig, ax = plt.subplots()
    ax.hist(final_coast_time)
    ax.set_xlim(0, 10)
    ax.set_title("histogram for final cost time")
    ax.set_xlabel("time")
    ax.set_ylabel("number")

    # # Mean and variance for radius
    # fig, ax = plt.subplots()
    # epochs = list(range(20))
    # meanst = np.mean(radius, axis=0)
    # var = np.var(radius, axis=0)
    # ax.plot(epochs, meanst, label="control radius mean")
    # ax.fill_between(epochs, meanst - var, meanst + var, alpha=0.3)
    # ax.set_xlabel("time steps")
    # ax.set_ylabel("control radius")
    # ax.set_title("control radius mean - variance")
    # ax.legend()
    # plt.show()

    # 95% confidence level for radius
    radius_df = pd.DataFrame(radius, columns=['{:d}'.format(i) for i in range(num_segments)])
    fig, ax = plt.subplots()
    df = pd.melt(frame=radius_df,
                 var_name='timestep',
                 value_name='control radius')
    sns.lineplot(ax=ax,
                 data=df,
                 x='timestep',
                 y='control radius',
                 sort=False).set(title='95% confidence intervel of control radius')


    # non-zero num for radius
    non_zero_radius = np.zeros((20, 1))
    for i in range(np.shape(radius)[0]):
        for j in range(np.shape(radius)[1]):
            if radius[i][j] > 0:
                non_zero_radius[j] += 1

    fig, ax = plt.subplots()
    epochs = list(range(20))
    ax.plot(epochs, non_zero_radius)
    ax.set_title("control radius non zero number, total: {:d}".format(num_feasible))
    ax.set_xlabel("time steps")
    ax.set_ylabel("control radius non zero number")

    # Covariance matrix for all variables
    # all_variables = normalize(all_variables, axis=0, norm='max')
    # all_variables = minmax_scale(all_variables, axis=0, feature_range=(-1, 1))
    df = pd.DataFrame(all_variables)
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)

    plt.show()

    print(
        "=======================================================================================================================================")

    # # Save basic statistics，an and std of each variables
    # solution_variable_mean_std = {
    #     "shooting_time_mean": np.mean(shooting_time),
    #     "shooting_time_std" : np.std(shooting_time),
    #     "init_coast_time_mean": np.mean(init_coast_time),
    #     "init_coast_time_std": np.std(init_coast_time),
    #     "final_coast_time_mean": np.mean(final_coast_time),
    #     "final_coast_time_std": np.std(final_coast_time),
    #     "control_radius_mean": np.mean(radius, axis=0).tolist(),
    #     "control_radius_std": np.std(radius, axis=0).tolist(),
    #     "final_mass_mean": np.mean(mass),
    #     "final_mass_std": np.std(mass),
    # }

    # parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # path = parent_path + "/result/statistics/uniform_sample"
    # if not os.path.isdir(path):
    #     os.makedirs(path)
    # file_name = "2000_samples_uniform_sample_above_average_mean_std.json"
    # full_file_path_name = path + "/" + file_name

    # with open(full_file_path_name, 'w') as f:
    #     json.dump(solution_variable_mean_std, f, indent=2)
    # print("Statistics successfully saved!")

    print(
        "=======================================================================================================================================")

if __name__ == "__main__":
    main()