import numpy as np
import copy as cp
import json
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class CR3BPInitGenerator:

    def __init__(self, sample_mode):
        self.sample_mode = sample_mode

        self._load_in_common_initial_guess()

    def get_earth_initial_guess(self, seed, sample_num, number_of_segments, maximum_shooting_time):
        """

        :param seed:
        :param num:
        :return: a list of initial guess for earth mission, specified by seed and number of initial guess required
        """

        # first three are times, then have 10 x 3 controls, with 10 x 2 angles and 10 x 1 radius. final one is mass
        if self.sample_mode == "example":
            if number_of_segments != 10:
                Warning("number of segments is not 10!")
                exit()
            optimal_earth_initial_guess = cp.copy(self.example_earth_initial_guess)
            return [optimal_earth_initial_guess]

        if self.sample_mode == "uniform_control":
            random_state = np.random.RandomState(seed=seed)
            theta = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
            psi = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
            r = random_state.uniform(0, 1, number_of_segments * sample_num)

            earth_initial_guess_list = []
            for i in range(sample_num):
                earth_initial_guess = []
                for j in range(3):
                    earth_initial_guess.append(self.example_earth_initial_guess[j])
                for j in range(number_of_segments):
                    earth_initial_guess.append(theta[i * number_of_segments + j])
                    earth_initial_guess.append(psi[i * number_of_segments + j])
                    earth_initial_guess.append(r[i * number_of_segments + j])
                earth_initial_guess.append(300)  # TODO: hack of final mass initialization
                earth_initial_guess = np.asarray(earth_initial_guess)
                earth_initial_guess_list.append(earth_initial_guess)
            return earth_initial_guess_list

        if self.sample_mode == "uniform_control_time":
            random_state = np.random.RandomState(seed=seed)
            theta = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
            psi = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
            r = random_state.uniform(0, 1, number_of_segments * sample_num)

            t_shooting = random_state.uniform(0, maximum_shooting_time, sample_num)
            t_init = random_state.uniform(0, 15.0, sample_num)
            t_final = random_state.uniform(0, 15.0, sample_num)

            earth_initial_guess_list = []
            for i in range(sample_num):
                earth_initial_guess = []

                # append time and control initial guess
                earth_initial_guess.append(t_shooting[i])
                earth_initial_guess.append(t_init[i])
                earth_initial_guess.append(t_final[i])
                for j in range(number_of_segments):
                    earth_initial_guess.append(theta[i * number_of_segments + j])
                    earth_initial_guess.append(psi[i * number_of_segments + j])
                    earth_initial_guess.append(r[i * number_of_segments + j])
                earth_initial_guess.append(300)  # TODO: hack of final mass initialization
                earth_initial_guess = np.asarray(earth_initial_guess)
                earth_initial_guess_list.append(earth_initial_guess)
            return earth_initial_guess_list

        if self.sample_mode == "uniform_control_time_mass":
            random_state = np.random.RandomState(seed=seed)
            theta = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
            psi = random_state.uniform(0, 2 * np.pi, number_of_segments * sample_num)
            r = random_state.uniform(0, 1, number_of_segments * sample_num)

            t_shooting = random_state.uniform(0, maximum_shooting_time, sample_num)
            t_init = random_state.uniform(0, 15.0, sample_num)
            t_final = random_state.uniform(0, 15.0, sample_num)

            mass = random_state.uniform(300, 450, sample_num)  # TODO: hack of final mass range

            earth_initial_guess_list = []
            for i in range(sample_num):
                earth_initial_guess = []

                # append time and control initial guess
                earth_initial_guess.append(t_shooting[i])
                earth_initial_guess.append(t_init[i])
                earth_initial_guess.append(t_final[i])

                for j in range(number_of_segments):
                    earth_initial_guess.append(theta[i * number_of_segments + j])
                    earth_initial_guess.append(psi[i * number_of_segments + j])
                    earth_initial_guess.append(r[i * number_of_segments + j])

                earth_initial_guess.append(mass[i])

                earth_initial_guess = np.asarray(earth_initial_guess)
                earth_initial_guess_list.append(earth_initial_guess)
            return earth_initial_guess_list

        if self.sample_mode == "independent_truncated_norm_control_time_mass":
            # Here we choose the mean and variance from the solution get from seed 0 to seed 9, with final mass over its average 398.27
            # /home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/solution/2022-06-17/seed_0_sample_num_200_mode_uniform_control_time_mass.pkl
            variable_mean_std_path = "/home/anjian/Desktop/project/global_optimization/pydylan-wrapper/result/statistics/uniform_sample/2000_samples_uniform_sample_above_average_mean_std.json"
            with open(variable_mean_std_path, 'r') as f:
                variable_mean_std = json.load(f)

            np.random.seed(seed=seed)

            t_shooting = get_truncated_normal(mean=variable_mean_std["shooting_time_mean"], sd=variable_mean_std["shooting_time_std"], low=1.57, upp=10.0).rvs(sample_num)
            t_init = get_truncated_normal(mean=variable_mean_std["init_coast_time_mean"], sd=variable_mean_std["init_coast_time_std"], low=0.0, upp=15.0).rvs(sample_num)
            t_final = get_truncated_normal(mean=variable_mean_std["final_coast_time_mean"], sd=variable_mean_std["final_coast_time_std"], low=0.0, upp=15.0).rvs(sample_num)

            theta = np.random.uniform(0, 2 * np.pi, number_of_segments * sample_num)
            psi = np.random.uniform(0, 2 * np.pi, number_of_segments * sample_num)

            # r size: num_of_segments x num_of_samples
            r = []
            for i in range(number_of_segments):
                r.append(get_truncated_normal(mean=variable_mean_std["control_radius_mean"][i], sd=variable_mean_std["control_radius_std"][i], low=0.0, upp=1.0).rvs(sample_num))
            r = np.asarray(r)

            mass = get_truncated_normal(mean=variable_mean_std["final_mass_mean"], sd=variable_mean_std["final_mass_std"], low=398.36, upp=429.86).rvs(sample_num)

            earth_initial_guess_list = []
            for i in range(sample_num):
                earth_initial_guess = []

                # append time and control initial guess
                earth_initial_guess.append(t_shooting[i])
                earth_initial_guess.append(t_init[i])
                earth_initial_guess.append(t_final[i])

                for j in range(number_of_segments):
                    earth_initial_guess.append(theta[i * number_of_segments + j])
                    earth_initial_guess.append(psi[i * number_of_segments + j])
                    earth_initial_guess.append(r[j, i])

                earth_initial_guess.append(mass[i])

                earth_initial_guess = np.asarray(earth_initial_guess)
                earth_initial_guess_list.append(earth_initial_guess)
                
            return earth_initial_guess_list

    def _load_in_common_initial_guess(self):

        self.example_earth_initial_guess = np.array([5.457582872128189,  # T
                                                     10.896135684377569,  # t0
                                                     14.306108632635908,  # t1
                                                     2.32832474320574,  # theta
                                                     2.399459560315589,  # alpha
                                                     1.0,  # radius
                                                     2.4573668767386687,
                                                     2.2545175879577015,
                                                     0.3719497426893984,
                                                     4.090146221967156,
                                                     3.7487405762327692,
                                                     0.0,
                                                     6.283185307179586,
                                                     5.176389314553339,
                                                     0.0,
                                                     4.924338646471816,
                                                     3.9828146069171977,
                                                     0.0,
                                                     3.3043157193384887,
                                                     3.412908536577056,
                                                     0.06610651864045772,
                                                     3.5263682837974746,
                                                     2.1933390102978882,
                                                     0.0,
                                                     4.5034060986351045,
                                                     1.8885061158250744,
                                                     0.3312640487820477,
                                                     6.283185307179586,
                                                     1.0987406889436244,
                                                     0.0,
                                                     4.331789430856885,
                                                     3.8924833028542905,
                                                     0.0,
                                                     415.4310254680328])  # mass

        self.zero_earth_initial_guess = np.array([5.457582872128189,  # T
                                                  10.896135684377569,  # t0
                                                  14.306108632635908,  # t1
                                                  0.0,  # theta
                                                  0.0,  # alpha
                                                  0.0,  # radius
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  415.4310254680328])

        self.example_moon_initial_guess = np.array([1.8486003452500226,  # T
                                                    1.4346239757077044,  # t0
                                                    6.887850988166001,  # t1
                                                    1.0510463134761914,  # theta
                                                    2.9706068982756784,  # alpha
                                                    0.3622163933745785,  # r
                                                    1.1209469082255838,
                                                    3.002058780469406,
                                                    0.0,
                                                    3.6034244738512657,
                                                    2.0725259485500285,
                                                    0.0,
                                                    3.3727250670623348,
                                                    3.677590975998773,
                                                    0.0,
                                                    1.0394286939112873,
                                                    3.934959710080541,
                                                    0.0,
                                                    0.7267237767413558,
                                                    4.752583784860011,
                                                    0.0,
                                                    0.45310140443856334,
                                                    3.1233670013809065,
                                                    0.0,
                                                    2.6709735053156285,
                                                    0.5387626496715113,
                                                    0.0,
                                                    4.17346768216723,
                                                    5.524355422277355,
                                                    0.0,
                                                    4.750089842290181,
                                                    4.186196063180113,
                                                    0.0,
                                                    4.472501838563253,
                                                    3.091954205608971,
                                                    1.0,
                                                    4.144176628260902,
                                                    2.955444713084693,
                                                    0.454873716306567,
                                                    0.8885494769195953,
                                                    6.283185307179586,
                                                    0.0,
                                                    6.283185307179586,
                                                    3.3601530506562014,
                                                    0.0,
                                                    6.283185307179586,
                                                    1.8211287560823008,
                                                    0.0,
                                                    6.283185307179586,
                                                    2.4091492891789303,
                                                    0.0,
                                                    4.926642209999231,
                                                    3.3276523993694536,
                                                    0.062306159146820646,
                                                    4.975369356305258,
                                                    3.3552406865524365,
                                                    1.0,
                                                    4.995232765211097,
                                                    3.38616249988325,
                                                    0.3691044395568039,
                                                    4.776999120831957,
                                                    3.0347866499384133,
                                                    0.0,
                                                    404.35893613364016])
