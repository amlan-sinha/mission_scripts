import numpy as np
import scipy as sc
import json


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return sc.stats.truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class CR3BPInitGenerator:

    def __init__(self, sample_mode):

        self.sample_mode = sample_mode

    def get_earth_initial_guess(self, seed, sample_num, number_of_segments, maximum_shooting_time):
        """

        :param seed:
        :param num:
        :return: a list of initial guess for earth mission, specified by seed and number of initial guess required
        """

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

        if self.sample_mode == "resonant_feasible_solution":

            # -----------------------------------------------------------------
            # 3to4 to 2to3
            # -----------------------------------------------------------------

            # return [np.array([9.70783430e+00, 9.91890175e-01, 5.60405139e-03, 2.36511300e-02,
            # 6.21695820e+00, 1.00000000e+00, 4.71429548e-03, 6.14407102e+00,
            # 9.99849198e-01, 3.83257073e-03, 0.00000000e+00, 9.99780892e-01,
            # 1.77045722e-01, 6.20085618e+00, 9.89612709e-01, 3.13412779e+00,
            # 7.32530533e-04, 9.34806522e-01, 0.00000000e+00, 0.00000000e+00,
            # 7.64194853e-02, 5.68485420e+00, 2.11237686e-03, 9.91253538e-01,
            # 5.38767842e+00, 3.93138718e+00, 4.46949465e-01, 3.79105915e+00,
            # 0.00000000e+00, 3.20284346e-01, 4.78060609e+00, 2.17396352e+00,
            # 7.97075298e-01, 3.95712047e+00, 0.00000000e+00, 3.59663474e-01,
            # 5.61884584e-02, 8.46783450e-01, 5.75377698e-01, 6.05048088e+00,
            # 5.65889482e+00, 9.46676225e-01, 7.76528856e-01, 2.18585181e+00,
            # 3.92266443e-01, 9.35341528e-03, 2.79663976e+00, 4.66376268e-01,
            # 2.48497316e-01, 3.44870210e+00, 9.63447920e-01, 1.01137950e-02,
            # 5.79628036e+00, 5.72740171e-03, 9.64184691e-01, 4.60466776e-03,
            # 1.00000000e+00, 1.30085315e-01, 1.78434195e-03, 4.04389935e-01,
            # 0.00000000e+00, 4.18943078e+00, 3.90155599e-01, 4.57468512e+02])]
            # return [np.array([8.03763314e+00, 9.65193717e-01, 9.03142876e-01, 4.85824464e+00,
            # 6.28318531e+00, 3.97842272e-01, 1.94294117e+00, 5.44442571e+00,
            # 0.00000000e+00, 0.00000000e+00, 2.41214030e+00, 0.00000000e+00,
            # 0.00000000e+00, 2.70848694e+00, 0.00000000e+00, 0.00000000e+00,
            # 5.63411078e+00, 0.00000000e+00, 0.00000000e+00, 2.26076831e+00,
            # 0.00000000e+00, 4.31650314e+00, 0.00000000e+00, 5.86457763e-01,
            # 4.66546813e+00, 0.00000000e+00, 1.00000000e+00, 6.28318531e+00,
            # 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.03594624e+00,
            # 0.00000000e+00, 3.16407493e+00, 3.14160729e+00, 4.19015903e-01,
            # 6.28318531e+00, 1.29970044e+00, 0.00000000e+00, 5.92524194e+00,
            # 6.28318531e+00, 1.00000000e+00, 3.30982186e+00, 6.05307415e+00,
            # 0.00000000e+00, 3.11306061e-02, 5.95966251e+00, 0.00000000e+00,
            # 0.00000000e+00, 1.75077398e+00, 0.00000000e+00, 1.04080489e+00,
            # 1.97157394e+00, 0.00000000e+00, 4.69783640e+00, 5.22408287e+00,
            # 0.00000000e+00, 0.00000000e+00, 6.53943098e-02, 0.00000000e+00,
            # 0.00000000e+00, 3.55631356e-01, 0.00000000e+00, 6.47672252e+02])]
            # return [np.array([9.85230085e+00, 5.58218551e-01, 9.90658239e-01, 6.28119965e+00,
            # 6.28254431e+00, 7.61322121e-01, 1.13820320e+00, 6.28022151e+00,
            # 7.38541837e-01, 3.21410733e-01, 2.68942643e-01, 6.78240064e-01,
            # 6.09122480e+00, 1.79129704e+00, 9.04386884e-01, 5.53332312e+00,
            # 5.88154796e+00, 9.89393002e-01, 0.00000000e+00, 5.80825519e+00,
            # 6.08088127e-01, 6.16084002e+00, 5.05237072e-02, 1.27071150e-02,
            # 1.56412377e+00, 4.32594905e+00, 6.67800791e-01, 6.74148325e-01,
            # 2.31282909e+00, 1.14535421e-02, 0.00000000e+00, 6.28318531e+00,
            # 0.00000000e+00, 4.85519427e-02, 9.32208710e-03, 4.61436097e-03,
            # 6.26063967e+00, 0.00000000e+00, 8.76010750e-03, 6.16542564e+00,
            # 9.45792643e-01, 7.28880512e-01, 5.75728051e-01, 2.97256682e+00,
            # 6.64732956e-01, 6.40934825e-01, 1.24834669e-01, 1.49372186e-01,
            # 1.05657593e+00, 6.03907662e+00, 9.99922356e-01, 1.20530000e+00,
            # 2.12415748e-01, 6.71817022e-01, 1.31497905e+00, 6.53344288e-01,
            # 3.22606700e-01, 5.32602978e+00, 1.02270793e-01, 3.72936634e-01,
            # 3.76874001e+00, 1.17405914e+00, 7.60553928e-01, 5.10474691e+02])]
            # return [np.array([9.18169090e+00, 0.00000000e+00, 0.00000000e+00, 1.67341948e+00,
            # 6.28318321e+00, 1.00000000e+00, 7.26578705e-01, 3.54187841e+00,
            # 0.00000000e+00, 1.17593302e+00, 1.15317596e+00, 0.00000000e+00,
            # 1.56338377e-01, 0.00000000e+00, 3.18504200e-01, 4.07130497e-02,
            # 6.10454819e-02, 0.00000000e+00, 3.93588149e-02, 8.63476184e-01,
            # 0.00000000e+00, 3.03496535e+00, 0.00000000e+00, 0.00000000e+00,
            # 9.73273001e-01, 4.45166925e+00, 0.00000000e+00, 6.28318531e+00,
            # 4.18199124e-01, 0.00000000e+00, 6.28318531e+00, 3.89254084e+00,
            # 0.00000000e+00, 3.41294533e+00, 6.28318531e+00, 6.85635584e-01,
            # 6.28318531e+00, 1.91599661e-01, 0.00000000e+00, 5.13315942e+00,
            # 3.23950124e+00, 0.00000000e+00, 6.24846002e+00, 5.51869267e-01,
            # 0.00000000e+00, 4.39047598e+00, 6.17226688e+00, 0.00000000e+00,
            # 2.24101276e+00, 4.09078486e+00, 0.00000000e+00, 1.45808695e+00,
            # 6.28318531e+00, 0.00000000e+00, 5.73286274e+00, 3.65438501e+00,
            # 0.00000000e+00, 2.00480627e+00, 1.19845040e+00, 0.00000000e+00,
            # 5.84733839e-01, 0.00000000e+00, 0.00000000e+00, 6.64800033e+02])]

            # -----------------------------------------------------------------
            # 2to3 to 3to4
            # -----------------------------------------------------------------
            return [np.array([4.45790073e+00, 0.00000000e+00, 0.00000000e+00, 6.28318531e+00,
            6.28318531e+00, 1.00000000e+00, 6.28318531e+00, 0.00000000e+00,
            1.00000000e+00, 6.28318531e+00, 1.06261201e-03, 1.00000000e+00,
            1.21184275e+00, 3.04650528e-02, 1.00000000e+00, 1.53191082e+00,
            7.67158358e-03, 1.00000000e+00, 4.93912577e+00, 3.15687570e+00,
            8.07488200e-01, 4.41619562e+00, 3.20020032e+00, 8.39350329e-01,
            5.80287390e+00, 4.77334979e+00, 6.36009801e-01, 6.28318531e+00,
            6.13179363e+00, 2.25857870e-01, 6.28318531e+00, 1.20201892e-02,
            1.00000000e+00, 1.52473285e+00, 5.90990099e+00, 1.00000000e+00,
            1.39854655e+00, 6.28318531e+00, 1.00000000e+00, 5.08888853e+00,
            3.82461135e+00, 8.06514600e-01, 5.39064126e+00, 2.73711427e+00,
            9.54493004e-01, 1.83274540e+00, 6.28318531e+00, 3.95872860e-01,
            5.74246201e+00, 2.71302870e+00, 8.39169737e-01, 1.38888754e+00,
            6.28318531e+00, 5.79891909e-01, 5.54705693e+00, 2.24543959e+00,
            7.43702794e-01, 3.30979205e+00, 3.40087076e+00, 5.83558313e-01,
            2.25974243e+00, 6.11040207e+00, 3.14641169e-03, 5.68545904e+02])]
            # return [np.array([5.42062225e+00, 0.00000000e+00, 0.00000000e+00, 6.27970782e+00,
            # 6.25712437e+00, 0.00000000e+00, 1.35304087e+00, 6.28020693e+00,
            # 1.00000000e+00, 5.92744266e+00, 3.15351150e-01, 0.00000000e+00,
            # 1.30617661e+00, 4.47299755e-03, 1.00000000e+00, 1.36085694e+00,
            # 6.28318531e+00, 1.00000000e+00, 3.79596895e+00, 3.21854034e+00,
            # 0.00000000e+00, 3.88525365e+00, 3.17291569e+00, 0.00000000e+00,
            # 3.56820086e+00, 3.53608037e+00, 0.00000000e+00, 6.28318531e+00,
            # 7.41273760e-01, 0.00000000e+00, 1.09393969e+00, 1.84715483e+00,
            # 0.00000000e+00, 1.61006249e+00, 6.28318531e+00, 1.00000000e+00,
            # 5.07588313e+00, 3.14184774e+00, 1.00000000e+00, 6.13470766e+00,
            # 6.19677647e+00, 0.00000000e+00, 6.14202081e+00, 5.86893287e+00,
            # 0.00000000e+00, 2.92979269e+00, 6.28318531e+00, 6.30584983e-01,
            # 3.29047935e+00, 6.28318531e+00, 4.23755875e-01, 4.18872832e+00,
            # 0.00000000e+00, 0.00000000e+00, 6.21638242e+00, 3.87285508e+00,
            # 0.00000000e+00, 6.61610926e-01, 3.68314578e+00, 0.00000000e+00,
            # 1.99165839e+00, 2.69772465e+00, 0.00000000e+00, 6.37220901e+02])]
