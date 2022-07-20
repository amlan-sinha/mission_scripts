import sys, os
sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding/debug')
sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding')
sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/extra/mission_scripts/my_support')
sys.path.append('./')
sys.path.append('../')

import pydylan

from lineplot import LinePlot
import numpy as np
from multiprocessing import Pool
import time
import datetime
import pickle
import argparse
import platform

from initial_guess_generator.cr3bp_init_generator import CR3BPInitGenerator
from my_support.support import get_GTO_in_CR3BP_units, get_LLO_in_CR3BP_units, html_colors

class CR3BPEarthMissionSimulator:

    def __init__(self, seed, sample_num, process_num, quiet_snopt, number_of_segments, maximum_shooting_time,
                 sample_mode, start_bdry, end_bdry, thrust):
        self.seed = seed
        self.sample_num = sample_num
        self.process_num = process_num
        self.quiet_snopt = quiet_snopt
        self.number_of_segments = number_of_segments
        self.maximum_shooting_time = maximum_shooting_time
        self.sample_mode = sample_mode
        self.start_bdry = start_bdry
        self.end_bdry = end_bdry
        self.thrust = thrust

        self.init_generator = CR3BPInitGenerator(sample_mode=self.sample_mode)

    def run(self):

        pool = Pool(processes=self.process_num)

        earth_initial_guess = self.init_generator.get_earth_initial_guess(seed=self.seed, sample_num=self.sample_num,
                                                                          number_of_segments=self.number_of_segments,
                                                                          maximum_shooting_time=self.maximum_shooting_time)

        start = time.time()
        result_data = pool.starmap(self.simulate, [(earth_initial_guess[i],) for i in range(self.sample_num)])
        end = time.time()
        print("the program takes", end - start)

        # Statistics
        total_num = 0
        feasible_num = 0
        for result in result_data:
            if result["feasibility"]:
                feasible_num += 1
            total_num += 1
        print("Feasible solution is", feasible_num / total_num)

        # Save data
        if platform.platform() == 'macOS-12.4-arm64-arm-64bit': 
            parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = parent_path + "/result/solution/" + str(datetime.datetime.now().date())
            if not os.path.isdir(path):
                os.makedirs(path)
            file_name = "seed_" + str(self.seed) + "_sample_num_" + str(self.sample_num) + "_mode_" \
                        + str(self.sample_mode) + "_start_bdry_" + str(self.start_bdry) + "_end_bdry_" \
                        + str(self.end_bdry) + "_thrust_" + str(self.thrust)
            full_file_path_name = path + "/" + file_name + ".pkl"
            with open(full_file_path_name, 'wb') as f:
                pickle.dump(result_data, f)

            print("results are successfully saved!")
        
        else:            
            path = '/scratch/network/amlans/data/mission_simulator/result/' + str(datetime.datetime.now().date())
            if not os.path.isdir(path):
                os.makedirs(path)
            file_name = "seed_" + str(self.seed) + "_sample_num_" + str(self.sample_num) + "_mode_" \
                        + str(self.sample_mode) + "_start_bdry_" + str(self.start_bdry) + "_end_bdry_" \
                        + str(self.end_bdry) + "_thrust_" + str(self.thrust)
            full_file_path_name = path + "/" + file_name + ".pkl"
            
            with open(full_file_path_name, 'wb') as f:
                pickle.dump(result_data, f)

            print("results are successfully saved!")

    def simulate(self, earth_initial_guess):

        # Set up environment and thruster #############################################################################
        earth = pydylan.Body("Earth")
        moon = pydylan.Body("Moon")

        cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)
        libration_point_information = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L1)
        desired_orbit_energy = libration_point_information[1] + 100E-4

        halo = pydylan.periodic_orbit.Halo(cr3bp, pydylan.enum.LibrationPoint.L1, desired_orbit_energy, 8000.)
        assert halo.solve_for_orbit() == pydylan.enum.OrbitGenerationResult.Success

        thruster_parameters = pydylan.ThrustParameters(fuel_mass=700., dry_mass=300., Isp=1000., thrust=self.thrust)  # the
        # ratio of thrust/total mass is within 8e-4, 8e-3, thus here we fix the mass, and set thrust to [0.5, 1.5]

        gto_spiral = pydylan.phases.lowthrust_spiral(cr3bp,
                                                     get_GTO_in_CR3BP_units(),
                                                     thruster_parameters)
        gto_spiral.evaluate(time_of_flight=self.start_bdry)  # original: 6.48423370092

        # Earth mission  ###############################################################################################
        # snopt setting
        snopt_options = pydylan.SNOPT_options_structure()
        snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
        snopt_options.quiet_SNOPT = self.quiet_snopt
        # snopt_options.time_limit = 15.  # original
        # snopt_options.time_limit = 300.
        snopt_options.time_limit = 500.
        snopt_options.total_iteration_limit = 15000
        # snopt_options.save_all_SNOPT_evaluations = True

        mbh_options = pydylan.MBH_options_structure()

        # earth mission
        thruster_parameters.fuel_mass = gto_spiral.get_states()[-1, 6]

        phase_options = pydylan.phase_options_structure()
        phase_options.number_of_segments = self.number_of_segments  # previously, 10
        phase_options.maximum_initial_coast_time = 15.
        phase_options.maximum_final_coast_time = 15.
        phase_options.maximum_shooting_time = self.maximum_shooting_time  # previously, 15
        phase_options.control_coordinate_transcription = pydylan.enum.polar

        # the start states are the final of gto_spiral
        earth_mission_start = pydylan.FixedBoundaryCondition(gto_spiral.get_final_states())
        # left integrate 8 unit time from the some point of the halo manifold, use the arc as the end boundary condition
        earth_manifold_arc = halo.generate_manifold_arc(0.2 * halo.orbit_period, self.end_bdry,  # both two
                                                        # parameters can be changed, now only changes the left time
                                                        pydylan.enum.PerturbationDirection.StableLeft)
        earth_mission_end = pydylan.FixedBoundaryCondition(earth_manifold_arc.get_end_state())

        # p = LinePlot()
        # p.grid()
        # p.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        # p.set_ylabel(ylabel_in=r'Y (DU)', fontsize=16)
        # p.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)

        # p = self._generate_and_plot_manifold(halo=halo, plot=p)
        # p.plot(xdata=earth_manifold_arc.mani_states[:, 0], ydata=earth_manifold_arc.mani_states[:, 1],
        #        color=html_colors['strong orange'])
        # p.plot(xdata=gto_spiral.get_states()[:, 0], ydata=gto_spiral.get_states()[:, 1],
        #        color=html_colors['baby blue'])
        # p.show()

        # Major optimization for earth mission #################################################################

        earth_mission = pydylan.Mission(cr3bp,
                                        earth_mission_start,
                                        earth_mission_end,
                                        pydylan.enum.snopt)  # specify the mode of the mission,

        earth_mission.add_phase_options(phase_options)
        earth_mission.set_thruster_parameters(thruster_parameters)
        earth_mission.add_control_initial_guess(earth_initial_guess)
        earth_mission.optimize(snopt_options, mbh_options)

        # results
        print("\n")
        print("is the solution for this intialization feasible?", earth_mission.is_best_solution_feasible())
        print("\n")

        # assert earth_mission.is_best_solution_feasible()  # TODO: remove the assert

        self._output_control_to_screen(earth_mission.get_control_state())
        print("\n")
        print("--------------------------------------------------------------------------------------------------")

        results = earth_mission.evaluate_and_return_solution(earth_mission.get_control_state())
        feasibility = earth_mission.is_best_solution_feasible()

        problem_results = earth_mission.get_all_feasible_solutions()
        # print('snopt evaluation shape', problem_results[0].snopt_control_evaluations.shape)

        # Only save those feasible data
        if earth_mission.is_best_solution_feasible():
            result_data = {"results.time": results.time, "results.states": results.states,
                           "results.control": results.control, "results.initial_guess": earth_initial_guess,
                           "feasibility": feasibility,
                           "snopt_control_evaluations": problem_results[0].snopt_control_evaluations,
                           "maximum_constraint_bounds_violation": problem_results[0].maximum_constraint_bounds_violation,
                           "maximum_zero_violation": problem_results[0].maximum_zero_violation,
                           "snopt_inform": problem_results[0].snopt_inform}
        else:
            result_data = {"results.time": results.time, "results.states": results.states,
                           "results.control": results.control, "results.initial_guess": earth_initial_guess,
                           "feasibility": feasibility,
                           "snopt_control_evaluations": None,
                           "maximum_constraint_bounds_violation": None,
                           "maximum_zero_violation": None,
                           "snopt_inform": None}
        return result_data

    def _generate_and_plot_manifold(self,
                                    halo: pydylan.periodic_orbit.Halo,
                                    plot: LinePlot,
                                    coordinates=(0, 1),
                                    manifold_time=8.,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableLeft):
        orbit_times = np.linspace(0., halo.orbit_period, 50)

        for orbit_time in orbit_times:
            manifold_arc = halo.generate_manifold_arc(orbit_time, manifold_time, manifold_direction)
            plot.plot(xdata=manifold_arc.mani_states[:, coordinates[0]],
                      ydata=manifold_arc.mani_states[:, coordinates[1]],
                      color='black', alpha=0.5)

        return plot

    def _output_control_to_screen(self, control):
        print('\nThe control vector:')
        for entry in control:
            print('{},'.format(entry))

    def _plot_result(self, to_plot):

        # plotting
        p = LinePlot()
        p.grid()
        p.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        p.set_ylabel(ylabel_in=r'Y (DU)', fontsize=16)
        p.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)

        p = self._generate_and_plot_manifold(halo=self.halo, plot=p)
        p = self._generate_and_plot_manifold(halo=self.halo, plot=p, coordinates=(0, 1), manifold_time=4.,
                                             manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight)
        p.plot(xdata=self.earth_manifold_arc.mani_states[:, 0], ydata=self.earth_manifold_arc.mani_states[:, 1],
               color=html_colors['strong orange'])
        p.plot(xdata=self.moon_manifold_arc.mani_states[:, 0], ydata=self.moon_manifold_arc.mani_states[:, 1],
               color=html_colors['strong orange'])
        p.plot(xdata=self.gto_spiral.get_states()[:, 0], ydata=self.gto_spiral.get_states()[:, 1],
               color=html_colors['baby blue'])
        p.plot(xdata=self.earth_states[:, 0], ydata=self.earth_states[:, 1], color=html_colors['light green'])
        p.plot(xdata=self.moon_states[:, 0], ydata=self.moon_states[:, 1], color=html_colors['light green'])

        p.new_plot()
        p.grid()
        p.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        p.set_ylabel(ylabel_in=r'Z (DU)', fontsize=16)
        p.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)

        p = self._generate_and_plot_manifold(halo=self.halo, plot=p, coordinates=(0, 2), manifold_time=8.,
                                             manifold_direction=pydylan.enum.PerturbationDirection.StableLeft)
        p = self._generate_and_plot_manifold(halo=self.halo, plot=p, coordinates=(0, 2), manifold_time=4.,
                                             manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight)
        p.plot(xdata=self.earth_manifold_arc.mani_states[:, 0], ydata=self.earth_manifold_arc.mani_states[:, 2],
               color=html_colors['strong orange'])
        p.plot(xdata=self.moon_manifold_arc.mani_states[:, 0], ydata=self.moon_manifold_arc.mani_states[:, 2],
               color=html_colors['strong orange'])
        p.plot(xdata=self.gto_spiral.get_states()[:, 0], ydata=self.gto_spiral.get_states()[:, 2],
               color=html_colors['baby blue'])
        p.plot(xdata=self.earth_states[:, 0], ydata=self.earth_states[:, 2], color=html_colors['light green'])
        p.plot(xdata=self.moon_states[:, 0], ydata=self.moon_states[:, 2], color=html_colors['light green'])

        p.new_plot()
        p.grid()
        p.set_xlabel(xlabel_in=r'time (TU)', fontsize=16)
        p.set_ylabel(ylabel_in=r'wet mass (kg)', fontsize=16)
        p.set_title(title_in='Fuel mass versus time', fontsize=18)

        p.plot(xdata=self.gto_spiral.get_time(), ydata=self.gto_spiral.get_states()[:, 6],
               color=html_colors['baby blue'])
        self.earth_time += self.gto_spiral.get_time()[-1]
        p.plot(xdata=self.earth_time, ydata=self.earth_states[:, 6], color=html_colors['light green'])
        earth_manifold_time = -self.earth_manifold_arc.mani_time
        earth_manifold_time += self.earth_time[-1]
        earth_manifold_mass = self.earth_states[-1, 6] * np.ones_like(self.earth_manifold_arc.mani_time)
        p.plot(xdata=earth_manifold_time, ydata=earth_manifold_mass, color=html_colors['strong orange'])
        moon_manifold_time = self.moon_manifold_arc.mani_time
        moon_manifold_time += earth_manifold_time[-1]
        moon_manifold_mass = self.earth_states[-1, 6] * np.ones_like(self.moon_manifold_arc.mani_time)
        p.plot(xdata=moon_manifold_time, ydata=moon_manifold_mass, color=html_colors['strong orange'])
        self.moon_time += moon_manifold_time[-1]
        p.plot(xdata=self.moon_time, ydata=self.moon_states[:, 6], color=html_colors['light green'])

        if to_plot:
            p.show()


class CR3BPResonantMissionSimulator:

    def __init__(self, seed, sample_num, process_num, quiet_snopt, number_of_segments, maximum_shooting_time,
                 sample_mode, start_bdry, end_bdry, thrust):
        self.seed = seed
        self.sample_num = sample_num
        self.process_num = process_num
        self.quiet_snopt = quiet_snopt
        self.number_of_segments = number_of_segments
        self.maximum_shooting_time = maximum_shooting_time
        self.sample_mode = sample_mode
        self.start_bdry = start_bdry
        self.end_bdry = end_bdry
        self.thrust = thrust

        self.init_generator = CR3BPInitGenerator(sample_mode=self.sample_mode)

    def run(self):

        pool = Pool(processes=self.process_num)

        earth_initial_guess = self.init_generator.get_earth_initial_guess(seed=self.seed, sample_num=self.sample_num,
                                                                          number_of_segments=self.number_of_segments,
                                                                          maximum_shooting_time=self.maximum_shooting_time)

        start = time.time()
        result_data = pool.starmap(self.simulate, [(earth_initial_guess[i],) for i in range(self.sample_num)])
        end = time.time()
        print("the program takes", end - start)

        # Statistics
        total_num = 0
        feasible_num = 0
        for result in result_data:
            if result["feasibility"]:
                feasible_num += 1
            total_num += 1
        print("Feasible solution is", feasible_num / total_num)

        # Save data
        if platform.platform() == 'macOS-12.4-arm64-arm-64bit': 
            parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = parent_path + "/result/solution/" + str(datetime.datetime.now().date())
            if not os.path.isdir(path):
                os.makedirs(path)
            file_name = "seed_" + str(self.seed) + "_sample_num_" + str(self.sample_num) + "_mode_" \
                        + str(self.sample_mode) + "_start_bdry_" + str(self.start_bdry) + "_end_bdry_" \
                        + str(self.end_bdry) + "_thrust_" + str(self.thrust)
            full_file_path_name = path + "/" + file_name + ".pkl"
            with open(full_file_path_name, 'wb') as f:
                pickle.dump(result_data, f)

            print("results are successfully saved!")
        
        else:            
            path = '/scratch/network/amlans/data/mission_simulator/result/' + str(datetime.datetime.now().date())
            if not os.path.isdir(path):
                os.makedirs(path)
            file_name = "seed_" + str(self.seed) + "_sample_num_" + str(self.sample_num) + "_mode_" \
                        + str(self.sample_mode) + "_start_bdry_" + str(self.start_bdry) + "_end_bdry_" \
                        + str(self.end_bdry) + "_thrust_" + str(self.thrust)
            full_file_path_name = path + "/" + file_name + ".pkl"
            with open(full_file_path_name, 'wb') as f:
                pickle.dump(result_data, f)

            print("results are successfully saved!")

    def simulate(self, earth_initial_guess):

        # Set up environment and thruster
        earth = pydylan.Body("Earth")
        moon = pydylan.Body("Moon")

        cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)
        
        libration_point_information = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L1)
        desired_orbit_energy = libration_point_information[1] + 100E-4

        # lyapunov_l1 = pydylan.periodic_orbit.Lyapunov(cr3bp, pydylan.enum.LibrationPoint.L1, desired_orbit_energy)
        # assert lyapunov_l1.solve_for_orbit() == pydylan.enum.OrbitGenerationResult.Success

        # libration_point_information = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L2)
        # desired_orbit_energy = libration_point_information[1] + 100E-4

        # lyapunov_l2 = pydylan.periodic_orbit.Lyapunov(cr3bp, pydylan.enum.LibrationPoint.L2, desired_orbit_energy)
        # assert lyapunov_l2.solve_for_orbit() == pydylan.enum.OrbitGenerationResult.Success

        # # left integrate 8 unit time from the some point of the halo manifold, use the arc as the end boundary condition
        # earth_manifold_arc = lyapunov_l1.generate_manifold_arc(0.25 * lyapunov_l1.orbit_period, self.end_bdry, pydylan.enum.PerturbationDirection.UnstableRight)
        # right_bc = pydylan.FixedBoundaryCondition(earth_manifold_arc.get_end_state())

        # moon_manifold_arc = lyapunov_l2.generate_manifold_arc(0.25 * lyapunov_l2.orbit_period, self.start_bdry, pydylan.enum.PerturbationDirection.StableRight)
        # left_bc = pydylan.FixedBoundaryCondition(moon_manifold_arc.get_end_state())

        # print("Difference in energy: ", cr3bp.Jacobi_integral_from_state(earth_manifold_arc.get_end_state()) - cr3bp.Jacobi_integral_from_state(moon_manifold_arc.get_end_state()))

        r1_options = pydylan.periodic_orbit.ResonanceOptions()
        r1_options.p, r1_options.q = 3, 4
        r1_options.x, r1_options.z, r1_options.zdot = 0.5, 0., 0.
        r1_orbit = pydylan.periodic_orbit.Resonance(cr3bp, r1_options)
        result_1 = r1_orbit.solve_for_orbit()
        if result_1 != pydylan.enum.OrbitGenerationResult.Success:
            r1_options.x = 0.1
        while result_1 != pydylan.enum.OrbitGenerationResult.Success:
            r1_options.x += 0.1
            print("Resonant Orbit Creation Unsuccessful, trying with new x")
            r1_orbit = pydylan.periodic_orbit.Resonance(cr3bp, r1_options)
            result_1 = r1_orbit.solve_for_orbit()

        r1_left_manifold_arc = r1_orbit.generate_manifold_arc(0.25 * r1_orbit.orbit_period, self.end_bdry, pydylan.enum.PerturbationDirection.UnstableRight)
        left_bc = pydylan.FixedBoundaryCondition(r1_left_manifold_arc.get_end_state())

        r2_options = pydylan.periodic_orbit.ResonanceOptions()
        r2_options.p, r2_options.q = 2, 3
        r2_options.x, r2_options.z, r2_options.zdot = 0.5, 0., 0.
        r2_orbit = pydylan.periodic_orbit.Resonance(cr3bp, r2_options)
        result_2 = r2_orbit.solve_for_orbit()
        if result_2 != pydylan.enum.OrbitGenerationResult.Success:
            r2_options.x = 0.1
        while result_2 != pydylan.enum.OrbitGenerationResult.Success:
            r2_options.x += 0.1
            print("Resonant Orbit Creation Unsuccessful, trying with new x")
            r2_orbit = pydylan.periodic_orbit.Resonance(cr3bp, r2_options)
            result_2 = r2_orbit.solve_for_orbit()

        r2_right_manifold_arc = r2_orbit.generate_manifold_arc(0.25 * r2_orbit.orbit_period, self.start_bdry, pydylan.enum.PerturbationDirection.StableRight)
        right_bc = pydylan.FixedBoundaryCondition(r2_right_manifold_arc.get_end_state())

        thruster_parameters = pydylan.ThrustParameters(fuel_mass=700., dry_mass=300., Isp=1000., thrust=self.thrust)

        # Earth mission  ###############################################################################################
        # snopt setting
        snopt_options = pydylan.SNOPT_options_structure()
        snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
        snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible
        snopt_options.quiet_SNOPT = False
        snopt_options.time_limit = 1 * 60 * 60.

        mbh_options = pydylan.MBH_options_structure()
        mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
        mbh_options.max_step_size = 1.
        mbh_options.quiet_MBH = False
        mbh_options.time_limit = 2 * 60 * 60.

        phase_options = pydylan.phase_options_structure()
        phase_options.number_of_segments = 20
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

        # p = LinePlot()
        # p.grid()
        # p.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        # p.set_ylabel(ylabel_in=r'Y (DU)', fontsize=16)
        # p.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)

        # p = self._generate_and_plot_manifold(lyapunov=lyapunov_l2, plot=p, manifold_time=self.start_bdry,
        #                             manifold_direction=pydylan.enum.PerturbationDirection.StableRight)
        # p = self._generate_and_plot_manifold(lyapunov=lyapunov_l1, plot=p, manifold_time=self.end_bdry,
        #                             manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight)
        # p.plot(xdata=earth_manifold_arc.mani_states[:, 0], ydata=earth_manifold_arc.mani_states[:, 1],
        #        color=html_colors['strong orange'])
        # p.plot(xdata=moon_manifold_arc.mani_states[:, 0], ydata=moon_manifold_arc.mani_states[:, 1],
        #        color=html_colors['baby blue'])
        # p.show()

        p1 = LinePlot()
        p1.grid()
        p1.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        p1.set_ylabel(ylabel_in=r'Y (DU)', fontsize=16)
        p1.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)

        p1 = self._generate_and_plot_manifold(lyapunov=r1_orbit, plot=p1, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.UnstableLeft)
        p1 = self._generate_and_plot_manifold(lyapunov=r1_orbit, plot=p1, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight)
        p1 = self._generate_and_plot_manifold(lyapunov=r1_orbit, plot=p1, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableLeft)
        p1 = self._generate_and_plot_manifold(lyapunov=r1_orbit, plot=p1, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableRight)

        p2 = LinePlot()
        p2.grid()
        p2.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        p2.set_ylabel(ylabel_in=r'Y (DU)', fontsize=16)
        p2.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)

        p2 = self._generate_and_plot_manifold(lyapunov=r2_orbit, plot=p2, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.UnstableLeft)
        p2 = self._generate_and_plot_manifold(lyapunov=r2_orbit, plot=p2, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight)
        p2 = self._generate_and_plot_manifold(lyapunov=r2_orbit, plot=p2, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableLeft)
        p2 = self._generate_and_plot_manifold(lyapunov=r2_orbit, plot=p2, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableRight)

        p = LinePlot()
        p.grid()
        p.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        p.set_ylabel(ylabel_in=r'Y (DU)', fontsize=16)
        p.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)

        p.plot(xdata=r1_left_manifold_arc.mani_states[:, 0], ydata=r1_left_manifold_arc.mani_states[:, 1],
               color='g')
        p.plot(xdata=r2_right_manifold_arc.mani_states[:, 0], ydata=r2_right_manifold_arc.mani_states[:, 1],
               color='b')
        p.show()

        # Major optimization for earth mission #################################################################

        earth_mission = pydylan.Mission(cr3bp,
                                        right_bc,
                                        left_bc,
                                        pydylan.enum.mbh)  # specify the mode of the mission,

        earth_mission.add_phase_options(phase_options)
        earth_mission.set_thruster_parameters(thruster_parameters)
        earth_mission.add_control_initial_guess(earth_initial_guess)
        earth_mission.optimize(snopt_options, mbh_options)

        # results
        print("\n")
        print("is the solution for this intialization feasible?", earth_mission.is_best_solution_feasible())
        print("\n")

        # assert earth_mission.is_best_solution_feasible()  # TODO: remove the assert

        self._output_control_to_screen(earth_mission.get_control_state())
        print("\n")
        print("--------------------------------------------------------------------------------------------------")

        results = earth_mission.evaluate_and_return_solution(earth_mission.get_control_state())
        feasibility = earth_mission.is_best_solution_feasible()

        problem_results = earth_mission.get_all_feasible_control_solutions()
        # print('snopt evaluation shape', problem_results[0].snopt_control_evaluations.shape)

        # Only save those feasible data
        if earth_mission.is_best_solution_feasible():
            # result_data = {"results.time": results.time, "results.states": results.states,
            #                "results.control": results.control, "results.initial_guess": earth_initial_guess,
            #                "feasibility": feasibility,
            #                "snopt_control_evaluations": problem_results[0].snopt_control_evaluations,
            #                "maximum_constraint_bounds_violation": problem_results[0].maximum_constraint_bounds_violation,
            #                "maximum_zero_violation": problem_results[0].maximum_zero_violation,
            #                "snopt_inform": problem_results[0].snopt_inform}
            result_data = {"results.time": results.time, "results.states": results.states,
                           "results.control": results.control, "results.initial_guess": earth_initial_guess,
                           "feasibility": feasibility,
                           "snopt_control_evaluations": problem_results}
        else:
            result_data = {"results.time": results.time, "results.states": results.states,
                           "results.control": results.control, "results.initial_guess": earth_initial_guess,
                           "feasibility": feasibility,
                           "snopt_control_evaluations": None,
                           "maximum_constraint_bounds_violation": None,
                           "maximum_zero_violation": None,
                           "snopt_inform": None}
        return result_data

    def _generate_and_plot_manifold(self,
                                    lyapunov: pydylan.periodic_orbit.Lyapunov,
                                    plot: LinePlot,
                                    coordinates=(0, 1),
                                    manifold_time=8.,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableLeft):
        orbit_times = np.linspace(0., lyapunov.orbit_period, 50)

        StableDirections = [pydylan.enum.PerturbationDirection.StableLeft, pydylan.enum.PerturbationDirection.StableRight]
        thiscolor=html_colors['baby blue'] if manifold_direction in StableDirections else html_colors['strong orange']

        for orbit_time in orbit_times:
            manifold_arc = lyapunov.generate_manifold_arc(orbit_time, manifold_time, manifold_direction)
            plot.plot(xdata=manifold_arc.mani_states[:, coordinates[0]],
                      ydata=manifold_arc.mani_states[:, coordinates[1]],
                      color=thiscolor, alpha=0.5)

        return plot

    def _output_control_to_screen(self, control):
        print('\nThe control vector:')
        for entry in control:
            print('{},'.format(entry))

    def _plot_result(self, to_plot):

        # plotting
        p = LinePlot()
        p.grid()
        p.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        p.set_ylabel(ylabel_in=r'Y (DU)', fontsize=16)
        p.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)

        p = self._generate_and_plot_manifold(lyapunov=self.lyapunov, plot=p)
        p = self._generate_and_plot_manifold(lyapunov=self.lyapunov, plot=p, coordinates=(0, 1), manifold_time=4.,
                                             manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight)
        p.plot(xdata=self.earth_manifold_arc.mani_states[:, 0], ydata=self.earth_manifold_arc.mani_states[:, 1],
               color=html_colors['strong orange'])
        p.plot(xdata=self.moon_manifold_arc.mani_states[:, 0], ydata=self.moon_manifold_arc.mani_states[:, 1],
               color=html_colors['strong orange'])
        p.plot(xdata=self.gto_spiral.get_states()[:, 0], ydata=self.gto_spiral.get_states()[:, 1],
               color=html_colors['baby blue'])
        p.plot(xdata=self.earth_states[:, 0], ydata=self.earth_states[:, 1], color=html_colors['light green'])
        p.plot(xdata=self.moon_states[:, 0], ydata=self.moon_states[:, 1], color=html_colors['light green'])

        p.new_plot()
        p.grid()
        p.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        p.set_ylabel(ylabel_in=r'Z (DU)', fontsize=16)
        p.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)

        p = self._generate_and_plot_manifold(lyapunov=self.lyapunov, plot=p, coordinates=(0, 2), manifold_time=8.,
                                             manifold_direction=pydylan.enum.PerturbationDirection.StableLeft)
        p = self._generate_and_plot_manifold(lyapunov=self.lyapunov, plot=p, coordinates=(0, 2), manifold_time=4.,
                                             manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight)
        p.plot(xdata=self.earth_manifold_arc.mani_states[:, 0], ydata=self.earth_manifold_arc.mani_states[:, 2],
               color=html_colors['strong orange'])
        p.plot(xdata=self.moon_manifold_arc.mani_states[:, 0], ydata=self.moon_manifold_arc.mani_states[:, 2],
               color=html_colors['strong orange'])
        p.plot(xdata=self.gto_spiral.get_states()[:, 0], ydata=self.gto_spiral.get_states()[:, 2],
               color=html_colors['baby blue'])
        p.plot(xdata=self.earth_states[:, 0], ydata=self.earth_states[:, 2], color=html_colors['light green'])
        p.plot(xdata=self.moon_states[:, 0], ydata=self.moon_states[:, 2], color=html_colors['light green'])

        p.new_plot()
        p.grid()
        p.set_xlabel(xlabel_in=r'time (TU)', fontsize=16)
        p.set_ylabel(ylabel_in=r'wet mass (kg)', fontsize=16)
        p.set_title(title_in='Fuel mass versus time', fontsize=18)

        p.plot(xdata=self.gto_spiral.get_time(), ydata=self.gto_spiral.get_states()[:, 6],
               color=html_colors['baby blue'])
        self.earth_time += self.gto_spiral.get_time()[-1]
        p.plot(xdata=self.earth_time, ydata=self.earth_states[:, 6], color=html_colors['light green'])
        earth_manifold_time = -self.earth_manifold_arc.mani_time
        earth_manifold_time += self.earth_time[-1]
        earth_manifold_mass = self.earth_states[-1, 6] * np.ones_like(self.earth_manifold_arc.mani_time)
        p.plot(xdata=earth_manifold_time, ydata=earth_manifold_mass, color=html_colors['strong orange'])
        moon_manifold_time = self.moon_manifold_arc.mani_time
        moon_manifold_time += earth_manifold_time[-1]
        moon_manifold_mass = self.earth_states[-1, 6] * np.ones_like(self.moon_manifold_arc.mani_time)
        p.plot(xdata=moon_manifold_time, ydata=moon_manifold_mass, color=html_colors['strong orange'])
        self.moon_time += moon_manifold_time[-1]
        p.plot(xdata=self.moon_time, ydata=self.moon_states[:, 6], color=html_colors['light green'])

        if to_plot:
            p.show()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='CR3BP earth mission simulator')
    parser.add_argument('--seed_option', '-s',
                        help='choose the seed option for this run',
                        default='0')
    parser.add_argument('--file_num',
                        help='specify the number of file to save',
                        default='1')
    parser.add_argument('--sample_num_per_file',
                        help='specify the number of samples saved in single file',
                        default='1')
    parser.add_argument('--process_num', '-p',
                        help='specify the number of processes to use',
                        default='1')
    parser.add_argument('--start_bdry',
                        help='specify the start boundary condition, how many seconds for gto spiral',
                        default='6.48423370092')
    parser.add_argument('--end_bdry',
                        help='specify the end boundary condition, how many seconds to left integrate from the halo manifold',
                        default='8.')
    parser.add_argument('--thrust',
                        help='specify the thrust',
                        default='1.')
    args = parser.parse_args()

    # seed_option = int(args.seed_option)
    # file_num = int(args.file_num)
    # sample_num_per_file = int(args.sample_num_per_file)
    # process_num = int(args.process_num)
    # start_bdry = float(args.start_bdry)
    # end_bdry = float(args.end_bdry)
    # thrust = float(args.thrust)

    seed_option = 0
    file_num = 1
    sample_num_per_file = 1
    process_num = 1
    start_bdry = 9.0
    end_bdry = 4.5
    thrust = 1.0

    for i in range(file_num):
        seed = i + 100 * seed_option
        print("seed is ", seed)
        sample_num = sample_num_per_file
        simulator = CR3BPResonantMissionSimulator(seed=seed, sample_num=sample_num, process_num=process_num,
                                                    quiet_snopt=True,
                                                    number_of_segments=20,
                                                    maximum_shooting_time=10.,
                                                    sample_mode="uniform_control_time_mass",
                                                    start_bdry=start_bdry,
                                                    end_bdry=end_bdry,
                                                    thrust=thrust)
        simulator.run()

    # for i in range(5):
    #     seed = i
    #     sample_num = 200
    #     process_num = 10
    #     simulator = CR3BPEarthMissionSimulator(seed=seed, sample_num=sample_num, process_num=process_num,
    #                                            quiet_snopt=True,
    #                                            number_of_segments=20,
    #                                            maximum_shooting_time=10.,
    #                                            sample_mode="independent_truncated_norm_control_time_mass")
    #     simulator.run()
