import sys, os

sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding/debug')
sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding')
sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/extra/mission_scripts/my_support')
sys.path.append('./')
sys.path.append('../')

import pydylan

from lineplot import LinePlot
import numpy as np
import scipy as sc
import pickle
import datetime
import argparse
import platform
import time

from multiprocessing import Pool

from initial_guess_generator.cr3bp_init_generator import CR3BPInitGenerator
from my_support.support import get_GTO_in_CR3BP_units, get_LLO_in_CR3BP_units, html_colors


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

        self.result = None
        self.left_mani_arc = None
        self.right_mani_arc = None
        self.left_periodic_orbit = None
        self.right_periodic_orbit = None
        self.eom = None
        self.poincare_puncture_points = None

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
        self.eom = cr3bp
        
        r1_options = pydylan.periodic_orbit.ResonanceOptions()
        r1_options.p, r1_options.q = 2, 3
        r1_options.x, r1_options.z, r1_options.zdot = 0.5, 0., 0.
        r1_orbit = pydylan.periodic_orbit.Resonance(cr3bp, r1_options)
        result_1 = r1_orbit.solve_for_orbit()
        if result_1 != pydylan.enum.OrbitGenerationResult.Success:
            r1_options.x = 0.1
        while result_1 != pydylan.enum.OrbitGenerationResult.Success:
            print("Resonant Orbit Creation Unsuccessful, trying with new x")
            r1_options.x += 0.1
            r1_orbit = pydylan.periodic_orbit.Resonance(cr3bp, r1_options)
            result_1 = r1_orbit.solve_for_orbit()

        r1_left_manifold_arc = r1_orbit.generate_manifold_arc(0.25 * r1_orbit.orbit_period, self.end_bdry, pydylan.enum.PerturbationDirection.UnstableRight)
        left_bc = pydylan.FixedBoundaryCondition(r1_left_manifold_arc.get_end_state())

        r2_options = pydylan.periodic_orbit.ResonanceOptions()
        r2_options.p, r2_options.q = 3, 4
        r2_options.x, r2_options.z, r2_options.zdot = 0.5, 0., 0.
        r2_orbit = pydylan.periodic_orbit.Resonance(cr3bp, r2_options)
        result_2 = r2_orbit.solve_for_orbit()
        if result_2 != pydylan.enum.OrbitGenerationResult.Success:
            r2_options.x = 0.1
        while result_2 != pydylan.enum.OrbitGenerationResult.Success:
            print("Resonant Orbit Creation Unsuccessful, trying with new x")
            r2_options.x += 0.1
            r2_orbit = pydylan.periodic_orbit.Resonance(cr3bp, r2_options)
            result_2 = r2_orbit.solve_for_orbit()

        r2_right_manifold_arc = r2_orbit.generate_manifold_arc(0.25 * r2_orbit.orbit_period, self.start_bdry, pydylan.enum.PerturbationDirection.UnstableRight)
        right_bc = pydylan.FixedBoundaryCondition(r2_right_manifold_arc.get_end_state())

        thruster_parameters = pydylan.ThrustParameters(fuel_mass=700., dry_mass=300., Isp=1000., thrust=self.thrust)

        snopt_options = pydylan.SNOPT_options_structure()
        snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
        snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible
        snopt_options.quiet_SNOPT = False
        snopt_options.time_limit = 1 # * 60 * 60.

        mbh_options = pydylan.MBH_options_structure()
        mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
        mbh_options.max_step_size = 1.
        mbh_options.quiet_MBH = False
        mbh_options.time_limit = 2 # * 60 * 60.

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

        earth_mission = pydylan.Mission(cr3bp,
                                        right_bc,
                                        left_bc,
                                        pydylan.enum.mbh)

        earth_mission.add_phase_options(phase_options)
        earth_mission.set_thruster_parameters(thruster_parameters)
        earth_mission.add_control_initial_guess(earth_initial_guess)
        earth_mission.optimize(snopt_options, mbh_options)

        print("--------------------------------------------------------------------------------------------------")

        print("\n")
        print("Is the solution for this intialization feasible?", earth_mission.is_best_solution_feasible())
        print("\n")

        # assert earth_mission.is_best_solution_feasible()

        print("--------------------------------------------------------------------------------------------------")

        print("\n")
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

        self.results = result_data
        self.left_mani_arc = r1_left_manifold_arc
        self.left_periodic_orbit = r1_orbit
        self.right_mani_arc = r2_right_manifold_arc
        self.right_periodic_orbit = r2_orbit

        self._plot_result(to_plot=False)

        l1_punctures_increasing, l2_punctures_increasing, l1_punctures_decreasing, l2_punctures_decreasing = [], [], [], []
        for i in range(5):
            l1p, l2p = self._propagate_to_poincare_section(poincare_section="y", crossing_direction="increasing", num_occ=i + 1,to_plot=False)
            l1_punctures_increasing.append(l1p)
            l2_punctures_increasing.append(l2p)
            l1p, l2p = self._propagate_to_poincare_section(poincare_section="y", crossing_direction="decreasing", num_occ=i + 1,to_plot=False)
            l1_punctures_decreasing.append(l1p)
            l2_punctures_decreasing.append(l2p)

        result_data["l1_punc_inc"] = l1_punctures_increasing
        result_data["l1_punc_dec"] = l1_punctures_decreasing
        result_data["l2_punc_inc"] = l2_punctures_increasing
        result_data["l2_punc_dec"] = l2_punctures_decreasing

        self.poincare_puncture_points = {"l1_punc_inc": l1_punctures_increasing, "l1_punc_dec": l1_punctures_decreasing, "l2_punc_inc": l2_punctures_increasing, "l2_punc_inc": l2_punctures_decreasing}
        
        return result_data


    def _propagate_to_poincare_section(self, poincare_section="x", crossing_direction="increasing", num_occ=1, to_plot=False):

        num_arcs = 100

        vs_xdot = False

        cr3bp = self.eom
        l1, l2 = self.left_periodic_orbit, self.right_periodic_orbit
        l1_orbit_times, l2_orbit_times = np.linspace(0., l1.orbit_period, num_arcs), np.linspace(0., l2.orbit_period, num_arcs)

        l1_punctures, l2_punctures = np.zeros((num_arcs, 6)), np.zeros((num_arcs, 6))

        if crossing_direction=="increasing":
            cd = pydylan.enum.CrossingDirection.Increasing
        elif crossing_direction=="decreasing":
            cd = pydylan.enum.CrossingDirection.Decreasing
        else:
            cd = pydylan.enum.CrossingDirection.Bidirectional
        
        if poincare_section=="x":
            events = pydylan.integrators.TargetStateEvent(0, 1 - cr3bp.mu, 1E-6, cd, num_occ)
        else:
            events = pydylan.integrators.TargetStateEvent(1, 0., 1E-6, cd, num_occ)

        rk54 = pydylan.integrators.RK54()
        rk54.set_eom(cr3bp)
        rk54.add_event(events)
        rk54.set_time(0., 20.)

        for i, (l1_orbit_time, l2_orbit_time) in enumerate(zip(l1_orbit_times, l2_orbit_times)):

            l1_mani_arc = l1.generate_manifold_arc(l1_orbit_time, 0.1, pydylan.enum.PerturbationDirection.UnstableRight)
            rk54.evaluate(l1_mani_arc.get_end_state())
            l1_punctures[i] = rk54.get_states()[-1, :]

            l2_mani_arc = l2.generate_manifold_arc(l2_orbit_time, 0.1, pydylan.enum.PerturbationDirection.StableRight)
            rk54.evaluate(l2_mani_arc.get_end_state())
            l2_punctures[i] = rk54.get_states()[-1, :]

        if platform.platform() == 'macOS-12.4-arm64-arm-64bit': 
            parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = parent_path + "/result/solution/" + str(datetime.datetime.now().date()) + "/figures"
            if not os.path.isdir(path):
                os.makedirs(path)
            if vs_xdot:
                file_name = "section_" + poincare_section + "_cross_" + crossing_direction + "_number_" + str(num_occ) + "_xdot"
            else:
                file_name = "section_" + poincare_section + "_cross_" + crossing_direction + "_number_" + str(num_occ) + "_ydot"
            full_file_path_name = path + "/" + file_name + ".png"
        else:            
            path = '/scratch/network/amlans/data/mission_simulator/result/' + str(datetime.datetime.now().date())
            if not os.path.isdir(path):
                os.makedirs(path)
            if vs_xdot:
                file_name = "section_" + poincare_section + "_cross_" + crossing_direction + "_number_" + str(num_occ) + "_xdot"
            else:
                file_name = "section_" + poincare_section + "_cross_" + crossing_direction + "_number_" + str(num_occ) + "_ydot"
            full_file_path_name = path + "/" + file_name + ".png"

            print("results are successfully saved!")

        puncture_plot = LinePlot()
        puncture_plot.grid()
        puncture_plot.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        puncture_plot.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)
        if poincare_section=="x":
            if vs_xdot:
                puncture_plot.set_ylabel(ylabel_in=r'$\dot{X}$ (DU)', fontsize=16)
                puncture_plot.plot(xdata=l1_punctures[:, 1], ydata=l1_punctures[:, 3], color='r', marker='o', markersize=5, linestyle='')
                puncture_plot.plot(xdata=l2_punctures[:, 1], ydata=l2_punctures[:, 3], color='g', marker='o', markersize=5, linestyle='')
            else:
                puncture_plot.set_ylabel(ylabel_in=r'$\dot{Y}$ (DU)', fontsize=16)
                puncture_plot.plot(xdata=l1_punctures[:, 1], ydata=l1_punctures[:, 4], color='r', marker='o', markersize=5, linestyle='')
                puncture_plot.plot(xdata=l2_punctures[:, 1], ydata=l2_punctures[:, 4], color='g', marker='o', markersize=5, linestyle='')
                puncture_plot.save_figure(full_file_path_name, dpi=100)
        else:
            if vs_xdot:
                puncture_plot.set_ylabel(ylabel_in=r'$\dot{X}$ (DU)', fontsize=16)
                puncture_plot.plot(xdata=l1_punctures[:, 0], ydata=l1_punctures[:, 3], color='r', marker='o', markersize=5, linestyle='')
                puncture_plot.plot(xdata=l2_punctures[:, 0], ydata=l2_punctures[:, 3], color='g', marker='o', markersize=5, linestyle='')
            else:
                puncture_plot.set_ylabel(ylabel_in=r'$\dot{Y}$ (DU)', fontsize=16)
                puncture_plot.plot(xdata=l1_punctures[:, 0], ydata=l1_punctures[:, 4], color='r', marker='o', markersize=5, linestyle='')
                puncture_plot.plot(xdata=l2_punctures[:, 0], ydata=l2_punctures[:, 4], color='g', marker='o', markersize=5, linestyle='')
                puncture_plot.save_figure(full_file_path_name, dpi=100)

        print("\n Successfully generated puncture points \n", flush=True)

        if to_plot:
            puncture_plot.show()

        return l1_punctures, l2_punctures

    def _generate_and_plot_manifold(self,
                                    periodic_orbit: pydylan.periodic_orbit,
                                    plot: LinePlot,
                                    coordinates=(0, 1),
                                    manifold_time=8.,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableLeft,
                                    color=None):
        orbit_times = np.linspace(0., periodic_orbit.orbit_period, 50)

        StableDirections = [pydylan.enum.PerturbationDirection.StableLeft, pydylan.enum.PerturbationDirection.StableRight]
        if color is not None: 
            thiscolor = color 
        else:
            thiscolor = html_colors['baby blue'] if manifold_direction in StableDirections else html_colors['strong orange']

        for orbit_time in orbit_times:
            manifold_arc = periodic_orbit.generate_manifold_arc(orbit_time, manifold_time, manifold_direction)
            plot.plot(xdata=manifold_arc.mani_states[:, coordinates[0]],
                      ydata=manifold_arc.mani_states[:, coordinates[1]],
                      color=thiscolor, alpha=0.5)

        return plot

    def _output_control_to_screen(self, control):
        print('Control Solution:')
        for entry in control:
            print('{},'.format(entry))

    def _plot_result(self, to_plot=False):

        result_data = self.results
        posx = result_data.get("results.states")[:, 0]
        posy = result_data.get("results.states")[:, 1]
        posz = result_data.get("results.states")[:, 2]

        # plotting
        p1 = LinePlot()
        p1.grid()
        p1.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        p1.set_ylabel(ylabel_in=r'Y (DU)', fontsize=16)
        p1.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)

        p1 = self._generate_and_plot_manifold(periodic_orbit=self.left_periodic_orbit, plot=p1, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.UnstableLeft, color='r')
        p1 = self._generate_and_plot_manifold(periodic_orbit=self.left_periodic_orbit, plot=p1, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight, color='r')
        p1 = self._generate_and_plot_manifold(periodic_orbit=self.left_periodic_orbit, plot=p1, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableLeft, color='r')
        p1 = self._generate_and_plot_manifold(periodic_orbit=self.left_periodic_orbit, plot=p1, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableRight, color='r')

        p2 = LinePlot()
        p2.grid()
        p2.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        p2.set_ylabel(ylabel_in=r'Y (DU)', fontsize=16)
        p2.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)

        p2 = self._generate_and_plot_manifold(periodic_orbit=self.right_periodic_orbit, plot=p2, manifold_time=self.end_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.UnstableLeft, color='g')
        p2 = self._generate_and_plot_manifold(periodic_orbit=self.right_periodic_orbit, plot=p2, manifold_time=self.end_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight, color='g')
        p2 = self._generate_and_plot_manifold(periodic_orbit=self.right_periodic_orbit, plot=p2, manifold_time=self.end_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableLeft, color='g')
        p2 = self._generate_and_plot_manifold(periodic_orbit=self.right_periodic_orbit, plot=p2, manifold_time=self.end_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableRight, color='g')

        p = LinePlot()
        p.grid()
        p.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        p.set_ylabel(ylabel_in=r'Y (DU)', fontsize=16)
        p.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)

        p = self._generate_and_plot_manifold(periodic_orbit=self.left_periodic_orbit, plot=p, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.UnstableLeft, color='r')
        p = self._generate_and_plot_manifold(periodic_orbit=self.left_periodic_orbit, plot=p, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight, color='r')
        p = self._generate_and_plot_manifold(periodic_orbit=self.left_periodic_orbit, plot=p, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableLeft, color='r')
        p = self._generate_and_plot_manifold(periodic_orbit=self.left_periodic_orbit, plot=p, manifold_time=self.start_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableRight, color='r')

        p = self._generate_and_plot_manifold(periodic_orbit=self.right_periodic_orbit, plot=p, manifold_time=self.end_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.UnstableLeft, color='g')
        p = self._generate_and_plot_manifold(periodic_orbit=self.right_periodic_orbit, plot=p, manifold_time=self.end_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight, color='g')
        p = self._generate_and_plot_manifold(periodic_orbit=self.right_periodic_orbit, plot=p, manifold_time=self.end_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableLeft, color='g')
        p = self._generate_and_plot_manifold(periodic_orbit=self.right_periodic_orbit, plot=p, manifold_time=self.end_bdry,
                                    manifold_direction=pydylan.enum.PerturbationDirection.StableRight, color='g')

        p.plot(xdata=np.array([self.left_mani_arc.mani_states[0, 0], ]), ydata=np.array([self.left_mani_arc.mani_states[0, 1], ]), color='r', marker='o', markersize=5, linewidth=3)
        p.plot(xdata=np.array([self.left_mani_arc.mani_states[-1, 0], ]), ydata=np.array([self.left_mani_arc.mani_states[-1, 1], ]), color='r', marker='x', markersize=5, linewidth=3)
        p.plot(xdata=self.left_mani_arc.mani_states[:, 0], ydata=self.left_mani_arc.mani_states[:, 1],
               color='r', linewidth = 3)
               
        p.plot(xdata=np.array([self.right_mani_arc.mani_states[0, 0], ]), ydata=np.array([self.right_mani_arc.mani_states[0, 1], ]), color='g', marker='o', markersize=5, linewidth=3)
        p.plot(xdata=np.array([self.right_mani_arc.mani_states[-1, 0], ]), ydata=np.array([self.right_mani_arc.mani_states[-1, 1], ]), color='g', marker='x', markersize=5, linewidth=3)
        p.plot(xdata=self.right_mani_arc.mani_states[:, 0], ydata=self.right_mani_arc.mani_states[:, 1],
               color='g', linewidth = 3)

        p.plot(xdata=posx, ydata=posy, color='k', linewidth = 3)

        q = LinePlot()
        q.grid()
        q.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
        q.set_ylabel(ylabel_in=r'Y (DU)', fontsize=16)
        q.set_title(title_in='CR3BP Canonical Reference Frame', fontsize=18)

        q.plot(xdata=posx, ydata=posz, color='k', linewidth = 3)

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
    start_bdry = 1
    end_bdry = 0.25
    thrust = 1.0

    for i in range(file_num):
        seed = i + 100 * seed_option
        print("seed is ", seed)
        sample_num = sample_num_per_file
        simulator = CR3BPResonantMissionSimulator(seed=seed, sample_num=sample_num, process_num=process_num,
                                                    quiet_snopt=True,
                                                    number_of_segments=20,
                                                    maximum_shooting_time=10.,
                                                    sample_mode="resonant_feasible_solution",
                                                    start_bdry=start_bdry,
                                                    end_bdry=end_bdry,
                                                    thrust=thrust)
        simulator.run()
