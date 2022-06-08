import sys, os
from tracemalloc import stop
from unittest import result

sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding/debug')
sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding')

import pydylan
import numpy as np

from lineplot import LinePlot
import matplotlib.pyplot as plt
from support import html_colors

strongorange, babyblue = html_colors['strong orange'], html_colors['baby blue']

days_to_seconds, seconds_to_days = 86400., 1/86400.

def initial_guess():
    return np.array([3.99102629e+00, 5.25680860e-01, 4.39154442e-02, 6.07369155e+00,
       0.00000000e+00, 1.00000000e+00, 6.05882391e+00, 0.00000000e+00,
       1.00000000e+00, 1.16456040e+00, 1.37501964e+00, 0.00000000e+00,
       6.22689172e+00, 0.00000000e+00, 1.00000000e+00, 5.99700122e+00,
       0.00000000e+00, 1.00000000e+00, 5.76251607e+00, 0.00000000e+00,
       1.00000000e+00, 5.59234470e+00, 0.00000000e+00, 1.00000000e+00,
       2.36232860e+00, 0.00000000e+00, 0.00000000e+00, 5.33641753e+00,
       0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 1.39648704e+00,
       0.00000000e+00, 5.47180565e+00, 0.00000000e+00, 1.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 1.00000000e+00, 4.07685466e-01, 3.13925639e+00,
       9.29233772e-01, 6.25455729e+00, 6.28318531e+00, 1.00000000e+00,
       6.28318531e+00, 6.28318531e+00, 1.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       1.00000000e+00, 1.04288765e+00, 0.00000000e+00, 1.00000000e+00,
       1.96595730e+00, 0.00000000e+00, 4.41212893e-02, 2.47944756e+00,
       0.00000000e+00, 9.98659301e-01, 1.41271614e+00, 0.00000000e+00,
       9.51667597e-01, 3.07627409e-02, 0.00000000e+00, 9.63677430e-01,
       2.72912016e-02, 0.00000000e+00, 1.00000000e+00, 1.47833531e-02,
       0.00000000e+00, 1.00000000e+00, 5.58312815e-02, 0.00000000e+00,
       1.00000000e+00, 6.03577260e-02, 0.00000000e+00, 1.00000000e+00,
       6.30270671e-02, 0.00000000e+00, 1.00000000e+00, 4.42784616e+00,
       6.28318531e+00, 0.00000000e+00, 6.00957270e-01, 1.23607474e-03,
       1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.25966462e-02,
       0.00000000e+00, 1.01628009e+00, 0.00000000e+00, 0.00000000e+00,
       6.61589027e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 1.27289993e+00, 0.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 1.47895850e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.64700184e+00,
       1.47337169e+00, 0.00000000e+00, 0.00000000e+00, 2.62334436e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
       6.28318531e+00, 0.00000000e+00, 0.00000000e+00, 6.28318531e+00,
       1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       2.25500613e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 2.45077453e+00])


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


def get_jacobi_energy_from_state(eom:pydylan.eom, state: np.ndarray) -> float:

    return -eom.Jacobi_integral_from_state(state) / 2


def generate_lyapunov_orbit_and_propagate_to_poincare_section(cr3bp:pydylan.eom, desired_orbit_energy:float):

    num_arcs = 25

    l1 = generate_lyapunov_orbit(cr3bp, pydylan.enum.LibrationPoint.L1, 100E-4, desired_orbit_energy)
    l2 = generate_lyapunov_orbit(cr3bp, pydylan.enum.LibrationPoint.L2, 100E-4, desired_orbit_energy)

    l1_orbit_times, l2_orbit_times = np.linspace(0., l1.orbit_period, num_arcs), np.linspace(0., l2.orbit_period, num_arcs)

    l1_unstable_right_punctures, l2_stable_left_punctures = np.zeros((num_arcs, 6)), np.zeros((num_arcs, 6))

    # print("################################# I AM HERE (1) #################################", flush=True)
    
    events = pydylan.integrators.TargetStateEvent(0, 1. - cr3bp.mu, 1E-6, pydylan.enum.CrossingDirection.Increasing)

    rk54_l1 = pydylan.integrators.RK54()
    rk54_l1.set_eom(cr3bp)
    rk54_l1.add_event(events)
    rk54_l1.set_time(0., 20.)

    for i, l1_orbit_time in enumerate(l1_orbit_times):
        l1_unstable_right = l1.generate_manifold_arc(l1_orbit_time, 0.5, pydylan.enum.PerturbationDirection.UnstableRight)
        rk54_l1.evaluate(l1_unstable_right.get_end_state())
        l1_unstable_right_punctures[i] = rk54_l1.get_states()[-1, :]

        # print("L1 INTERSECTIONS: ", l1_unstable_right_punctures[i], flush=True)

    # print("################################# I AM HERE (2) #################################", flush=True)
    
    events = pydylan.integrators.TargetStateEvent(0, 1. - cr3bp.mu, 1E-6, pydylan.enum.CrossingDirection.Decreasing)

    rk54_l2 = pydylan.integrators.RK54()
    rk54_l2.set_eom(cr3bp)
    rk54_l2.add_event(events)
    rk54_l2.set_time(0., -20.)

    for i, l2_orbit_time in enumerate(l2_orbit_times):
        l2_stable_left = l2.generate_manifold_arc(l2_orbit_time, 0.5, pydylan.enum.PerturbationDirection.StableLeft)
        rk54_l2.evaluate(l2_stable_left.get_end_state())
        l2_stable_left_punctures[i] = rk54_l2.get_states()[-1, :]

        # print("L2 INTERSECTIONS: ", l2_stable_left_punctures[i], flush=True)

    return l1_unstable_right_punctures, l2_stable_left_punctures
        

def propagate_state_to_poincare_section(cr3bp:pydylan.eom, states: np.ndarray, num_samples = 5):

    events = pydylan.integrators.TargetStateEvent(0, 1. - cr3bp.mu, 1E-6, pydylan.enum.CrossingDirection.Increasing)

    rk54_left = pydylan.integrators.RK54()
    rk54_left.set_eom(cr3bp)
    rk54_left.add_event(events)

    coarse_samples = states[np.sort(np.random.randint(0, states.shape[0], num_samples)), :6]

    p = LinePlot()
    p.grid()
    p.set_xlabel(xlabel_in=r'X (DU)', fontsize=16)
    p.set_ylabel(ylabel_in=r'Y (DU)', fontsize=16)

    states_left, states_right = np.zeros_like(coarse_samples), np.zeros_like(coarse_samples)
    rk54_left.set_time(0., -20.)
    for i, si in enumerate(coarse_samples):
        rk54_left.evaluate(si)
        states_left[i] = rk54_left.get_states()[-1, :]

        energy = get_jacobi_energy_from_state(cr3bp, si)

        print("ENERGY LEFT: ", energy, flush=True)

    rk54_right = pydylan.integrators.RK54()
    rk54_right.set_eom(cr3bp)
    rk54_right.add_event(events)

    rk54_right.set_time(0., 20.)
    for i, si in enumerate(coarse_samples):
        rk54_right.evaluate(si)
        states_right[i] = rk54_right.get_states()[-1, :]

        energy = get_jacobi_energy_from_state(cr3bp, si)

        print("ENERGY RIGHT: ", energy, flush=True)

    for i, si in enumerate(coarse_samples):

        energy = get_jacobi_energy_from_state(cr3bp, si)

        print("ENERGY: ", energy, flush=True)

        l1_intersection, l2_intersection = generate_lyapunov_orbit_and_propagate_to_poincare_section(cr3bp, energy)

        p.new_plot()
        p.plot(xdata=np.array([states_left[i, 1], ]), ydata=np.array([states_left[i, 3], ]), color='black', marker='o', markersize=10)
        p.plot(xdata=np.array([states_right[i, 1], ]), ydata=np.array([states_right[i, 3], ]), color='black', marker='x', markersize=10)
        p.plot(xdata=l1_intersection[:, 1], ydata=l1_intersection[:, 3], color='red')
        p.plot(xdata=l2_intersection[:, 1], ydata=l2_intersection[:, 3], color='blue')
        p.save_figure("./Plots_For_AIAA_AAS_2022/state_%d.png"%i, dpi=100)

    plt.show()

    pass


def generate_and_plot_manifold(plot: LinePlot, orbit: pydylan.periodic_orbit, manifold_time=10., manifold_direction=pydylan.enum.PerturbationDirection.StableLeft, coordinates=(0, 1)):

    orbit_times = np.linspace(0., orbit.orbit_period, 50)

    StableDirections = [pydylan.enum.PerturbationDirection.StableLeft, pydylan.enum.PerturbationDirection.StableRight]
    defcolor=html_colors['baby blue'] if manifold_direction in StableDirections else html_colors['strong orange']

    for orbit_time in orbit_times:
        manifold_arc = orbit.generate_manifold_arc(orbit_time, manifold_time, manifold_direction)
        plot.plot(xdata=manifold_arc.mani_states[:, coordinates[0]], ydata=manifold_arc.mani_states[:, coordinates[1]], color=defcolor, alpha=0.2)

    return plot


def periodic_orbit_continuation(periodic_orbit: pydylan.periodic_orbit, desired_value: float) -> pydylan.periodic_orbit:

    continuation_settings = pydylan.periodic_orbit.ContinuationParameterInfo()
    continuation_settings.parameter_t = pydylan.enum.ParameterType.energy
    continuation_settings.desired_value = desired_value
    continuation_settings.step_size = 1e-4
    continuation_settings.min_step_size = 1e-10
    periodic_orbit.set_continuation_information(continuation_settings, iLimit=1000)
    result = periodic_orbit.solve_for_orbit()

    return periodic_orbit


def generate_lyapunov_orbit(cr3bp: pydylan.eom, libration_point: pydylan.enum.LibrationPoint, energy_offset: float, desired_orbit_energy=None) -> pydylan.periodic_orbit:

    libration_point_information = cr3bp.find_equilibrium_point(libration_point)

    if desired_orbit_energy is None: 
        desired_orbit_energy = libration_point_information[1] + energy_offset        

    lyapunov = pydylan.periodic_orbit.Lyapunov(cr3bp, libration_point, desired_orbit_energy)
    lyapunov = periodic_orbit_continuation(lyapunov, desired_orbit_energy)

    return lyapunov


def solve_for_second_phase(eom: pydylan.eom, thruster_parameters: pydylan.ThrustParameters, L1L: pydylan.periodic_orbit.Lyapunov, L2L: pydylan.periodic_orbit.Lyapunov) -> (np.ndarray, np.ndarray):

    L1_manifold_arc = L1L.generate_manifold_arc(L1L.orbit_period, 3.5, pydylan.enum.PerturbationDirection.UnstableRight)    
    left_boundary_condition = pydylan.FixedBoundaryCondition(L1_manifold_arc.get_end_state())

    L2_manifold_arc = L2L.generate_manifold_arc(L2L.orbit_period, 4., pydylan.enum.PerturbationDirection.StableLeft)
    right_boundary_condition = pydylan.FixedBoundaryCondition(L2_manifold_arc.get_end_state())

    print("INITIAL STATE: ", L1_manifold_arc.get_end_state(), "\n FINAL STATE: ", L2_manifold_arc.get_end_state())

    snopt_options = pydylan.SNOPT_options_structure()
    snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
    snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible
    snopt_options.quiet_SNOPT = False
    snopt_options.time_limit = 1 #60. * 30

    mbh_options = pydylan.MBH_options_structure()
    mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
    mbh_options.max_step_size = 1
    mbh_options.quiet_MBH = False
    mbh_options.time_limit = 1 #60. * 60

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
    mission.add_control_initial_guess(initial_guess())
    mission.optimize(snopt_options, mbh_options)

    output_control_to_screen(mission.get_control_state())

    assert mission.is_best_solution_feasible()
    # np.save("./Plots_For_AIAA_AAS_2022/CLtour_p2_init.npy", mission.get_control_state())
    # np.save("./Plots_For_AIAA_AAS_2022/CLtour_p2_feasible_control_solutions.npy", mission.get_all_feasible_control_solutions())

    results = mission.evaluate_and_return_solution(mission.get_control_state(), forward_backward_shooting=True)

    times = results.time
    states = results.states

    return times, states, L1_manifold_arc, L2_manifold_arc


def solve():

    earth = pydylan.Body("Earth")
    moon = pydylan.Body("Moon")
    cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)

    lyapunov_L1 = generate_lyapunov_orbit(cr3bp, pydylan.enum.LibrationPoint.L1, 100E-4)
    lyapunov_L2 = generate_lyapunov_orbit(cr3bp, pydylan.enum.LibrationPoint.L2, 100E-4 - 0.008092373007264175)

    thruster_parameters = pydylan.ThrustParameters(fuel_mass=2.5, dry_mass=12.5, Isp=2500., thrust=1.5E-3)

    time, states, L1_unstable_mani_arc, L2_stable_mani_arc = solve_for_second_phase(cr3bp, thruster_parameters, lyapunov_L1, lyapunov_L2)
    time = convert_to_days(cr3bp, time)

    propagate_state_to_poincare_section(cr3bp, states)

    # Pack the result into a dictionary
    result_dict = {
            'times': time, 
            'states': states,
            'L1_Lyapunov': lyapunov_L1,
            'L2_Lyapunov': lyapunov_L2,
            'L1_unstable_mani_arc': L1_unstable_mani_arc,
            'L2_stable_mani_arc': L2_stable_mani_arc}

    return result_dict


def plot_phase(result_dict, pathname):

    # Unpack the result into a dictionary

    time, states = result_dict.get('times'), result_dict.get('states')
    lyapunov_L1, lyapunov_L2 = result_dict.get('L1_Lyapunov'), result_dict.get('L2_Lyapunov')
    L1_unstable_mani_arc, L2_stable_mani_arc = result_dict.get('L1_unstable_mani_arc'), result_dict.get('L2_stable_mani_arc')

    p = LinePlot()
    p.grid()
    p.set_xlabel(xlabel_in=r'X (DU)', fontsize=15)
    p.set_ylabel(ylabel_in=r'Y (DU)', fontsize=15)
    p = generate_and_plot_manifold(plot=p, orbit=lyapunov_L1, manifold_time=5., manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight, coordinates=(0, 1))
    p = generate_and_plot_manifold(plot=p, orbit=lyapunov_L2, manifold_time=7., manifold_direction=pydylan.enum.PerturbationDirection.StableLeft, coordinates=(0, 1))
    p.plot(xdata=states[:, 0], ydata=states[:, 1], color='black', linewidth=1)
    p.plot(xdata=states[:, 0], ydata=states[:, 1], color='black', linewidth=1)
    p.plot(xdata=L1_unstable_mani_arc.mani_states[:, 0], ydata=L1_unstable_mani_arc.mani_states[:, 1], color='black', linewidth=1)
    p.plot(xdata=L2_stable_mani_arc.mani_states[:, 0], ydata=L2_stable_mani_arc.mani_states[:, 1], color='black', linewidth=1)
    p.save_figure(pathname+'CLtour_full_xy.png', dpi=100)
        
    t = LinePlot()
    t.grid()
    t.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    t.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
    t.plot(xdata=time, ydata=get_throttle_history(states[:, -3:]), color='black')
    t.save_figure(pathname+'CLtour_full_throttle.png', dpi=100)

    m = LinePlot()
    m.grid()
    m.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    m.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
    m.plot(xdata=time, ydata=states[:, 6], color='black')
    m.save_figure(pathname+'CLtour_full_mass.png', dpi=100)
    
    plt.show()


if __name__ == '__main__':

    try:
        os.mkdir('./Plots_For_AIAA_AAS_2022')
    except FileExistsError:
        print("Directory already exists")

    result_dict = solve()

    plot_phase(result_dict, pathname='./Plots_For_AIAA_AAS_2022/') # first+second phase
