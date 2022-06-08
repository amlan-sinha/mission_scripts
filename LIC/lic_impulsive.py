import sys, os

sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding/debug')
sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding')

import pydylan
import numpy as np
from GSFC_reference_waypoints import get_post_deployment_epoch_in_MJS_and_state_in_EJ2000, get_post_flyby_epoch_in_MJS_and_state_in_EJ2000, get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000

import matplotlib.pyplot as plt
from lineplot import LinePlot
from support import html_colors

strongorange, babyblue = html_colors['strong orange'], html_colors['baby blue']

days_to_seconds, seconds_to_days = 86400., 1/86400.


# Feasible initial condition with Fixed+Fixed+Fixed BC (Phase I)
def p1_initial_guess():

    return np.array([4.82482866e+05, 0.00000000e+00, 0.00000000e+00, 1.91579015e+00,
       4.51057070e-01, 1.90839267e-03, 1.65312883e+00, 0.00000000e+00,
       0.00000000e+00, 1.16583202e-01, 2.79846662e+00, 1.10813911e-03])


# Feasible initial condition with Fixed+Fixed+Fixed BC (Phase II)
def p2_initial_guess():

    return np.array([1.55321644e+07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 3.94207093e-05, 1.78186450e+00, 0.00000000e+00,
       0.00000000e+00, 2.18656193e+00, 6.28318531e+00, 0.00000000e+00,
       6.28318531e+00, 2.34690709e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       6.16235447e+00, 0.00000000e+00, 3.44085330e-03, 5.01890203e-01,
       9.84760037e-02, 1.84357500e-01, 6.28318531e+00, 0.00000000e+00,
       0.00000000e+00, 3.45259005e+00, 6.28318531e+00, 0.00000000e+00])


def output_control_to_screen(control):
    print('\nLength of control vector:', len(control), flush=True)
    print('\nThe control vector:', flush=True)
    for entry in control:
        print('{},'.format(entry), flush=True)


def get_plots_of_phase_integration(eom: pydylan.eom.Ephemeris_nBP) -> (LinePlot, LinePlot):

    epoch, state = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000() # at deployment
    duration = 7 * days_to_seconds

    rk54 = pydylan.integrators.RK54()
    rk54.set_eom(eom)
    rk54.set_time(epoch, epoch + duration)
    rk54.evaluate(state)

    p = LinePlot()
    p.grid()
    p.set_xlabel(xlabel_in=r'X (km)', fontsize=15)
    p.set_ylabel(ylabel_in=r'Y (km)', fontsize=15)
    # p.set_title(title_in='EJ2000 MJS: {}'.format(epoch), fontsize=20)
    p.plot(xdata=rk54.get_states()[:, 0], ydata=rk54.get_states()[:, 1], color='black', alpha=0.3)

    q = LinePlot()
    q.grid()
    q.set_xlabel(xlabel_in=r'X (km)', fontsize=15)
    q.set_ylabel(ylabel_in=r'Z (km)', fontsize=15)
    # q.set_title(title_in='EJ2000 MJS: {}'.format(epoch), fontsize=20)
    q.plot(xdata=rk54.get_states()[:, 0], ydata=rk54.get_states()[:, 2], color='black', alpha=0.3)

    p.plot(xdata=np.array([state[0], ]), ydata=np.array([state[1], ]), color='black', marker='x', markersize=5)
    q.plot(xdata=np.array([state[0], ]), ydata=np.array([state[2], ]), color='black', marker='x', markersize=5)

    post_flyby_epoch, state = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000() # after flyby around earth
    duration = 180 * days_to_seconds

    rk54 = pydylan.integrators.RK54()
    rk54.set_eom(eom)
    rk54.set_time(post_flyby_epoch, post_flyby_epoch + duration)
    rk54.evaluate(state)

    p.plot(xdata=rk54.get_states()[:, 0], ydata=rk54.get_states()[:, 1], color='black', alpha=0.3)
    q.plot(xdata=rk54.get_states()[:, 0], ydata=rk54.get_states()[:, 2], color='black', alpha=0.3)
    
    p.plot(xdata=np.array([state[0], ]), ydata=np.array([state[1], ]), color='black', marker='x', markersize=5)
    q.plot(xdata=np.array([state[0], ]), ydata=np.array([state[2], ]), color='black', marker='x', markersize=5)

    NRHO_insertion_epoch, state = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000() # right before NRHO insertion
    duration = 7 * days_to_seconds

    rk54 = pydylan.integrators.RK54()
    rk54.set_eom(eom)
    rk54.set_time(NRHO_insertion_epoch, NRHO_insertion_epoch + duration)
    rk54.evaluate(state)

    p.plot(xdata=rk54.get_states()[:, 0], ydata=rk54.get_states()[:, 1], color='black', alpha=0.3)
    q.plot(xdata=rk54.get_states()[:, 0], ydata=rk54.get_states()[:, 2], color='black', alpha=0.3)

    p.plot(xdata=np.array([state[0], ]), ydata=np.array([state[1], ]), color='black', marker='x', markersize=5)
    q.plot(xdata=np.array([state[0], ]), ydata=np.array([state[2], ]), color='black', marker='x', markersize=5)

    # using spline method to retrieve SPICE data
    number_of_steps = 5000
    number_of_spline_points = int(number_of_steps / 5)
    time_points = np.linspace(epoch, rk54.get_time()[-1], num=number_of_steps)

    splined_moon = pydylan.Body("Moon", True, number_of_spline_points, epoch, rk54.get_time()[-1])

    moon_states = np.zeros((len(time_points), 6))
    for i, time in enumerate(time_points):
        moon_states[i] = splined_moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(time)

    p.plot(xdata=moon_states[:, 0], ydata=moon_states[:, 1], color='black', alpha=0.3)
    q.plot(xdata=moon_states[:, 0], ydata=moon_states[:, 2], color='black', alpha=0.3)
    
    return p, q


# obtain the trajectory from the POST_DEPLOYMENT_STATE to NRHO_INSERTION_STATE with Fixed+Fixed+Fixed BCs
def solve(eom, thruster_parameters) -> (np.ndarray, np.ndarray):

    initial_epoch, initial_state = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()
    middle_epoch, middle_state = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000()
    target_epoch, target_state = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000()

    left_boundary_condition = pydylan.FixedBoundaryCondition(initial_state)
    middle_boundary_condition = pydylan.FixedBoundaryCondition(middle_state)
    right_boundary_condition = pydylan.FixedBoundaryCondition(target_state)

    snopt_options = pydylan.SNOPT_options_structure()
    snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.finite_differencing
    snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible
    snopt_options.quiet_SNOPT = False
    snopt_options.time_limit = 1 #60 * 60.

    mbh_options = pydylan.MBH_options_structure()
    mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
    mbh_options.max_step_size = 0.9
    mbh_options.quiet_MBH = False
    mbh_options.time_limit = 1 #60 * 300.

    phase_options_1 = pydylan.phase_options_structure()
    phase_options_1.number_of_segments = 2

    phase_options_1.earliest_initial_date_in_MJS = initial_epoch
    phase_options_1.minimum_initial_coast_time = 0.
    phase_options_1.maximum_initial_coast_time = 0.
    phase_options_1.minimum_final_coast_time = 0.
    phase_options_1.maximum_final_coast_time = 0.
    phase_options_1.minimum_shooting_time = middle_epoch - initial_epoch
    phase_options_1.maximum_shooting_time = middle_epoch - initial_epoch

    phase_options_1.match_point_position_constraint_tolerance = 1E+3
    phase_options_1.match_point_velocity_constraint_tolerance = 1E-2
    phase_options_1.control_coordinate_transcription = pydylan.enum.polar

    phase_options_2 = pydylan.phase_options_structure()
    phase_options_2.number_of_segments = 10
    
    phase_options_2.earliest_initial_date_in_MJS = middle_epoch
    phase_options_2.minimum_initial_coast_time = 0.
    phase_options_2.maximum_initial_coast_time = 0.
    phase_options_2.minimum_final_coast_time = 0.
    phase_options_2.maximum_final_coast_time = 0.
    phase_options_2.minimum_shooting_time = target_epoch - middle_epoch
    phase_options_2.maximum_shooting_time = target_epoch - middle_epoch

    phase_options_2.match_point_position_constraint_tolerance = 1E+3
    phase_options_2.match_point_velocity_constraint_tolerance = 1E-2
    phase_options_2.control_coordinate_transcription = pydylan.enum.polar

    mission = pydylan.Mission(eom, left_boundary_condition, middle_boundary_condition, pydylan.enum.mbh)
    mission.set_thruster_parameters(thruster_parameters)
    mission.add_phase_options(phase_options_1)
    mission.add_control_initial_guess(p1_initial_guess())
    mission.add_boundary_condition(right_boundary_condition)
    mission.add_phase_options(phase_options_2)
    mission.add_control_initial_guess(p2_initial_guess())
    mission.optimize(snopt_options, mbh_options)

    output_control_to_screen(mission.get_control_state())
    
    assert mission.is_best_solution_feasible()
    np.save("./Plots_For_LIC/lic_im_feasible_control_solutions.npy", mission.get_all_feasible_control_solutions())

    result = mission.evaluate_and_return_solution(mission.get_control_state(), forward_backward_shooting=True)

    time = result.time
    states = result.states

    return time, states

if __name__ == '__main__':
    
    # Load SPICE kernels
    kernels = pydylan.spice.load_spice()

    earth = pydylan.Body("Earth")
    sun = pydylan.Body("Sun")
    moon = pydylan.Body("Moon")

    eom = pydylan.eom.Ephemeris_nBP(earth)
    eom.add_secondary_body(sun)
    eom.add_secondary_body(moon)

    initial_epoch, _ = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()

    thruster_parameters = pydylan.ThrustParameters(fuel_mass=0., dry_mass=0., Isp=0., thrust=0., is_low_thrust=False)

    # Unforced dynamics integrated forward in time
    p, q = get_plots_of_phase_integration(eom)

    time, states = solve(eom, thruster_parameters)
    time = (time - initial_epoch) * seconds_to_days

    p.plot(xdata=states[:, 0], ydata=states[:, 1], color='black')
    q.plot(xdata=states[:, 0], ydata=states[:, 2], color='black')

    p.save_figure("./Plots_For_LIC/lic_im_xy.png")
    q.save_figure("./Plots_For_LIC/lic_im_xz.png")

    fig, axs = plt.subplots(3, 2, sharex=True)
    axs[0, 0].plot(time, states[:, 0])
    axs[0, 0].set_title('X-coordinate')
    axs[0, 0].grid()
    axs[1, 0].plot(time, states[:, 1])
    axs[1, 0].set_title('Y-coordinate')
    axs[1, 0].grid()
    axs[2, 0].plot(time, states[:, 2])
    axs[2, 0].set_title('Z-coordinate')
    axs[2, 0].grid()
    axs[0, 1].plot(time, states[:, 3])
    axs[0, 1].set_title('VX-coordinate')
    axs[0, 1].grid()
    axs[1, 1].plot(time, states[:, 4])
    axs[1, 1].set_title('VY-coordinate')
    axs[1, 1].grid()
    axs[2, 1].plot(time, states[:, 5])
    axs[2, 1].set_title('VZ-coordinate')
    axs[2, 1].grid()

    print('\nTransfer time: {} (days)'.format(time[-1]))

    plt.show()

    # Unload SPICE kernels
    pydylan.spice.unload_spice(kernels)
