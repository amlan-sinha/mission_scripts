import sys, os

sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding/debug')
sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding')

import pydylan
import numpy as np
import pandas as pd

from lineplot import LinePlot
import matplotlib.pyplot as plt
from support import html_colors

strongorange, babyblue = html_colors['strong orange'], html_colors['baby blue']


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


def generate_and_plot_manifold(orbit: pydylan.periodic_orbit, plot: LinePlot, coordinates=(0, 1), manifold_time=10., manifold_direction=pydylan.enum.PerturbationDirection.StableLeft):

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


def plot_phase(pathname):

    df = pd.read_csv(filepath, skipinitialspace = True, quotechar = '"')

    epoch = df.time.to_numpy()
    states = np.vstack((df.x.to_numpy(), df.y.to_numpy(), df.z.to_numpy(), 
                        df.xdot.to_numpy(), df.ydot.to_numpy(), df.zdot.to_numpy(), 
                        df.mass.to_numpy(), 
                        df.ux.to_numpy(), df.uy.to_numpy(), df.uz.to_numpy())).T

    print(epoch.shape)
    print(states[:,1])

    earth = pydylan.Body("Earth")
    moon = pydylan.Body("Moon")
    cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)

    # L1 Lyapunov
    libration_point_information_L1 = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L1)
    desired_orbit_energy_L1 = libration_point_information_L1[1] + 100E-4
    lyapunov_L1 = pydylan.periodic_orbit.Lyapunov(cr3bp, pydylan.enum.LibrationPoint.L1, desired_orbit_energy_L1)
    lyapunov_L1 = periodic_orbit_continuation(lyapunov_L1, desired_orbit_energy_L1)

    # L2 Lyapunov
    libration_point_information_L2 = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L2)
    desired_orbit_energy_L2 = desired_orbit_energy_L1
    lyapunov_L2 = pydylan.periodic_orbit.Lyapunov(cr3bp, pydylan.enum.LibrationPoint.L2, desired_orbit_energy_L2)
    lyapunov_L2 = periodic_orbit_continuation(lyapunov_L2, desired_orbit_energy_L2)

    L1_unstable_mani_arc = lyapunov_L1.generate_manifold_arc(lyapunov_L1.orbit_period, 3.5, pydylan.enum.PerturbationDirection.UnstableRight)    

    L2_stable_mani_arc = lyapunov_L2.generate_manifold_arc(lyapunov_L2.orbit_period, 4., pydylan.enum.PerturbationDirection.StableLeft)

    p = LinePlot()
    p.grid()
    p.set_xlabel(xlabel_in=r'X (DU)', fontsize=15)
    p.set_ylabel(ylabel_in=r'Y (DU)', fontsize=15)
    p = generate_and_plot_manifold(orbit=lyapunov_L1, plot=p, coordinates=(0, 1), manifold_time=5., manifold_direction=pydylan.enum.PerturbationDirection.UnstableRight)
    p = generate_and_plot_manifold(orbit=lyapunov_L2, plot=p, coordinates=(0, 1), manifold_time=7., manifold_direction=pydylan.enum.PerturbationDirection.StableLeft)
    p.plot(xdata=states[:, 0], ydata=states[:, 1], color='black', linewidth=1)
    p.plot(xdata=states[:, 0], ydata=states[:, 1], color='black', linewidth=1)

    p.plot(xdata=L1_unstable_mani_arc.mani_states[:, 0], ydata=L1_unstable_mani_arc.mani_states[:, 1], color='black', linewidth=1)
    p.plot(xdata=L2_stable_mani_arc.mani_states[:, 0], ydata=L2_stable_mani_arc.mani_states[:, 1], color='black', linewidth=1)

    # p.save_figure(pathname+'CLtour_full_xy.png', dpi=100)
        
    t = LinePlot()
    t.grid()
    t.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    t.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
    t.plot(xdata=epoch, ydata=get_throttle_history(states[:, -3:]), color='black')
    # t.save_figure(pathname+'CLtour_full_throttle.png', dpi=100)

    m = LinePlot()
    m.grid()
    m.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    m.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
    m.plot(xdata=epoch, ydata=states[:, 6], color='black')
    # m.save_figure(pathname+'CLtour_full_mass.png', dpi=100)
    
    plt.show()

if __name__ == '__main__':

    filepath = sys.argv[1]
    plot_phase(filepath)