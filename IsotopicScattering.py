"""
Isotopic scattering class.

Author: Jyotirmai (Joe) Singh
"""

from UtilityMethods import *
from BoundaryInteractions import propagate

def isotopic_scatter_rate(box, particle):
    """
    Calculates the isotopic scatter rate for the 
    phonon, given the details of the material of
    the box. 
    
    :param box: The box in which the phonon is.
    :param particle: The phonon selected for simulation. 
    :return: The isotopic scatter rate.
    """
    return box.get_material().get_isotope_scatter_rate() * (particle.get_f() ** 4)


def phonon_isotope_scatter(particle, t, box, points, colours):
    """
    Simulation of an isotopic scatter. The velocity direction is changed 
    randomly while the magnitude is kept fixed (since phonon types do not 
    change here) and then the particle is allowed to move forward for a time
    t. 
    
    :param particle: The phonon undergoing the isotopic scatter.
    :param t: The time interval over which the process occurs. 
    :param box: The box in which the simulation is happening. 
    :param points: The points associated with the matplotlib 3d scatterplot. 
                   This is updated after the particle's propagation step updates
                   its position coordinates. 
    """
    # advance time:
    particle.set_t(particle.get_t() + t)
    # box.update_time(particle.get_t())
    # simulate change of trajectory due to scatter
    curr_vx = particle.get_vx()
    curr_vy = particle.get_vy()
    curr_vz = particle.get_vz()
    x = particle.get_x()
    y = particle.get_y()
    z = particle.get_z()

    v_mag = get_magnitude(curr_vx, curr_vy, curr_vz)
    new_vx, new_vy, new_vz = create_random_spherical_vel(v_mag)
    particle.set_velocity(new_vx, new_vy, new_vz)

    event_str = particle.get_name() + ": Interaction Event occurred at %s" % particle.get_t() \
                + ". Isotopic scatter at (" + str(x) + ", " + str(y) + ", " + str(z) + ")." \
                + "New velocity: (" + str(new_vx) + ", " + str(new_vy) + ", " + str(new_vz) + ")."
    particle.add_event(event_str)
    print(event_str)

    # After scatter, simulate moving forward. Need to be careful about hitting boundary

    propagate(particle, box, t, points, colours)

    x_points = box.get_x_array()
    y_points = box.get_y_array()
    z_points = box.get_z_array()

    data = (x_points, y_points, z_points)

    points._offsets3d = data
    return
