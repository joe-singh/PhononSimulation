"""
Diffusive Propagation Methods

Author: Jyotirmai (Joe) Singh 6/8/18
"""

from UtilityMethods import closest_distance_to_box, beyond_boundary, create_random_spherical_vel
from AnharmonicDecay import anharmonic_decay_LLT, anharmonic_decay_LTT
import numpy as np


def check_diffusive_prop(particle, box, t_isotopic, t_anharmonic, t_boundary, isotopic_rate, anharmonic_rate,
                   sigma_tolerance=4, time_tolerance=100.0, isotopic_anharmonic_tolerance=100.0):

    type = particle.get_type()

    # Different sigma for the different diffusive propagation processes. Account for fact
    # that Transverse phonons do not undergo bulk anharmonic decay.

    sigma_squared = (1 / 3.0) * (t_isotopic * particle.get_v()) ** 2

    if type == 3:
        sigma_squared = (1 / 3.0) * t_isotopic * t_anharmonic * particle.get_v() ** 2

    sigma = sigma_squared ** .5
    print("SIGMA: %f" % sigma)

    # Check to see if far away from boundary. Sigma tolerance can be adjusted to set the tolerance
    # on how far from the boundary we must be to do diffusive propagation. The larger the sigma_tolerance
    # the slower the simulation will be. Distance here is the closest distance to the box, i.e. the perpendicular
    # distance to the closest face of the box.

    l_border = closest_distance_to_box(box, particle)
    in_bulk = l_border > sigma_tolerance * sigma
    if in_bulk:
        print(">>>>>>> In Bulk")

    # Condition of t_iso << t_boundary so that we don't have a ballistic phonon.
    # Time tolerance enforces the << and can be adjusted to balance speed and physical
    # correctness. The larger time_tolerance, the slower the simulation but the more
    # accurate physically.

    not_ballistic = t_isotopic < t_boundary / time_tolerance
    if not_ballistic:
        print(">>>>>>> Not Ballistic")

    # Condition that isotopic scatters should happen much more frequently than anharmonic
    # decays for diffusive propagation. Set by default to 10 so that we will do diffusive
    # propagation to increase efficiency when on average we have 10 isotopic scatters for every
    # anharmonic decay. Increaisng this number slows the simulation.

    isotopic_dominant = isotopic_rate / anharmonic_rate > isotopic_anharmonic_tolerance
    if isotopic_dominant:
        print(">>>>>>> Isotopic Dominant")

    return in_bulk and not_ballistic and isotopic_dominant, sigma


def diffusive_propagation(particle, box, sigma, t_anharmonic, t_isotopic,
                          t_anharmonic_LTT, t_anharmonic_LLT, points, colours):

    print("DIFFUSIVE PROPAGATION")
    x, y, z = particle.get_x(), particle.get_y(), particle.get_z()

    x_diffusive = np.random.normal(x, sigma)
    y_diffusive = np.random.normal(y, sigma)
    z_diffusive = np.random.normal(z, sigma)

    particle.set_x(x_diffusive)
    particle.set_y(y_diffusive)
    particle.set_z(z_diffusive)

    # While loop to ensure edge cases are dealt with. If we diffusively propagate outside
    # of the box, keep drawing numbers until we are inside the box.
    while beyond_boundary(particle, box):
        x_diffusive = np.random.normal(x, sigma)
        y_diffusive = np.random.normal(y, sigma)
        z_diffusive = np.random.normal(z, sigma)

        particle.set_x(x_diffusive)
        particle.set_y(y_diffusive)
        particle.set_z(z_diffusive)

    # Simulate random change of particle type
    # Pick new phonon type at random and choose velocity
    new_type = (np.random.choice(3, 1, p=[1/3.0, 1/3.0, 1/3.0]) + 1)[0]
    v_mag = box.get_material().get_particle_velocity(new_type)

    particle.set_type(new_type)
    colours[box.get_particle_no(particle.get_name())] = new_type

    # v_mag = get_magnitude(curr_vx, curr_vy, curr_vz)
    new_vx, new_vy, new_vz = create_random_spherical_vel(v_mag)
    particle.set_velocity(new_vx, new_vy, new_vz)

    if new_type == 3:
        particle.set_t(particle.get_t() + t_anharmonic)

        if min(t_anharmonic_LLT, t_anharmonic_LTT) == t_anharmonic_LLT:
            print("DIFFUSIVE ANHARMONIC LLT")
            anharmonic_decay_LLT(particle, box, 0, points, colours)
        else:
            print("DIFFUSIVE ANHARMONIC LTT")
            anharmonic_decay_LTT(particle, box, 0, points, colours)
    else:
        particle.set_t(particle.get_t() + t_isotopic)

    return
