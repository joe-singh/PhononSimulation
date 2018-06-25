"""File containing methods to simulate anharmonic decay."""

from Particle import Particle
from Box import Box
import numpy as np
from UtilityMethods import *

# TODO: PUT THE SCARY INTEGRAL STUFF IN HERE LATER.
def get_anharmonic_rate(particle):

    # Make the anharmonic rate 0 if this is not a longitudinal phonon
    if particle.get_type() != 3:
        return 0.0

    return 1.62e-54 * (particle.get_f() ** 5)


def anharmonic_decay_LLT(particle, box, t, points, colours, title):

    # Advance time
    box.update_time(particle.get_t() + t)

    # Create new particle that will be the transverse phonon. Choose
    # randomly from ST and FT

    new_phonon_type = np.random.randint(1,3)
    curr_x, curr_y, curr_z = particle.get_x(), particle.get_y(), particle.get_z()

    # Create new phonon with 0 momentum and frequency traveling in the same direction as the initial.
    # We will change all these variables below.
    transverse_phonon = Particle(curr_x, curr_y, curr_z, 0, 0, 0,
                            "Particle " + str(box.get_num_particles()),
                            new_phonon_type, 0, t=particle.get_t())

    w_0 = particle.get_w()

    # Now need to pull new omega value based on the statistics. From the papers
    # (https://arxiv.org/pdf/1109.1193.pdf and Tamura PRB 48 #13 1993) we know
    # the distribution for w_L'/w_L is given by the following function.

    d = V_LONGITUDINAL / V_TRANSVERSE
    x_min = (d - 1) / (d + 1)

    def omega_longitudinal_distribution(x):

        # Make sure within bounds where this is a valid pdf
        assert x_min <= x <= 1

        y = (x ** -2) * (1 - x ** 2) ** 2 * \
            ((1 + x) ** 2 - (d ** 2) * (1 - x) ** 2) * \
            (1 + x ** 2 - (d ** 2) * (1 - x) ** 2) ** 2
        return y

        # For the accept reject method, we need a good way of identifying a suitable maximum value
        # that is also economical in terms of computational time. Since for germanium we can plot this
        # distribution since we know d exactly, we can hardcode an economical maximum value in for now.

    rand_max = 2.5
    omega_ratio = 0.0

    while True:
        # Draw a random number over the range of possible ratios where the pdf is defined
        rand_ratio = np.random.uniform(x_min, 1)

        # Draw a random number of the range 0 to rand_max:
        rand_draw = np.random.uniform(0, rand_max)

        if rand_draw <= omega_longitudinal_distribution(rand_ratio):
            omega_ratio = rand_ratio
            break

    # Set new particle omegas.
    w_L = omega_ratio * w_0
    w_T = w_0 - w_L

    particle.set_w(w_L)
    transverse_phonon.set_w(w_T)

    # We can also use omega_ratio to get the final angles of the two phonons now
    # from the initial velocity of the incident phonon, which we will use as the z-axis.

    x = omega_ratio

    theta_L_after = np.arccos((1 + x ** 2 - (d ** 2) * (1 - x) ** 2) / (2 * x))
    theta_T_after = np.arccos((1 - x ** 2 + (d ** 2) * (1 - x) ** 2) / (2 * d * (1 - x)))

    phi_L = np.random.uniform(0, 2 * PI)
    phi_T = np.random.uniform(0, 2 * PI)

    # This final velocity is in terms of the particle's intrinsic coordinate system with z axis aligned
    # with initial velocity. We need to convert this to the global coordinate system aligned with the
    # boundaries.
    v_L_x, v_L_y, v_L_z = spherical_to_cartesian(V_LONGITUDINAL, theta_L_after, phi_L)
    v_L_x, v_L_y, v_L_z = convert_from_particle_to_global_cartesian(v_L_x, v_L_y, v_L_z)

    v_T_x, v_T_y, v_T_z = spherical_to_cartesian(V_TRANSVERSE, theta_T_after, phi_T)
    v_T_x, v_T_y, v_T_z = convert_from_particle_to_global_cartesian(v_T_x, v_T_y, v_T_z)

    # Now can set velocity coordinates.
    particle.set_velocity(v_L_x, v_L_y, v_L_z)
    transverse_phonon.set_velocity(v_T_x, v_T_y, v_T_z)

    # Set new k vectors
    particle.calculate_new_k()
    transverse_phonon.calculate_new_k()

    # Now having done all of this, put the new transverse phonon into the box.
    box.add_particle(transverse_phonon)

    # Propagate both particles.
    propagate(particle, box, t, title)
    propagate(transverse_phonon, box, t, title)

    old_position = np.array([curr_x, curr_y, curr_z])
    new_position = np.array([particle.get_x(), particle.get_y(), particle.get_z()])

    print("Time: %f" % t)
    print("Delta = " + str(new_position - old_position))

    # Update particle creation on display.
    colours[box.get_num_particles() - 1] = transverse_phonon.get_type()

    colour_array = get_colour_array(colours.values())
    points._facecolor3d = colour_array
    points._edgecolor3d = colour_array

    x_points = box.get_x_array()
    y_points = box.get_y_array()
    z_points = box.get_z_array()

    data = (x_points, y_points, z_points)

    event_str = particle.get_name() + ": Interaction Event occurred at %s" % particle.get_t() \
                + ".  " + particle.get_name() + " splits to produce " + transverse_phonon.get_name() \
                + " at (" + str(particle.get_x()) + ", " + str(particle.get_y()) + ") with velocity (" \
                + str(particle.get_vx()) + ", " + str(particle.get_vy()) + ")."

    particle.add_event(event_str)
    transverse_phonon.add_event(event_str)
    print(event_str)

    points._offsets3d = data
    title.set_text('Phonon Simulation: time={0:.8f}'.format(particle.get_t()))

    return



def anharmonic_decay(particle, box, t, points, colours, title):

    # Firstly advance time
    box.update_time(particle.get_t() + t)

    random_type = np.random.randint(1, 4)
    curr_x, curr_y, curr_z = particle.get_x(), particle.get_y(), particle.get_z()

    # Create new particle with 0 momentum/frequency. We will give it
    # correct frequency in a bit.
    new_particle = Particle(curr_x, curr_y, curr_z, 0, 0, 0,
                            "Particle " + str(box.get_num_particles()),
                            random_type, 0, t=particle.get_t())

    # Give new and old particle correct properties. Using old particle
    # with 0 velocity/wavevector/frequency means this process is effectively
    # like breaking one particle into two.
    particle.simulate_momentum_conservation(new_particle)

    # Put new particle in.
    box.add_particle(new_particle)

    # Propagate both particles.
    propagate(particle, box, t, title)
    propagate(new_particle, box, t, title)

    old_position = np.array([curr_x, curr_y, curr_z])
    new_position = np.array([particle.get_x(), particle.get_y(), particle.get_z()])

    print("Time: %f" % t)
    print("Delta = " + str(new_position - old_position))

    # Update particle creation on display.
    colours[box.get_num_particles() - 1] = new_particle.get_type()

    colour_array = get_colour_array(colours.values())
    points._facecolor3d = colour_array
    points._edgecolor3d = colour_array

    x_points = box.get_x_array()
    y_points = box.get_y_array()
    z_points = box.get_z_array()

    data = (x_points, y_points, z_points)

    event_str = particle.get_name() + ": Interaction Event occurred at %s" % particle.get_t() \
                + ".  " + particle.get_name() + " splits to produce " + new_particle.get_name() \
                + " at (" + str(particle.get_x()) + ", " + str(particle.get_y()) + ") with velocity (" \
                + str(particle.get_vx()) + ", " + str(particle.get_vy()) + ")."

    particle.add_event(event_str)
    new_particle.add_event(event_str)
    print(event_str)

    points._offsets3d = data
    title.set_text('Phonon Simulation: time={0:.8f}'.format(particle.get_t()))

    return