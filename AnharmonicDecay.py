"""File containing methods to simulate anharmonic decay."""

from Particle import Particle
from numpy import sin, cos, arccos, arctan2
from UtilityMethods import *

"""Process controls whether we calculate rate for LTT or LLT. 1 = LLT, 0 = LTT"""
def get_anharmonic_rate(box, particle, LLT):

    # TODO: MAKE MATERIAL CLASS THAT STORES ALL MATERIAL SPECIFIC CONSTANTS. 
    # Make the anharmonic rate basically 0 if this is not a longitudinal phonon
    if particle.get_type() != 3:
        return 1e-69

    material = box.get_material()
    f = particle.get_f()

    if LLT:
        return material.get_LLT_rate() * (f ** 5)

    return material.get_LTT_rate() * (f ** 5)



def accept_reject(x_min, x_max, f, rand_max):

    while True:
        # Draw a random number over the range of possible ratios where the pdf is defined
        rand_ratio = np.random.uniform(x_min, x_max)

        # Draw a random number of the range 0 to rand_max:
        rand_draw = np.random.uniform(0, rand_max)

        if rand_draw <= f(rand_ratio):
            omega_ratio = rand_ratio
            break

    return omega_ratio


def convert_particle_to_global(phi, theta, phi_p, theta_p):

    phi_global = arccos(-sin(phi) * sin(theta_p) * cos(phi_p) + cos(phi) * cos(theta_p))
    theta_global = theta - arctan2(-sin(theta_p) * sin(phi_p) * sin(phi) * cos(theta_p) - cos(phi) * cos(phi_p), 1)

    return phi_global, theta_global


def anharmonic_final_step(particle, box, t, colours, title, vx, vy, vz, new_particle=0):

    particle.set_velocity(vx, vy, vz)
    particle.calculate_new_k()

    if new_particle:
        box.add_particle(particle)
        colours[box.get_num_particles()-1] = particle.get_type()
    else:
        colours[box.get_particle_no(particle.get_name())] = particle.get_type()

    propagate(particle, box, t, title)
    return


def update_display(original_particle, new_particle, box, points, title, colours):

    colour_array = get_colour_array(colours.values())
    points._facecolor3d = colour_array
    points._edgecolor3d = colour_array

    x_points = box.get_x_array()
    y_points = box.get_y_array()
    z_points = box.get_z_array()

    data = (x_points, y_points, z_points)

    event_str = original_particle.get_name() + ": Interaction Event occurred at %s" % new_particle.get_t() \
                + ".  " + original_particle.get_name() + " splits to produce " + new_particle.get_name() \
                + " at (" + str(original_particle.get_x()) + ", " + str(original_particle.get_y()) + ", " \
                + str(original_particle.get_z()) + ") with velocity (" + str(original_particle.get_vx()) + ", " \
                + str(original_particle.get_vy()) + ", " + str(original_particle.get_vz()) + ")."

    original_particle.add_event(event_str)
    new_particle.add_event(event_str)
    print(event_str)

    points._offsets3d = data
    title.set_text('Phonon Simulation: time={0:.8f}'.format(original_particle.get_t()))
    return


def anharmonic_decay_LLT(particle, box, t, points, colours, title):

    material = box.get_material()
    V_TRANSVERSE = material.get_transverse_vel()
    V_LONGITUDINAL = material.get_longitudinal_vel()

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

    # Get angles of initial velocity. Need this for coordinate conversion later.
    theta, phi = get_velocity_angles(particle.get_vx(), particle.get_vy(), particle.get_vz())

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
    omega_ratio = accept_reject(x_min, 1, omega_longitudinal_distribution, rand_max)

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

    # Convert to spherical coordinates with z axis now pointing up. From https://arxiv.org/pdf/1109.1193.pdf
    # p 12, eqs 15 & 16.
    phi_L_original = phi_L

    phi_L, theta_L = convert_particle_to_global(phi, theta, phi_L, theta_L_after)
    phi_T, theta_T = convert_particle_to_global(phi, theta, 2 * PI - phi_L_original, theta_T_after)

    v_L_x, v_L_y, v_L_z = spherical_to_cartesian(V_LONGITUDINAL, theta_L, phi_L)
    v_T_x, v_T_y, v_T_z = spherical_to_cartesian(V_TRANSVERSE, theta_T, phi_T)

    # Set velocity coordinates and new k vector.
    anharmonic_final_step(particle, box, t, colours, title, v_L_x, v_L_y, v_L_z)
    anharmonic_final_step(transverse_phonon, box, t, colours, title, v_T_x, v_T_y, v_T_z, new_particle=1)

    # Update display
    update_display(particle, transverse_phonon, box, points, title, colours)
    return


def anharmonic_decay_LTT(particle, box, t, points, colours, title):

    material = box.get_material()
    V_TRANSVERSE = material.get_transverse_vel()
    V_LONGITUDINAL = material.get_longitudinal_vel()

    # Advance time
    box.update_time(particle.get_t() + t)

    # Create new particle that will be the transverse phonon. Choose
    # randomly from ST and FT

    new_phonon_type = np.random.randint(1, 3)
    old_phonon_new_type = np.random.randint(1, 3)

    # Change the old particle to a transverse phonon type.
    particle.set_type(old_phonon_new_type)

    curr_x, curr_y, curr_z = particle.get_x(), particle.get_y(), particle.get_z()

    # Create new phonon with 0 momentum and frequency traveling in the same direction as the initial.
    # We will change all these variables below.
    transverse_phonon = Particle(curr_x, curr_y, curr_z, 0, 0, 0,
                                 "Particle " + str(box.get_num_particles()),
                                 new_phonon_type, 0, t=particle.get_t())

    w_0 = particle.get_w()

    # Get angles of initial velocity. Need this for coordinate conversion later.
    theta, phi = get_velocity_angles(particle.get_vx(), particle.get_vy(), particle.get_vz())

    # Now need to pull new omega value based on the statistics. From the papers
    # (https://arxiv.org/pdf/1109.1193.pdf and Tamura PRB 48 #13 1993) we know
    # the distribution for w_L'/w_L is given by the following function.

    d = V_LONGITUDINAL / V_TRANSVERSE
    x_1 = (d - 1) / 2.0
    x_2 = (d + 1) / 2.0

    def omega_longitudinal_distribution(x):

        # Make sure within bounds where this is a valid pdf
        assert x_1 <= x <= x_2
        material = box.get_material()

        # Material specific decay constants.

        b = material.get_beta()
        g = material.get_gamma()
        l = material.get_lambda()
        m = material.get_mu()

        A = (1 / 2) * (1 - d ** 2) * (b + l + (1 + d ** 2) * (g + m))
        B = b + l + 2 * (d ** 2) * (g + m)
        C = b + l + 2 * (g + m)
        D = (1 - d ** 2) * (2 * b + 4 * g + l + 3 * m)

        y = (A + B * d * x - B * x ** 2) ** 2 + (C * x * (d - x) - (D / (d - x)) * (x - d - (1 - d ** 2) / (4 * x))) ** 2
        return y

    # For the accept reject method, we need a good way of identifying a suitable maximum value
    # that is also economical in terms of computational time. Since for germanium we can plot this
    # distribution since we know d exactly, we can hardcode an economical maximum value in for now.

    rand_max = 10.0
    omega_ratio = accept_reject(x_1, x_2, omega_longitudinal_distribution, rand_max)

    # Set new particle omegas. Watch out for different definition of x here, its omega_ratio * d
    w_T1 = omega_ratio * w_0 / d
    w_T2 = w_0 - w_T1

    particle.set_w(w_T1)
    transverse_phonon.set_w(w_T2)

    # We can also use omega_ratio to get the final angles of the two phonons now
    # from the initial velocity of the incident phonon, which we will use as the z-axis.

    x = omega_ratio

    cos_theta_t1 = (1 - d ** 2 + 2 * x * d)/(2 * x)

    # From https://arxiv.org/pdf/1109.1193.pdf
    # p 12, eqs 15 & 16. Incorrect formulas though, equations 13 and 14
    # are not strictly within -1 to 1 for the given range. I have recalculated
    # my own from 4-momentum conservation.
    theta_T1 = arccos(cos_theta_t1)
    theta_T2 = arccos((1 - x * cos_theta_t1) / (d - x))

    phi_T1 = np.random.uniform(0, 2 * PI)
    phi_T1_original = phi_T1

    # Convert to spherical coordinates with z axis now pointing up. From https://arxiv.org/pdf/1109.1193.pdf
    # p 12, eqs 15 & 16.

    phi_T1, theta_T1 = convert_particle_to_global(phi, theta, phi_T1, theta_T1)
    phi_T2, theta_T2 = convert_particle_to_global(phi, theta, 2 * PI - phi_T1_original, theta_T2)

    v_T1_x, v_T1_y, v_T1_z = spherical_to_cartesian(V_TRANSVERSE, theta_T1, phi_T1)
    v_T2_x, v_T2_y, v_T2_z = spherical_to_cartesian(V_TRANSVERSE, theta_T2, phi_T2)

    # Now can set velocity coordinates. Recalculate k vectors also.
    anharmonic_final_step(particle, box, t, colours, title, v_T1_x, v_T1_y, v_T1_z)
    anharmonic_final_step(transverse_phonon, box, t, colours, title, v_T2_x, v_T2_y, v_T2_z, new_particle=1)

    update_display(particle, transverse_phonon, box, points, title, colours)
    return