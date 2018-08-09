"""
Anharmonic decay class. Includes L->L+T and L->T+T decays. 
 
Author: Jyotirmai (Joe) Singh 10/7/18
"""

from Particle import Particle
from numpy import sin, cos, arccos, arctan2
from UtilityMethods import *
import BoundaryInteractions


def get_anharmonic_rate(box, particle, LLT):
    """
    Calculates the anharmonic decay rate associated with this 
    phonon. 
    
    :param box: The box in which the phonon is.  
    :param particle: The phonon under consideration. 
    :param LLT: Parameter controlling whether the rate calculated is for
                L->L+T or L->T+T interactions. If LLT != 0, then the rate is
                for the LLT reaction, otherwise for the LTT reaction. 
    :return: The anharmonic rate for the appropriate action chosen by the LLT parameter. 
             Automatically 0 if this is not a longitudinal phonon. 
    """
    # Make the anharmonic rate basically 0 if this is not a longitudinal phonon
    if particle.get_type() != 3:
        return 1e-70

    material = box.get_material()
    f = particle.get_f()

    if LLT:
        return material.get_LLT_rate() * (f ** 5)

    return material.get_LTT_rate() * (f ** 5)


def accept_reject(x_min, x_max, f, d, LTT, box, rand_max):
    """
    Accept-Reject algorithm to simulate drawing from a 
    distribution f which is a pdf in the range x_min and x_max. 
    
    :param x_min: The lower bound of the pdf's support.  
    :param x_max: The upper bound of the pdf's support. 
    :param f: The pdf to be drawn from. 
    :param rand_max: The maximum ceiling for the guesses, must 
                     be larger than the max of f over the range 
                     defined by x_min and x_max. 
    :return: the value drawn from distribution. 
    """

    while True:
        # Draw a random number over the range of possible ratios where the pdf is defined
        rand_ratio = np.random.uniform(x_min, x_max)

        # Draw a random number of the range 0 to rand_max:
        rand_draw = np.random.uniform(0, rand_max)

        if rand_draw <= f(rand_ratio, d, LTT, box):
            omega_ratio = rand_ratio
            break

    return omega_ratio


def convert_particle_to_global(phi, theta, phi_p, theta_p):
    """
    Method to convert spherical coordinates from a particle's frame of 
    reference (polar axis being the particle's velocity) to the lab 
    frame (polar axis being z-axis). 
    
    :param phi: The polar angle of the particle frame polar axis, relative
                to the global coordinates. 
    :param theta: The azimuthal angle of the particle frame polar axis, relative
                  to the global coordinates. 
    :param phi_p: The particle frame phi to be converted.
    :param theta_p: The particle frame theta to be converted. 
    :return: The converted phi, and converted theta. 
    """

    phi_global = arccos(-sin(phi) * sin(theta_p) * cos(phi_p) + cos(phi) * cos(theta_p))
    theta_global = theta - arctan2(-sin(theta_p) * sin(phi_p) * sin(phi) * cos(theta_p) - cos(phi) * cos(phi_p), 1)

    return phi_global, theta_global


def anharmonic_final_step(particle, box, t, colours, points, vx=0, vy=0, vz=0, new_particle=0):
    """
    Utility method to update relevant quantities once all anharmonic processes
    have been carried out. If vx, vy, and vz are all 0 (as is default), the method
    assumes the particle velocity has already been set by some other method and we are
    just updating colours/propagating the particle now. 
    
    :param particle: The particle undergoing surface decay
    :param box: The daughter particle produced as a result
    :param t: The time for which particles propagate afterwards.
    :param colours: The current colour configuration of the phonons
    :param points: The points giving the phonons' locations
    :param vx: The x component of the velocity
    :param vy: The y component of the velocity
    :param vz: The z component of the velocity
    :param new_particle: The new particle created in the anharmonic decay
    """
    print("###### Post Anharmonic Energy: %f, 2 * Energy Gap: %f" % (particle.get_energy()*1e19, 2 * ENERGY_GAP*1e19))

    if particle.get_energy() < 2 * ENERGY_GAP:
        if not new_particle:
            remove_particle(particle, box, points)
        return

    if vx or vy or vz:
        particle.set_velocity(vx, vy, vz)
        particle.calculate_new_k()

    if new_particle:
        box.add_particle(particle)
        colours[box.get_num_particles() - 1] = particle.get_type()
        x_points = box.get_x_array()
        y_points = box.get_y_array()
        z_points = box.get_z_array()

        data = (x_points, y_points, z_points)
        points._offsets3d = data

        colour_array = get_colour_array(box.colours.values())
        points._facecolor3d = colour_array
        points._edgecolor3d = colour_array
    else:
        colours[box.get_particle_no(particle.get_name())] = particle.get_type()

    BoundaryInteractions.propagate(particle, box, t, points, colours)
    return


# t argument should be 0 since surface interactions assume particle has come and
# hit the boundary already and now will undergo the change so no propagation should happen.
def surface_anharmonic_final_step(particle, new_particle, box, colours, t,
                                  points, theta_1, phi_1, theta_2, phi_2):
    """
    Utility method to finish execution of anharmonic methods once other boundary 
    processes have been carried out. 
    
    :param particle: The particle undergoing surface decay
    :param new_particle: The daughter particle produced as a result
    :param box: The box in which the interaction is occurring
    :param colours: The current colour configuration of the phonons
    :param t: The time argument is expected to be 0 since the principle 
              of boundary interactions is that they occur after the particle 
              has propagated to the wall but this may be changed if further propagation is
              wanted
    :param points: The points giving the phonons' locations
    :param theta_1: The polar angle with which the first particle reflects off the boundary
    :param phi_1: The azimuthal angle with which the first particle reflects off the boundary
    :param theta_2: The polar angle with which the second particle reflects off the boundary
    :param phi_2: The azimuthal angle with which the second particle reflects off the boundary 
    """

    adjust_boundary_velocity(particle, box, theta_1, phi_1)
    adjust_boundary_velocity(new_particle, box, theta_2, phi_2)
    anharmonic_final_step(particle, box, t, colours, points)
    anharmonic_final_step(new_particle, box, t, colours, points, new_particle=1)
    update_display(particle, new_particle, box, points, colours)


def update_display(original_particle, new_particle, box, points, colours):
    """
    Utility method to update the animation. 
    
    :param original_particle: The original particle which underwent the anharmonic
                              decay.
    :param new_particle: The new particle created by the decay. 
    :param box: The box in which we are simulating. 
    :param points: The phonon location information for the matplotlib animation. 
    :param colours: The current colour configuration of the phonons. 
    """

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
    return


def omega_pdf(x, d, LTT, box):

    material = box.get_material()

    if LTT:
            b = material.get_beta()
            g = material.get_gamma()
            l = material.get_lambda()
            m = material.get_mu()

            A = (1 / 2) * (1 - d ** 2) * (b + l + (1 + d ** 2) * (g + m))
            B = b + l + 2 * (d ** 2) * (g + m)
            C = b + l + 2 * (g + m)
            D = (1 - d ** 2) * (2 * b + 4 * g + l + 3 * m)

            y = (A + B * d * x - B * x ** 2) ** 2 + \
                (C * x * (d - x) - (D / (d - x)) * (x - d - (1 - d ** 2) / (4 * x))) ** 2
            return y

    y = (x ** -2) * (1 - x ** 2) ** 2 * \
        ((1 + x) ** 2 - (d ** 2) * (1 - x) ** 2) * \
        (1 + x ** 2 - (d ** 2) * (1 - x) ** 2) ** 2
    return y


def get_omega_pdf_bounds(d, LTT):

    if LTT:
        x_min = (d - 1) / 2.0
        x_max = (d + 1) / 2.0
    else:
        x_min = (d - 1) / (d + 1)
        x_max = 1.0

    return x_min, x_max


def get_post_anharmonic_split_cos_theta_dist(x, d, LTT):
    """
    Returns the cos(theta) values for the phonons following a 
    decay. LTT dictates whether these values are for the LTT case or
    the LLT one. From https://arxiv.org/pdf/1109.1193.pdf pg 11-12 although
    the ones for the LTT case are incorrect (their formula for cos(theta) is 
    not strictly bounded by ±1 so I have calculated my own using conservation 
    of 4-momentum. 
    """
    cos_theta_1, cos_theta_2 = 0.0, 0.0
    if LTT:
        cos_theta_1 = (1 - d ** 2 + 2 * x * d)/(2 * x)
        cos_theta_2 = (1 - x * cos_theta_1) / (d - x)
    else:
        cos_theta_1 = (1 + x ** 2 - (d ** 2) * (1 - x) ** 2) / (2 * x)
        cos_theta_2 = (1 - x ** 2 + (d ** 2) * (1 - x) ** 2) / (2 * d * (1 - x))

    return cos_theta_1, cos_theta_2


def generic_anharmonic_decay(particle, box, t, points, colours, boundary, LTT):
    """
    Method to simulate anharmonic decay in both LTT and LLT modes, and to simulate
    anharmonic decays on the boundary. Anharmonic decays turn one phonon into two 
    with the angles first defined relative to the original velocity vector of the 
    incident particle and then converted to the global coordinate system with the 
    polar axis pointing in the +z direction. 
    
    The omega distribution is governed by pdfs from Tamura, Phys. Rev. B 31, #4 
    and the distribution of cos(theta) of the new particles is from 
    https://arxiv.org/pdf/1109.1193.pdf
    
    :param particle: The particle undergoing the anharmonic decay
    :param box: The box in which the decay is happening
    :param t: The time of propagation of the daughter phonons
    :param points: The array of points associated with the phonon locations
    :param colours: The colour configuration of the phonons
    :param boundary: Flag to signal if this is a boundary event or not
    :param LTT: Flag to signal if we are simulating LTT or LLT decay. 0 if LLT 
                else LTT
    """
    material = box.get_material()
    V_TRANSVERSE = material.get_transverse_vel()
    V_LONGITUDINAL = material.get_longitudinal_vel()

    particle.set_t(particle.get_t() + t)
    # box.update_time(particle.get_t() + t)

    # Choose a new random type 1 or 2, corresponding to transverse phonons.
    new_phonon_type = np.random.randint(1, 3)

    # Get angles of initial velocity. Need this for coordinate conversion later.
    theta, phi = get_velocity_angles(particle.get_vx(), particle.get_vy(), particle.get_vz())

    # Change original particle to a transverse type if LTT decay.
    if LTT:
        # Choose a new random type 1 or 2, corresponding to transverse phonons.
        old_phonon_type = np.random.randint(1, 3)
        particle.set_type(old_phonon_type)

    curr_x, curr_y, curr_z = particle.get_x(), particle.get_y(), particle.get_z()

    # Create new phonon with 0 momentum and frequency traveling in the same direction as the initial.
    # We will change all these variables below.
    new_phonon = Particle(curr_x, curr_y, curr_z, 0, 0, 0,
                            "Particle " + str(box.get_num_particles()),
                            new_phonon_type, 0, t=0)

    w_0 = particle.get_w()

    # If we are using this for a boundary anharmonic decay, simply pick a random direction obeying
    # Lambertian diffusion.
    if boundary:
        theta_1, theta_2 = get_cos_angle(), get_cos_angle()
        phi_1, phi_2 = np.random.uniform(0, 2 * PI), np.random.uniform(0, 2 * PI)

    d = V_LONGITUDINAL / V_TRANSVERSE

    x_min, x_max = get_omega_pdf_bounds(d, LTT)

    rand_max = 2.5

    if LTT:
        rand_max = 10.0

    omega_ratio = accept_reject(x_min, x_max, omega_pdf, d, LTT, box, rand_max)

    if LTT:
        w_0_after = omega_ratio * w_0 / d
    else:
        w_0_after = omega_ratio * w_0

    w_new_after = w_0 - w_0_after

    particle.set_w(w_0_after)
    new_phonon.set_w(w_new_after)

    # Energy Gap defined in Utility Methods.

    v_0 = V_LONGITUDINAL
    v_new = V_TRANSVERSE

    if LTT:
        v_0 = V_TRANSVERSE

    # Having set the energy, if this is the boundary case, we can just change the
    # velocities and end this.
    if boundary:

        v_0_x, v_0_y, v_0_z = spherical_to_cartesian(v_0, theta_1, phi_1)
        v_new_x, v_new_y, v_new_z = spherical_to_cartesian(v_new, theta_2, phi_2)

        particle.set_velocity(v_0_x, v_0_y, v_0_z)
        new_phonon.set_velocity(v_new_x, v_new_y, v_new_z)

        surface_anharmonic_final_step(particle, new_phonon, box, colours,
                                      0, points, theta_1, phi_1, theta_2,
                                      phi_2)
        return

    x = omega_ratio

    cos_theta_1, cos_theta_2 = get_post_anharmonic_split_cos_theta_dist(x, d, LTT)
    theta_1, theta_2 = arccos(cos_theta_1), arccos(cos_theta_2)
    phi_1 = np.random.uniform(0, 2 * PI)

    phi_0_after, theta_0_after = convert_particle_to_global(phi, theta, phi_1, theta_1)
    phi_new_after, theta_new_after = convert_particle_to_global(phi, theta, 2 * PI - phi_1, theta_2)

    v_0_x, v_0_y, v_0_z = spherical_to_cartesian(v_0, theta_0_after, phi_0_after)
    v_new_x, v_new_y, v_new_z = spherical_to_cartesian(v_new, theta_new_after, phi_new_after)

    # Now can set velocity coordinates. Recalculate k vectors also.
    anharmonic_final_step(particle, box, t, colours, points, v_0_x, v_0_y, v_0_z)
    anharmonic_final_step(new_phonon, box, t, colours, points, v_new_x, v_new_y, v_new_z, new_particle=1)
    return


def anharmonic_decay_LLT(particle, box, t, points, colours, boundary=0):
    generic_anharmonic_decay(particle, box, t, points, colours, boundary, 0)


def anharmonic_decay_LTT(particle, box, t, points, colours, boundary=0):
    generic_anharmonic_decay(particle, box, t, points, colours, boundary, 1)

