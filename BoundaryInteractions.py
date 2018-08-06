"""
Class simulating boundary interactions. Assumes particles are
already at boundary. 

Author: Jyotirmai (Joe) Singh 10/7/18

"""
from UtilityMethods import *
import AnharmonicDecay


def specular_scatter(particle, box):
    """
    Simulates specular scattering. 
    
    :param particle: The particle undergoing the interaction
    :param box: The box in which the particle is in  
    """
    print("SPECULAR SCATTER")
    x, y, z = particle.get_x(), particle.get_y(), particle.get_z()
    vx, vy, vz = particle.get_vx(), particle.get_vy(), particle.get_vz()

    if x <= 0 or x >= box.width:
        vx = -vx

    elif y <= 0 or y >= box.height:
        vy = -vy

    elif z <= 0 or z >= box.depth:
        vz = -vz

    particle.set_velocity(vx, vy, vz)


def get_lambertian_probability(particle):

    # Convert frequency to GHz
    f = particle.get_f()/1e9

    return -2.98e-11 * (f**4) + 1.71e-8 * (f**3) - 2.47e-6 * (f**2) + 7.83e-4 * f + 5.88e-2


def get_sad_probability(particle):

    # Convert frequency to GHz
    f = particle.get_f()/1e9

    return 1.51e-14 * (f**5)  


def get_specular_probability(particle):

    # Convert frequency to GHz
    f = particle.get_f()/1e9

    return 2.9e-13 * (f**4) + 3.1e-9 * (f**3) - 3.21e-6 * (f**2) - 2.03e-4 * f + 0.928


def lambertian_scatter(particle, box):
    """
    Simulates scattering by a Lambertian process with 
    a polar angle dependent on a cos(theta) distribution. 
    
    :param particle: The particle undergoing the interaction
    :param box: The box in which the particle is in 
    """
    diffusive_angle = get_cos_angle()
    azimuthal_angle = np.random.uniform(0, 2 * PI)

    print("DIFFUSIVE SCATTER")
    adjust_boundary_velocity(particle, box, diffusive_angle, azimuthal_angle)


def cylindrical_vol_and_sa(r, h):
    """
    Calculate volume and surface area of a cylinder. 
    
    :param r: The radius in m
    :param h: The height in m
    :return: The volume and surface area
    """
    return PI * h * r ** 2, 2 * PI * (r * h + r ** 2)


def convert_w_to_T(w):
    """
    Converts from angular frequency to temperature as in Eq (4) 
    in Klistner and Pohl. 
    
    :param w: Angular frequency, 2 * PI * v_dom where v_dom is the dominant frequency
              Klistner and Pohl mention, Eq. 4 with the factor of 4.25
    :return: The temperature. 
    """
    return (h * w) / (2 * PI * k_b * 4.25)


def diffusive_scatter_rate(T, particle, box):
    """
    The total diffusive scatter rate (Lambertian + Surface Anharmonic) as 
    defined by Fig. 1 in Klistner and Pohl. The equation here is obtained by
    extracting the points from their plot and then fitting via matplotlib. 
    Multiply by 100 to correct for cm vs m. 
    
    :param T: Temperature
    :param particle: The particle
    :param box: The box
    :return: The decay rate.
    """
    material = box.get_material()
    phonon_velocity = material.get_particle_velocity(particle.get_type())
    return 100 * (0.036452 * (T ** 2) + 0.216544) * phonon_velocity


def boundary_interaction(particle, box, points, colours):
    """
    Method to simulate boundary interactions. Assumes the particle
    is already present at the boundary and then simulates Lambertian 
    or SAD scattering as necessary. 
    
    :param particle: The particle undergoing the interactions
    :param box: The box in which the particle is in 
    :param points: The array of points associated with the phonon locations
    :param colours: The colour configuration of the phonons
    """

    # First check if we are going to potentially be absorbed by the phonon detectors.

    if np.random.rand() < box.get_coverage():
        # Particle incident on aluminium
        if np.random.rand() < box.get_material().get_sensor_absorb_probability():
            # Particle absorbed by aluminium
            remove_particle(particle, box, points)
            #box.remove_particle(particle)

            #x_points = box.get_x_array()
            #y_points = box.get_y_array()
            #z_points = box.get_z_array()

            #data = (x_points, y_points, z_points)
            #points._offsets3d = data

            #colour_array = get_colour_array(box.colours.values())
            #points._facecolor3d = colour_array
            #points._edgecolor3d = colour_array
            return

    # If no absorption, continue the process.

    material = box.get_material()

    lambertian_prob = get_lambertian_probability(particle)
    sad_prob = get_sad_probability(particle)
    specular_prob = get_specular_probability(particle)

    # Rescale probabilities to account for minor errors that mean
    # probabilities do not sum exactly to 1
    total_prob = lambertian_prob + sad_prob + specular_prob

    lambertian_prob /= total_prob
    sad_prob /= total_prob
    specular_prob /= total_prob

    if particle.get_type() != 3:
        process = (np.random.choice(2, 1, p=[1-specular_prob, specular_prob]) + 1)[0]

    else:
        process = (np.random.choice(3, 1, p=[lambertian_prob, specular_prob, sad_prob]) + 1)[0]

    if process == 1:
        lambertian_scatter(particle, box)
    elif process == 2:
        specular_scatter(particle, box)
    else:
        assert process == 3
        LTT_prob = material.get_LTT_ratio()

        if np.random.rand() < LTT_prob:
            # Do LTT boundary decay
            print("SURFACE ANHARMONIC LTT")
            AnharmonicDecay.anharmonic_decay_LTT(particle, box, 0, points, colours, 1)

        else:
            # Do LLT boundary decay
            print("SURFACE ANHARMONIC LLT")
            AnharmonicDecay.anharmonic_decay_LLT(particle, box, 0, points, colours, 1)


def propagate(particle, box, t, points, colours):
    """
    Simulate particle moving forward for time t, taking hitting
    wall into account. 

    :param particle: Particle moving. 
    :param box: Box in which particle is in. 
    :param t: Time for which box is simulated to move. 
    """

    t_boundary, x_boundary, y_boundary, z_boundary = time_to_boundary(box, particle)

    # If particle propagates for time t and goes beyond the boundary, stop it at the
    # boundary and change its velocity to simulate bouncing off the wall
    if t_boundary <= t:
        particle.set_x(x_boundary)
        particle.set_y(y_boundary)
        particle.set_z(z_boundary)
        # Simulates specular/diffusive scattering.
        boundary_interaction(particle, box, points, colours)

    # Otherwise can safely propagate for time t without hitting the wall.
    else:
        print("In propagation non boundary case.")
        particle.advance(t)
