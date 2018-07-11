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
              Klistner and Pohl mention. 
    :return: The temperature. 
    """
    return (h * w) / (2 * PI * k_b)


def diffusive_scatter_rate(T, particle, box):
    """
    The total diffusive scatter rate (Lambertian + Surface Anharmonic) as 
    defined by Fig. 1 in Klistner and Pohl. The equation here is obtained by
    extracting the points from their plot and then fitting via matplotlib. 
    Multiply by 100 to correct for cm vs m. 
    
    :param T: Temperature
    :return: The decay rate.
    """
    material = box.get_material()
    phonon_velocity = material.get_particle_velocity(particle.get_type())
    return 100 * (0.036452 * (T ** 2) + 0.216544) * phonon_velocity


def get_diffusive_scatter_rates(particle, box):
    """
    Get scattering rates for cosine scattering and surface anharmonic decay. 
    First get rate for surface decay using Trumpp and Eisenmenger. Major 
    problem is that they have not defined the exact dimensions of their sample. 
    
    We assume a model of the form G_sad = <v>/4 [f_b * A_b / V + f_d * A_d / V] 
    where G is the decay rate, A_d is the area covered with detectors and A_b is the bare area. f_b is 
    the probability of anharmonic decay on the bare surface, f_d is the anharmonic
    decay on the detector surface, and <v> is the average phonon speed. We want to find f_b, 
    and we can fit their data which is a function of A_d / V so their y-intercept will give us
    
    y_0 = <v>/4 [f_b * A_b / V] but we do not know either A_b or A_d or the total surface area. They
    do give the range of dimensions of their samples, which they say are cylindrical. The diameter is 
    15 mm and length is 6-20 mm. We will assume here, for lack of better knowledge, a length of 13 mm.
    
    :param particle: The particle for which to calculate rates. 
    :param box: The box in which the decay is taking place. 
    :return The lambertian scattering and surface anharmonic decay rate. 
    """
    w = particle.get_w()
    T = convert_w_to_T(w)
    lambertian_rate = diffusive_scatter_rate(T, particle, box)

    # Get the velocity in the material for the given phonon type.
    v_avg = box.get_material().get_particle_velocity(particle.get_type())
    vol, SA = cylindrical_vol_and_sa(7.5e-3, 13e-3)

    # The intercept of the Trumpp Eisenmenger Line in Fig. 4, fitted to give 66.67 µsec
    y_0 = 1/66.67e-6
    f_SAD_280GHz = 4 * v_avg * y_0 * vol / SA

    # Scales as w^5, multiply by 2PI to convert frequency to angular frequency.
    f_SAD = (w / (2 * PI * 280e9)) ** 5 * f_SAD_280GHz

    # Rate of surface anharmonic decay when no detector space present, y intercept of
    # Fig. 4. Scales as w^5
    SAD_rate = y_0 * (w / (2 * PI * 280e9)) ** 5

    # Anharmonic decay only for longitudinal phonons.
    if particle.get_type() != 3:
        SAD_rate = 1e-70

    assert lambertian_rate > 0.0 and SAD_rate > 0.0

    return lambertian_rate, SAD_rate


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

    material = box.get_material()
    lambertian_rate, sad_rate = get_diffusive_scatter_rates(particle, box)

    r = np.random.rand()
    t_lambert = -np.log(r)/lambertian_rate
    t_sad = -np.log(r)/sad_rate

    if t_lambert < t_sad:
        lambertian_scatter(particle, box)
    else:
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
