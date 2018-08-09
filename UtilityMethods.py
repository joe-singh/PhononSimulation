"""
Utility methods for things such as controlling bouncing
off the box walls and calculating velocity angles. 

Author: Jyotirmai (Joe) Singh 26/6/18
"""
import numpy as np
import os
import sys
import Material

PI = np.pi
h = 6.63e-34
hbar = h/(2 * PI)
k_b = 1.38e-23

# Al superconducting gap, in Joules. Value used is 
# 1.75e-4 eV. 

ENERGY_GAP = 1.75e-4 * (1.6e-19)

# Material specific constants from https://arxiv.org/pdf/1109.1193.pdf table 1.
# Silicon debye frequency from https://lampx.tugraz.at/~hadley/ss1/phonons/table/dosdebye.html
# Germanium debye frequency not correct! Completely random.
Germanium = Material.Material("Germanium", 3.67e-41, 6.43e-55, 5310, 3250, -0.732, -0.708, 0.376, 0.561,
                              5.32, 0.260, 0.5)
Silicon = Material.Material("Silicon", 2.43e-42, 7.41e-56, 9000, 5400, -0.429, -0.945, 0.524, 0.680,
                            2.33, 0.204, 0.5)

# Characteristic Colours to represent different phonon types.
# ST - Slow Transverse
# FT - Fast Transverse
# L - Longitudinal

ST = np.array([0.0,0.0,1.0,1.0])
FT = np.array([0.0,1.0,0.0,1.0])
L = np.array([1.0,0.0,0.0,1.0])

colour_dictionary = {
    1: ST,
    2: FT,
    3: L
}


def get_polar_angle(corner=0):
    """
    Getting a random polar angle sampling from a
    sin(theta) distribution. 
    :param corner: 1 if we are in a corner, to restrict 
                   polar angle to 0 - PI/2.
    :return: The polar angle sampled from a sin(theta) distribution.
    """
    # We need to use random numbers from -1 to 1 to get this angle.
    # Because the CDF of sin(x) from 0 to pi is F = 1/2(1-cos(x))
    # so x = arccos(1-2F) where F is from 0 to 1 which means
    # 1 - 2F is from -1 to 1. The adjustment from 0 to 1 gives
    # polar angles of 0 to pi/2, needed for the corners.
    rand = np.random.uniform(-1, 1)

    if corner:
        rand = np.random.uniform(0, 1)

    return np.arccos(rand)


def get_magnitude(vx, vy, vz):
    """
    Get vector magnitude. 
    :param vx: x component.
    :param vy: y component. 
    :param vz: z component. 
    :return: Magnitude of (vx, vy, vz) 
    """
    return (vx ** 2 + vy ** 2 + vz ** 2)**.5


def get_colour_array(num_array):
    """
    Assigns colours to each number in num_array,
    which represents individual particles. 
    
    :param num_array: Array containing numbers to identify
                      particles with and assign colours.
    :return: An array where each position corresponds to the colour of the
             particle at the corresponding position in num_array. 
    """
    return np.array([colour_dictionary[num] for num in num_array])


def beyond_boundary(particle, box):
    """
    Checks if particle is outside or on the boundaries of the box. 
    
    :param particle: Particle under consideration. 
    :param box: The box in which the particle is in. 
    :return: True if particle is outisde the box, False otherwise. 
    """
    x = particle.get_x()
    y = particle.get_y()
    z = particle.get_z()

    width = box.get_width()
    height = box.get_height()
    depth = box.get_depth()

    return x >= width or x <= 0 or y >= height or y <= 0 or z <= 0 or z >= depth


def check_no_particle_outside_box(box):
    """
    Checks if any particle is outside the box. 
     
    :param box: Box. 
    :return: True if all particles are inside, False if one is outside. 
    """
    width = box.get_width()
    height = box.get_height()
    depth = box.get_depth()
    for ptcle in box.particles:
        particle = box.particles[ptcle]
        x = particle.get_x()
        y = particle.get_y()
        z = particle.get_z()
        if x > width or x < 0 or y > height or y < 0 or z < 0 or z > depth:
            particle.get_info()
            print(get_velocity_angles(particle.vx, particle.vy, particle.vz))
            return False
    return True


def spherical_to_cartesian(r, polar, azimuthal):
    """
    Converts spherical coordinates to cartesian coordinates.
    
    :param r: Radius coordinate.  
    :param polar: Polar angle. 
    :param azimuthal: Azimuthal angle. 
    :return: x, y, z coordinates. 
    """
    x = r * np.sin(polar) * np.cos(azimuthal)
    y = r * np.sin(polar) * np.sin(azimuthal)
    z = r * np.cos(polar)

    return x, y, z


def create_random_spherical_vel(v_mag):
    """
    Creates a velocity with magnitude v_mag 
    and randomised direction. 
    
    :param v_mag: The output velocity magnitude. 
    :return: 
    """
    rand_polar = get_polar_angle()
    rand_azimuthal = np.random.uniform(0, 2 * PI)
    return spherical_to_cartesian(v_mag, rand_polar, rand_azimuthal)


def get_velocity_angles(vx, vy, vz):
    """
    Get polar and azimuthal angles of vector (vx, vy, vz)
    
    :param vx: x component.
    :param vy: y component. 
    :param vz: z component. 
    :return: Polar and azimuthal angles. 
    """

    polar = 0
    azimuthal = 0

    plane_component = (vx ** 2 + vy ** 2) ** .5

    # Get polar angle first, from 0 to pi
    if vz >= 0:
        polar = np.arctan(plane_component/vz)
    else:
        polar = PI + np.arctan(plane_component/vz)

    # Get azimuthal angle, from 0 to 2pi

    if vx >= 0 and vy >= 0:
        azimuthal = np.arctan(vy/vx)
    elif vx < 0:
        azimuthal = PI + np.arctan(vy/vx)
    elif vx >= 0 and vy < 0:
        azimuthal = 2*PI + np.arctan(vy/vx)

    return polar, azimuthal


def time_to_boundary(box, particle):
    """
    Calculates the time until particle hits 
    the boundaries of the box with its current velocity vector. 
    
    :param box: Box which particle is in. 
    :param particle: Particle.
    :return: Time that particle would take to hit the 
             wall at current velocity. 
    """

    vx = particle.get_vx()
    vy = particle.get_vy()
    vz = particle.get_vz()

    x = particle.get_x()
    y = particle.get_y()
    z = particle.get_z()

    width = box.get_width()
    height = box.get_height()
    depth = box.get_depth()

    # To identify which face, first find angle of velocity
    polar, azimuthal = get_velocity_angles(vx, vy, vz)

    # The premise is that if it hits one of the faces,
    # either the y or the x coordinate will go beyond the
    # box boundary, they will go together only if it hits the
    # corner, which shouldn't be a problem with this approach.

    x_boundary = 0
    y_boundary = 0
    z_boundary = 0
    t_boundary = 0

    t_x, t_y, t_z = 0,0,0

    if 0.0 <= polar < PI / 2:
        t_z = abs((depth - z) / vz)
        z_boundary = depth

    elif PI / 2 <= polar <= PI:
        t_z = abs(z / vz)
        z_boundary = 0
    else:
        print("Error, polar angle incorrect: %f" % polar)
        os._exit(1)

    if 0.0 <= azimuthal <= PI/2:
        t_x = abs((width - x) / vx)
        t_y = abs((height - y) / vy)

        x_boundary = 0
        y_boundary = 0

        t_boundary = min(t_x, t_y, t_z)

        if t_boundary == t_x:
            x_boundary = width
            y_boundary = y + vy * t_x
            z_boundary = z + vz * t_x
        elif t_boundary == t_y:
            x_boundary = x + vx * t_y
            y_boundary = height
            z_boundary = z + vz * t_y
        elif t_boundary == t_z:
            # In this case use the pre assigned value above for z_boundary
            x_boundary = x + vx * t_z
            y_boundary = y + vy * t_z

    elif PI/2 < azimuthal <= PI:
        t_x = abs(x/vx)
        t_y = abs((height - y) / vy)

        x_boundary = 0
        y_boundary = 0

        t_boundary = min(t_x, t_y, t_z)

        if t_boundary == t_x:
            x_boundary = 0
            y_boundary = y + vy * t_x
            z_boundary = z + vz * t_x
        elif t_boundary == t_y:
            x_boundary = x + vx * t_y
            y_boundary = height
            z_boundary = z + vz * t_y
        elif t_boundary == t_z:
            x_boundary = x + vx * t_z
            y_boundary = y + vy * t_z

    elif 3 * PI/2 <= azimuthal <= 2 * PI:
        t_x = abs((height - x) / vx)
        t_y = abs(y / vy)

        x_boundary = 0
        y_boundary = 0

        t_boundary = min(t_x, t_y, t_z)

        if t_boundary == t_x:
            x_boundary = width
            y_boundary = y + vy * t_x
            z_boundary = z + vz * t_x
        elif t_boundary == t_y:
            x_boundary = x + vx * t_y
            y_boundary = 0
            z_boundary = z + vz * t_y
        elif t_boundary == t_z:
            x_boundary = x + vx * t_z
            y_boundary = y + vy * t_z

    elif PI <= azimuthal < 3 * PI / 2:
        t_x = abs(x / vx)
        t_y = abs(y / vy)
        t_boundary = min(t_y, t_x, t_z)

        if t_boundary == t_x:
            x_boundary = 0
            y_boundary = y + vy * t_x
            z_boundary = z + vz * t_x
        elif t_boundary == t_y:
            x_boundary = x + vx * t_y
            y_boundary = 0
            z_boundary = z + vz * t_y
        elif t_boundary == t_z:
            x_boundary = x + vx * t_z
            y_boundary = y + vy * t_z

    return t_boundary, x_boundary, y_boundary, z_boundary


def adjust_boundary_velocity(particle, box, polar_angle, azimuthal_angle):

    curr_vx, curr_vy, curr_vz = particle.get_vx(), particle.get_vy(), particle.get_vz()
    v_in = get_magnitude(curr_vx, curr_vy, curr_vz)

    cos_azimuthal_vel = v_in * np.sin(polar_angle) * np.cos(azimuthal_angle)
    sin_azimuthal_vel = v_in * np.sin(polar_angle) * np.sin(azimuthal_angle)
    polar_vel = abs(v_in * np.cos(polar_angle))

    # Corner cases first: will randomise this properly later because have to
    # include proper angular bounds for angles.

    coordinate = [particle.get_x(), particle.get_y(), particle.get_z()]

    if coordinate in box.get_corners():
        rand_polar = get_polar_angle(1)
        rand_azimuthal = np.random.uniform(0, PI / 2)

        vx = abs(v_in * np.sin(rand_polar) * np.cos(rand_azimuthal))
        vy = abs(v_in * np.sin(rand_polar) * np.sin(rand_azimuthal))
        vz = abs(v_in * np.cos(rand_polar))

        if coordinate[0] == box.get_width():
            vx = -vx

        if coordinate[1] == box.get_height():
            vy = -vy

        if coordinate[2] == box.get_depth():
            vz = -vz

        particle.set_velocity(vx, vy, vz)

    # Now Faces

    # x = 0
    elif particle.get_x() <= 0:
        particle.set_velocity(polar_vel, cos_azimuthal_vel, sin_azimuthal_vel)

    # x = width
    elif particle.get_x() >= box.width:
        particle.set_velocity(-polar_vel, cos_azimuthal_vel, sin_azimuthal_vel)

    # y = 0
    elif particle.get_y() <= 0:
        particle.set_velocity(cos_azimuthal_vel, polar_vel, sin_azimuthal_vel)

    # y = height
    elif particle.get_y() >= box.height:
        particle.set_velocity(cos_azimuthal_vel, -polar_vel, sin_azimuthal_vel)

    # z = 0
    elif particle.get_z() <= 0:
        particle.set_velocity(cos_azimuthal_vel, sin_azimuthal_vel, polar_vel)

    # z = depth
    elif particle.get_z() >= box.depth:
        particle.set_velocity(cos_azimuthal_vel, sin_azimuthal_vel, -polar_vel)

    event_str = particle.get_name() + ": Boundary hit occurred at %s" % particle.get_t() \
                + " at (" + str(particle.get_x()) + ", " + str(particle.get_y()) + ", " + str(particle.get_z()) \
                + ") with new velocity (" + str(particle.get_vx()) + ", " + str(particle.get_vy()) + ", " \
                + str(particle.get_vz()) + ")"
    particle.add_event(event_str)

    print(event_str)

def get_cos_angle():
    """
    Draw from a cos(x) distribution from 0 to pi/2
    :return: Value drawn from cos(x) distribution  
    """
    # CDF of cos(x) from 0 to pi/2 is F = sin(x)
    # so x = arcsin(F) where F in range 0 to 1.

    return np.arcsin(np.random.rand())


def closest_distance_to_box(box, particle):

    x, y, z = particle.get_x(), particle.get_y(), particle.get_z()
    height = box.get_height()
    width = box.get_width()
    depth = box.get_depth()

    top_dist = depth - z
    bottom_dist = z

    right_dist = width - x
    left_dist = x

    front_dist = height - y
    back_dist = y

    return min(top_dist, bottom_dist, right_dist, left_dist, front_dist, back_dist)


def remove_particle(particle, box, points):
    box.remove_particle_from_box(particle)

    x_points = box.get_x_array()
    y_points = box.get_y_array()
    z_points = box.get_z_array()

    data = (x_points, y_points, z_points)
    points._offsets3d = data

    colour_array = get_colour_array(box.colours.values())
    points._facecolor3d = colour_array
    points._edgecolor3d = colour_array
