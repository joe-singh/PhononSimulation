import numpy as np
import os
PI = np.pi

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
    return (vx ** 2 + vy ** 2 + vz ** 2)**.5

def get_colour_array(num_array):
    return np.array([colour_dictionary[num] for num in num_array])


def beyond_boundary(particle, box):

    x = particle.get_x()
    y = particle.get_y()
    z = particle.get_z()

    width = box.get_width()
    height = box.get_height()
    depth = box.get_depth()

    return x >= width or x <= 0 or y >= height or y <= 0 or z <= 0 or z >= depth


def check_no_particle_outside_box(box):
    width = box.get_width()
    height = box.get_height()
    depth = box.get_depth()
    for particle in box.particles:
        x = particle.get_x()
        y = particle.get_y()
        z = particle.get_z()
        if x > width or x < 0 or y > height or y < 0 or z < 0 or z > depth:
            particle.get_info()
            print(get_velocity_angles(particle.vx, particle.vy, particle.vz))
            return False
    return True


def spherical_to_cartesian(r, polar, azimuthal):
    x = r * np.sin(polar) * np.cos(azimuthal)
    y = r * np.sin(polar) * np.sin(azimuthal)
    z = r * np.cos(polar)

    return x, y, z

def create_random_spherical_vel(v_mag):
    rand_polar = get_polar_angle()
    rand_azimuthal = np.random.uniform(0, 2 * PI)
    return spherical_to_cartesian(v_mag, rand_polar, rand_azimuthal)

def get_velocity_angles(vx, vy, vz):

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


def adjust_boundary_velocity(particle, box, title):

    curr_vx, curr_vy, curr_vz = particle.get_vx(), particle.get_vy(), particle.get_vz()
    v_in = get_magnitude(curr_vx, curr_vy, curr_vz)

    # Generate random angle to randomise bounce.
    rand_polar = get_polar_angle()
    # Define azimuthal angle
    rand_azimuthal = np.random.uniform(0, 2 * PI)

    cos_azimuthal_vel = v_in * np.cos(rand_polar) * np.cos(rand_azimuthal)
    sin_azimuthal_vel = v_in * np.cos(rand_polar) * np.sin(rand_azimuthal)
    polar_vel = v_in * np.sin(rand_polar)

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
    title.set_text('Phonon Simulation: Time={0:.8f}'.format(particle.get_t()))


def propagate(particle, box, t, title):

    t_boundary, x_boundary, y_boundary, z_boundary = time_to_boundary(box, particle)

    # If particle propagates for time t and goes beyond the boundary, stop it at the
    # boundary and change its velocity to simulate bouncing off the wall
    if t_boundary <= t:
        particle.set_x(x_boundary)
        particle.set_y(y_boundary)
        particle.set_z(z_boundary)
        adjust_boundary_velocity(particle, box, title)

    # Otherwise can safely propagate for time t without hitting the wall.
    else:
        print("In propagation non boundary case.")
        particle.advance(t)
        title.set_text('Phonon Simulation: time={0:.8f}'.format(particle.get_t()))