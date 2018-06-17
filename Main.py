import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d
from Particle import Particle
from Box import Box
import os

# Parameter to set range on random velocity generation
VELOCITY_MAX = 2000

# Characteristic Colours to represent different phonon types.
# ST - Slow Transverse
# FT - Fast Transverse
# L - Longitudinal

ST = np.array([0.0,0.0,1.0,1.0])
FT = np.array([0.0,1.0,0.0,1.0])
L = np.array([1.0,0.0,0.0,1.0])

# Max characteristic phonon frequency.
MAX_FREQ = 10e12

# Probability of colour change given reaction
COLOUR_CHANGE_RATE = 1e-9

# Probability of split given reaction
SPLIT_RATE = 1000

PI = np.pi

colour_dictionary = {
    1: ST,
    2: FT,
    3: L
}

corners = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0],
           [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]


def get_polar_angle(corner=0):
    # We need to use random numbers from -1 to 1 to get this angle.
    # Because the CDF of sin(x) from 0 to pi is F = 1/2(1-cos(x))
    # so x = arccos(1-2F) where F is from 0 to 1 which means
    # 1 - 2F is from -1 to 1. The adjustment from 0 to 1 gives
    # polar angles of 0 to pi/2, needed for the corners.
    rand = np.random.uniform(-1,1)

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

    return x,y,z

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
    elif vx < 0 and vy >= 0:
        azimuthal = PI + np.arctan(vy/vx)
    elif vx < 0 and vy < 0:
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

    elif PI / 2 <= polar < PI:
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

    elif -PI/2 <= azimuthal < 0.0:
        t_x = abs((height - x) / vx)
        t_y = abs(y / vy)
        t_boundary = min(t_x, t_y, t_z)

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

    elif -PI <= azimuthal < -PI/2:
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

def phonon_isotope_scatter(particle, box, t):
    pass

"""Simulation step, Delta_t corresponds to the time between snapshots."""


def simulate_step(frames, box, points, colours, title):

    # Pick random particle from box
    particle_index = np.random.randint(0, box.get_num_particles())
    particle = box.get_particle(particle_index)

    # Store current positions
    curr_x, curr_y, curr_z = particle.get_x(), particle.get_y(), particle.get_z()
    curr_vx, curr_vy, curr_vz = particle.get_vx(), particle.get_vy(), particle.get_vz()
    curr_t = particle.get_t()

    boundary_info = time_to_boundary(box, particle)
    t_split = np.log(1 / np.random.random()) / SPLIT_RATE
    t_change_type = np.log(1 / np.random.random()) / COLOUR_CHANGE_RATE
    t_boundary = boundary_info[0]
    x_boundary = boundary_info[1]
    y_boundary = boundary_info[2]
    z_boundary = boundary_info[3]

    # Final positions
    new_x, new_y, new_z = 0,0,0

    smallest_time = min(t_split, t_change_type, t_boundary)

    if smallest_time == t_change_type:
        print('Min time = change_type')
        # Simulate split this step. Advance time to time of interaction.
        new_t = curr_t + t_change_type
        box.update_time(new_t)
        ptcle_type = particle.get_type()

        # Cyclically change colour: blue->green, green->red, red->blue
        if ptcle_type == 1:
            particle.set_type(2)
            colours[particle_index] = 2
        elif ptcle_type == 2:
            particle.set_type(3)
            colours[particle_index] = 3
        elif ptcle_type == 3:
            particle.set_type(1)
            colours[particle_index] = 1

        event_str = particle.get_name() + ": Interaction Event occurred at %s" % particle.get_t() \
                    + ". Change from " + str(ptcle_type) + " to " + str(particle.get_type()) \
                    + " at (" + str(particle.get_x()) + ", " + str(particle.get_y()) + str(particle.get_z()) + ")"

        particle.add_event(event_str)
        print(event_str)


        # Advance particle for time of process then change its colour.
        new_z = curr_x + curr_vx * smallest_time
        new_y = curr_y + curr_vy * smallest_time
        new_x = curr_z + curr_vz * smallest_time

        colour_array = get_colour_array(colours.values())

        points._facecolor3d = colour_array
        points._edgecolor3d = colour_array

        title.set_text('Phonon Simulation: time={0:.5f}'.format(particle.get_t()))

    elif smallest_time == t_split:
        # Advance time
        new_t = curr_t + t_split
        box.update_time(new_t)

        # Initiate split. First create new particle. We will choose a random output
        # for the particles for now, can integrate real physics (i.e. momentum consv) later.
        random_vx = np.random.uniform(-VELOCITY_MAX, VELOCITY_MAX)
        random_vy = np.random.uniform(-VELOCITY_MAX, VELOCITY_MAX)
        random_vz = np.random.uniform(-VELOCITY_MAX, VELOCITY_MAX)
        random_frequency = np.random.uniform(0, MAX_FREQ)
        random_type = np.random.randint(1, 4)

        new_z = curr_x + curr_vx * smallest_time
        new_y = curr_y + curr_vy * smallest_time
        new_x = curr_z + curr_vz * smallest_time

        new_particle = Particle(curr_x, curr_y, curr_z, random_vx, random_vy, random_vz,
                                "Particle " + str(box.get_num_particles()),
                                random_type, random_frequency, t=particle.get_t())
        box.add_particle(new_particle)

        # Store information of new particle's creation

        colours[box.get_num_particles()-1] = new_particle.get_type()
        x_points = box.get_x_array()
        y_points = box.get_y_array()
        z_points = box.get_z_array()

        colour_array = get_colour_array(colours.values())
        points._facecolor3d = colour_array
        points._edgecolor3d = colour_array
        # data = np.array([[x_points[i], y_points[i], z_points[i]] for i in range(len(x_points))])
        data = (x_points, y_points, z_points)

        event_str = particle.get_name() + ": Interaction Event occurred at %s" % particle.get_t() \
                    + ".  " + particle.get_name() + " splits to produce " + new_particle.get_name() \
                    + " at (" + str(particle.get_x()) + ", " + str(particle.get_y()) + ") with velocity (" \
                    + str(particle.get_vx()) + ", " + str(particle.get_vy()) + ")."

        particle.add_event(event_str)
        new_particle.add_event(event_str)
        print(event_str)

        points._offsets3d = data
        title.set_text('Phonon Simulation: time={0:.5f}'.format(particle.get_t()))

    else:
        # Otherwise begin propagation. #
        # Advance time
        new_t = curr_t + smallest_time
        box.update_time(new_t)

        new_x = x_boundary
        new_y = y_boundary
        new_z = z_boundary

        # Check if the new position is at the boundary or beyond. If it is change
        # the velocity vector to make it reflect.
        if beyond_boundary(particle, box):

            # Add event time to particle register - still need to register a 'collision'
            particle.add_event(new_t)
            v_in = get_magnitude(curr_vx, curr_vy, curr_vz)

            # Generate random angle to randomise bounce.
            rand_polar = get_polar_angle()
            # Define azimuthal angle
            rand_azimuthal = np.random.uniform(0, 2*PI)

            cos_azimuthal_vel = v_in * np.cos(rand_polar) * np.cos(rand_azimuthal)
            sin_azimuthal_vel = v_in * np.cos(rand_polar) * np.sin(rand_azimuthal)
            polar_vel = v_in * np.sin(rand_polar)

            # Corner cases first: will randomise this properly later because have to
            # include proper angular bounds for angles.

            # TODO: RANDOMISE CORNER ANGLES PROPERLY
            coordinate = [particle.get_x(), particle.get_y(), particle.get_z()]

            if coordinate in corners:
                rand_polar = get_polar_angle(1)
                rand_azimuthal = np.random.uniform(0, PI/2)

                vx = v_in * np.sin(rand_polar) * np.cos(rand_azimuthal)
                vy = v_in * np.sin(rand_polar) * np.sin(rand_azimuthal)
                vz = v_in * np.cos(rand_polar)

                print("Corner case")
                if coordinate == [0,0,0]:
                    pass

                elif coordinate == [0,0,1]:
                    vz = -vz

                elif coordinate == [0,1,0]:
                    vy = -vy

                elif coordinate == [1,0,0]:
                    vx = -vx

                elif coordinate == [1,1,0]:
                    vx = -vx
                    vy = -vy

                elif coordinate == [1,0,1]:
                    vx = -vx
                    vz = -vz

                elif coordinate == [0,1,1]:
                    vy = -vy
                    vz = -vz

                elif coordinate == [1,1,1]:
                    vx = -vx
                    vy = -vy
                    vz = -vz

                particle.set_velocity(vx, vy, vz)

            # Now Faces

            # x = 0
            elif particle.get_x() <= 0:
                particle.set_velocity(polar_vel, cos_azimuthal_vel, sin_azimuthal_vel)

                # particle.set_vx(v_in * np.sin(rand_polar))
                # particle.set_vy(v_in * np.cos(rand_polar) * np.cos(rand_azimuthal))
                # particle.set_vz(v_in * np.cos(rand_polar) * np.sin(rand_azimuthal))

            # x = width
            elif particle.get_x() >= box.width:
                particle.set_velocity(-polar_vel, cos_azimuthal_vel, sin_azimuthal_vel)

                # particle.set_vx(-v_in * np.sin(rand_polar))
                # particle.set_vy(v_in * np.cos(rand_polar) * np.cos(rand_azimuthal))
                # particle.set_vz(v_in * np.cos(rand_polar) * np.sin(rand_azimuthal))

            # y = 0
            elif particle.get_y() <= 0:
                particle.set_velocity(cos_azimuthal_vel, polar_vel, sin_azimuthal_vel)

                # particle.set_vy(v_in * np.sin(rand_polar))
                # particle.set_vx(v_in * np.cos(rand_polar) * np.cos(rand_azimuthal))
                # particle.set_vz(v_in * np.cos(rand_polar) * np.sin(rand_azimuthal))

            # y = height
            elif particle.get_y() >= box.height:
                particle.set_velocity(cos_azimuthal_vel, -polar_vel, sin_azimuthal_vel)

                # particle.set_vy(-v_in * np.sin(rand_polar))
                # particle.set_vx(v_in * np.cos(rand_polar) * np.cos(rand_azimuthal))
                # particle.set_vz(v_in * np.cos(rand_polar) * np.sin(rand_azimuthal))

            # z = 0
            elif particle.get_z() <= 0:
                particle.set_velocity(cos_azimuthal_vel, sin_azimuthal_vel, polar_vel)

                # particle.set_vz(v_in * np.sin(rand_polar))
                # particle.set_vx(v_in * np.cos(rand_polar) * np.cos(rand_azimuthal))
                # particle.set_vy(v_in * np.cos(rand_polar) * np.sin(rand_azimuthal))

            # z = depth
            elif particle.get_z() >= box.depth:
                particle.set_velocity(cos_azimuthal_vel, sin_azimuthal_vel, -polar_vel)

                # particle.set_vz(-v_in * np.sin(rand_polar))
                # particle.set_vx(v_in * np.cos(rand_polar) * np.cos(rand_azimuthal))
                # particle.set_vy(v_in * np.cos(rand_polar) * np.sin(rand_azimuthal))

            event_str = particle.get_name() + ": Boundary hit occurred at %s" % particle.get_t() \
                        + " at (" + str(particle.get_x()) + ", " + str(particle.get_y()) + ", " + str(particle.get_z()) \
                        + ") with new velocity (" + str(particle.get_vx()) + ", " + str(particle.get_vy()) + ")"
            particle.add_event(event_str)

            print(event_str)
            title.set_text('Phonon Simulation: Time={0:.5f}'.format(particle.get_t()))

        # Now with the appropriate position and velocity, just propagate the particle.

    particle.set_x(new_x)
    particle.set_y(new_y)
    particle.set_z(new_z)

    x_points = box.get_x_array()
    y_points = box.get_y_array()
    z_points = box.get_z_array()

    # Final check to make sure no particle has jumped outside.
    if not check_no_particle_outside_box(box):
        os._exit(1)

    data = (x_points, y_points, z_points)

    points._offsets3d = data
    title.set_text('Phonon Simulation: Time={0:.5f}'.format(particle.get_t()))
    return


def run(num_particles, box_width, box_height, box_depth, num_steps):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    particle_array = []
    colour_dict = {}

    for i in range(num_particles):
        random_x = np.random.uniform(0, box_width)
        random_y = np.random.uniform(0, box_height)
        random_z = np.random.uniform(0, box_depth)
        # Map particle indices to colours
        rand_colour = np.random.randint(1, 4)
        colour_dict[i] = rand_colour

        random_vx = np.random.uniform(-VELOCITY_MAX, VELOCITY_MAX)
        random_vy = np.random.uniform(-VELOCITY_MAX, VELOCITY_MAX)
        random_vz = np.random.uniform(-VELOCITY_MAX, VELOCITY_MAX)

        random_freq = np.random.uniform(0, MAX_FREQ)

        ptcle = Particle(random_x, random_y, random_z, random_vx, random_vy, random_vz,
                         "Particle " + str(i), rand_colour, random_freq)
        particle_array.append(ptcle)

    # Box with initial starting configuration
    box = Box(box_width, box_height, box_depth, particle_array, colour_dict)

    points = ax.scatter(box.get_x_array(), box.get_y_array(), box.get_z_array(),
                        facecolors=get_colour_array(colour_dict.values()))
    ax.set_ylim(0, box_height)
    ax.set_xlim(0, box_width)
    ax.set_zlim(0, box_depth)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    title = ax.set_title('3D Test')
    ani = animation.FuncAnimation(fig, simulate_step, frames=np.arange(0, num_steps),
                                  fargs=(box, points, colour_dict, title),
                                  interval=200)

    plt.grid()
    plt.show()

run(2, 1, 1, 1, 4000)
