import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Particle import Particle
import mpl_toolkits.mplot3d.axes3d as p3
from Box import Box
import os

# Parameter to set range on random velocity generation
VELOCITY_MAX = 2000
# Probability of interaction
REACTION_PROB = 0.2

# Probability of colour change given reaction
COLOUR_CHANGE_RATE = 100

# Probability of split given reaction
SPLIT_RATE = 100

PI = np.pi

colour_dictionary = {
    1: np.array([0.0,0.0,1.0,1.0]),
    2: np.array([0.0,1.0,0.0,1.0]),
    3: np.array([1.0,0.0,0.0,1.0])
}


def get_magnitude(vx, vy):
    return (vx ** 2 + vy ** 2)**.5


def get_colour_array(num_array):
    return np.array([colour_dictionary[num] for num in num_array])

def beyond_boundary(particle, box):

    x = particle.get_x()
    y = particle.get_y()
    width = box.get_width()
    height = box.get_height()

    return x >= width or x <= 0 or y >= height or y <= 0

def check_no_particle_outside_box(box):
    width = box.get_width()
    height = box.get_height()
    for particle in box.particles:
        x = particle.get_x()
        y = particle.get_y()
        if x > width or x < 0 or y > height or y < 0:
            particle.get_info()
            print(get_velocity_angle(particle.vx, particle.vy))
            return False
    return True


def get_velocity_angle(vx, vy):
    if vx >= 0:
        return np.arctan(vy/vx)
    elif vx < 0 and vy >= 0:
        return PI + np.arctan(vy/vx)
    elif vx < 0 and vy < 0:
        return -PI + np.arctan(vy/vx)


def time_to_boundary(box, particle):

    vx = particle.get_vx()
    vy = particle.get_vy()
    x = particle.get_x()
    y = particle.get_y()
    width = box.get_width()
    height = box.get_height()

    # To identify which face, first find angle of velocity
    angle = get_velocity_angle(vx, vy)

    # The premise is that if it hits one of the faces,
    # either the y or the x coordinate will go beyond the
    # box boundary, they will go together only if it hits the
    # corner, which shouldn't be a problem with this approach.

    x_boundary = 0
    y_boundary = 0
    t_boundary = 0

    if 0.0 <= angle <= PI/2:
        # Test top and right face.
        t_x = abs((width - x) / vx)
        t_y = abs((height - y) / vy)
        t_boundary = min(t_y, t_x)

        if t_x <= t_y:
            # Impact with right face
            x_boundary = width
            y_boundary = y + vy * t_x

        else:
            # Impact with top face
            x_boundary = x + vx * t_y
            y_boundary = height

    elif PI/2 < angle <= PI:
        t_x = abs(x/vx)
        t_y = abs((height - y) / vy)
        t_boundary = min(t_y, t_x)

        if t_x <= t_y:
            # Impact with left face
            x_boundary = 0
            y_boundary = y + vy * t_x

        else:
            # Top face
            x_boundary = x + vx * t_y
            y_boundary = height

    elif -PI/2 <= angle < 0.0:
        t_x = abs((height - x) / vx)
        t_y = abs(y / vy)
        t_boundary = min(t_y, t_x)

        if t_x <= t_y:
            #  Right face
            x_boundary = width
            y_boundary = y + vy * t_x

        else:
            # Bottom face
            x_boundary = x + vx * t_y
            y_boundary = 0

    elif -PI <= angle < -PI/2:
        t_x = abs(x / vx)
        t_y = abs(y / vy)
        t_boundary = min(t_y, t_x)

        if t_x <= t_y:
            # Left face
            x_boundary = 0
            y_boundary = y + vy * t_x

        else:
            # Bottom face
            x_boundary = x + vx * t_y
            y_boundary = 0

    return t_boundary, x_boundary, y_boundary


def at_corner(particle, box):

    top_right = particle.get_x() == box.width and particle.get_y() == box.height
    bottom_right = particle.get_x() == box.width and particle.get_y() == 0
    top_left = particle.get_x() == 0 and particle.get_y() == box.height
    bottom_left = particle.get_x() == 0 and particle.get_y() == 0

    return top_left or top_right or bottom_left or bottom_right

"""Simulation step, Delta_t corresponds to the time between snapshots."""


def simulate_step(frames, box, points, colours, title):

    # Pick random particle from box
    particle_index = np.random.randint(0, box.get_num_particles())
    particle = box.get_particle(particle_index)

    # Store current positions
    curr_x, curr_y, curr_z = particle.get_x(), particle.get_y(), particle.get_z()
    curr_vx, curr_vy = particle.get_vx(), particle.get_vy()
    curr_t = particle.get_t()

    boundary_info = time_to_boundary(box, particle)
    t_split = np.log(1 / np.random.random()) / SPLIT_RATE
    t_change_type = np.log(1 / np.random.random()) / COLOUR_CHANGE_RATE
    t_boundary = boundary_info[0]
    x_boundary = boundary_info[1]
    y_boundary = boundary_info[2]


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
                    + " at (" + str(particle.get_x()) + ", " + str(particle.get_y()) + ")"

        particle.add_event(event_str)
        print(event_str)

        colour_array = get_colour_array(colours.values())

        points._facecolor3d = colour_array
        points._edgecolor3d = colour_array

        title.set_text('Phonon Simulation: time={}'.format(particle.get_t()))

    elif smallest_time == t_split:
        # Advance time
        new_t = curr_t + t_split
        box.update_time(new_t)

        # Initiate split. First create new particle. We will choose a random output
        # for the particles for now, can integrate real physics (i.e. momentum consv) later.
        random_vx = np.random.uniform(-VELOCITY_MAX, VELOCITY_MAX)
        random_vy = np.random.uniform(-VELOCITY_MAX, VELOCITY_MAX)
        new_type = np.random.randint(1, 4)
        new_particle = Particle(curr_x, curr_y, curr_z, random_vx, random_vy,
                                "Particle " + str(box.get_num_particles()),
                                new_type, t=particle.get_t())
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
        title.set_text('Phonon Simulation: time={}'.format(particle.get_t()))
        #points.scatter(x_points, y_points, z_points)

    else:
        # Otherwise begin propagation. #
        # Advance time
        new_t = curr_t + smallest_time
        box.update_time(new_t)

        new_x = x_boundary
        new_y = y_boundary
        particle.set_x(new_x)
        particle.set_y(new_y)

        # Check if the new position is at the boundary or beyond. If it is change
        # the velocity vector to make it reflect.
        if beyond_boundary(particle, box):

            # Add event time to particle register - still need to register a 'collision'
            particle.add_event(new_t)
            v_in = get_magnitude(curr_vx, curr_vy)

            # Generate random angle to randomise bounce. Define two angles for convenience
            rand_angle_side = np.random.uniform(-PI / 2, PI / 2)
            rand_angle_top = np.random.uniform(0, PI)

            # Corner cases first:

            if particle.get_x() == particle.get_y() or particle.get_x() == -particle.get_y():
                particle.set_vx(-curr_vx)
                particle.set_vy(-curr_vy)

            # Now Faces

            # Left
            elif particle.get_x() <= 0:
                particle.set_vx(v_in * np.cos(rand_angle_side))
                particle.set_vy(v_in * np.sin(rand_angle_side))
            # Right
            elif particle.get_x() >= box.width:
                particle.set_vx(-v_in * np.cos(rand_angle_side))
                particle.set_vy(v_in * np.sin(rand_angle_side))
            # Bottom
            elif particle.get_y() <= 0:
                particle.set_vx(v_in * np.cos(rand_angle_top))
                particle.set_vy(v_in * np.sin(rand_angle_top))
            # Top
            elif particle.get_y() >= box.height:
                particle.set_vx(v_in * np.cos(rand_angle_top))
                particle.set_vy(-v_in * np.sin(rand_angle_top))

            event_str = particle.get_name() + ": Boundary hit occurred at %s" % particle.get_t() \
                        + " at (" + str(particle.get_x()) + ", " + str(particle.get_y()) + ", " + str(particle.get_z()) \
                        + ") with new velocity (" + str(particle.get_vx()) + ", " + str(particle.get_vy()) + ")"
            particle.add_event(event_str)

            print(event_str)

        # If neither of the if cases is activated, the position should just continue straight.

        x_points = box.get_x_array()
        y_points = box.get_y_array()
        z_points = box.get_z_array()

        # Final check to make sure no particle has jumped outside.
        if not check_no_particle_outside_box(box):
            os._exit(1)
        # data = np.array([[x_points[i], y_points[i], z_points[i]] for i in range(len(x_points))])
        data = (x_points, y_points, z_points)
        # points.scatter(x_points, y_points, z_points)
        points._offsets3d = data
        title.set_text('Phonon Simulation: Time={}'.format(particle.get_t()))
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
        ptcle = Particle(random_x, random_y, random_z, random_vx, random_vy, "Particle " + str(i), rand_colour)
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

run(1, 1, 1, 1, 4000)
