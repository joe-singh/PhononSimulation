import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d
from AnharmonicDecay import *
from IsotopicScattering import *

# Parameter to set range on random velocity generation
VELOCITY_MAX = 10000

# Max characteristic phonon frequency. Set at 10 THz
MAX_FREQ = 1e12

# Probability of colour change
COLOUR_CHANGE_RATE = 1e-10

# Probability of split
SPLIT_RATE = 100

PI = np.pi

corners = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0],
           [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]

def simulate_step(frames, box, points, colours, title):

    # Pick random particle from box
    particle_index = np.random.randint(0, box.get_num_particles())
    particle = box.get_particle(particle_index)

    particle.get_info()

    # Store current positions
    curr_x, curr_y, curr_z = particle.get_x(), particle.get_y(), particle.get_z()
    curr_vx, curr_vy, curr_vz = particle.get_vx(), particle.get_vy(), particle.get_vz()
    curr_t, curr_f = particle.get_t(), particle.get_f()

    boundary_info = time_to_boundary(box, particle)
    anharmonic_rate = 1e5  # get_anharmonic_rate(particle)
    if not ((colour_dictionary[particle.type] == L).all()):
        anharmonic_rate = 1e-10
    isotopic_rate = 1e-10 # isotopic_scatter_rate(particle)

    t_isotopic = np.log(1/np.random.random()) / isotopic_rate
    t_split = np.log(1 /np.random.random()) / SPLIT_RATE
    t_change_type = np.log(1 /np.random.random()) / COLOUR_CHANGE_RATE
    t_anharmonic = np.log(1/np.random.random()) / anharmonic_rate

    t_boundary = boundary_info[0]
    x_boundary = boundary_info[1]
    y_boundary = boundary_info[2]
    z_boundary = boundary_info[3]

    smallest_time = min(t_split, t_change_type, t_boundary, t_isotopic, t_anharmonic)
    print("t_boundary: %f, t_anharmonic: %f, t_isotopic: %f" % (t_boundary, t_anharmonic, t_isotopic))

    if smallest_time == t_change_type:
        print("CHANGE_TYPE")
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
        particle.advance(smallest_time)

        colour_array = get_colour_array(colours.values())

        points._facecolor3d = colour_array
        points._edgecolor3d = colour_array

        title.set_text('Phonon Simulation: time={0:.8f}'.format(particle.get_t()))

    elif smallest_time == t_isotopic:
        print("ISOTOPIC")
        phonon_isotope_scatter(particle, t_isotopic, title, box)

    elif smallest_time == t_anharmonic:
        print("ANHARMONIC DECAY")
        anharmonic_decay_LLT(particle, box, t_anharmonic, points, colours, title)

    else:
        print("BOUNDARY")
        # Otherwise begin propagation to boundary #
        # Advance time
        new_t = curr_t + smallest_time
        box.update_time(new_t)

        particle.set_x(x_boundary)
        particle.set_y(y_boundary)
        particle.set_z(z_boundary)
        # Check if the new position is at the boundary or beyond. If it is change
        # the velocity vector to make it reflect.
        adjust_boundary_velocity(particle, box, title)

        # Now with the appropriate position and velocity, just propagate the particle.

    x_points = box.get_x_array()
    y_points = box.get_y_array()
    z_points = box.get_z_array()

    Delta_V = get_magnitude(particle.get_vx(), particle.get_vy(), particle.get_vz()) \
              - velocity_dictionary[particle.get_type()]

    if abs(Delta_V) > 1e-6:
        print("Velocities not being conserved properly! Delta_V: %f" % Delta_V)
        os._exit(1)

    # Final check to make sure no particle has jumped outside.
    if not check_no_particle_outside_box(box):
        os._exit(1)

    data = (x_points, y_points, z_points)

    points._offsets3d = data
    title.set_text('Phonon Simulation: Time={0:.8f}'.format(particle.get_t()))
    return


def run(num_particles, box_width, box_height, box_depth, num_steps):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    particle_array = []
    colour_dict = {}

    # Generating all initial particles.
    for i in range(num_particles):
        random_x = np.random.uniform(0, box_width)
        random_y = np.random.uniform(0, box_height)
        random_z = np.random.uniform(0, box_depth)
        # Map particle indices to colours
        rand_type = np.random.randint(1, 4)
        colour_dict[i] = rand_type

        # Ensures that phonons are generated with the appropriate velocity
        # based on type. This velocity magnitude is fixed but the direction is
        # randomised.
        velocity = velocity_dictionary[rand_type]

        random_vx, random_vy, random_vz = create_random_spherical_vel(velocity)
        random_freq = np.random.uniform(0, MAX_FREQ)

        ptcle = Particle(random_x, random_y, random_z, random_vx, random_vy, random_vz,
                         "Particle " + str(i), rand_type, random_freq)
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

    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))

    title = ax.set_title('3D Test')
    ani = animation.FuncAnimation(fig, simulate_step, frames=np.arange(0, num_steps),
                                  fargs=(box, points, colour_dict, title),
                                  interval=200)

    plt.grid()
    plt.show()

run(1, 1e-3, 1e-3, 1e-3, 4000)
