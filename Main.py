import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d
from AnharmonicDecay import *
from IsotopicScattering import *
from Box import Box

# Max characteristic phonon frequency. Set at 10 THz
MAX_FREQ = 5e12

PI = np.pi

def simulate_step(frames, box, points, colours, title):


    # Pick random particle from box
    particle_index = np.random.randint(0, box.get_num_particles())
    particle = box.get_particle(particle_index)

    particle.get_info()

    # Store current information
    curr_t, curr_f = particle.get_t(), particle.get_f()

    boundary_info = time_to_boundary(box, particle)
    anharmonic_LTT_rate = get_anharmonic_rate(box, particle, 0)
    anharmonic_LLT_rate = get_anharmonic_rate(box, particle, 1)
    isotopic_rate = isotopic_scatter_rate(box, particle)

    t_isotopic = np.log(1/np.random.random()) / isotopic_rate
    t_anharmonic_LLT = np.log(1/np.random.random()) / anharmonic_LLT_rate
    t_anharmonic_LTT = np.log(1/np.random.random()) / anharmonic_LTT_rate

    t_boundary = boundary_info[0]
    x_boundary = boundary_info[1]
    y_boundary = boundary_info[2]
    z_boundary = boundary_info[3]

    smallest_time = min(t_boundary, t_isotopic, t_anharmonic_LTT, t_anharmonic_LLT)
    print("t_boundary: %f, t_anharmonic_LTT: %f, t_anharmonic_LLT: %f, t_isotopic: %f" %
          (t_boundary, t_anharmonic_LTT, t_anharmonic_LLT, t_isotopic))

    if smallest_time == t_isotopic:
        print("ISOTOPIC")
        phonon_isotope_scatter(particle, t_isotopic, title, box, points)

    elif smallest_time == t_anharmonic_LLT:
        print("ANHARMONIC DECAY LLT")
        anharmonic_decay_LLT(particle, box, t_anharmonic_LLT, points, colours, title)

    elif smallest_time == t_anharmonic_LTT:
        print("ANHARMONIC DECAY LTT")
        anharmonic_decay_LTT(particle, box, t_anharmonic_LTT, points, colours, title)
    else:
        print("BOUNDARY")
        # Otherwise begin propagation to boundary
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
              - box.get_material().get_particle_velocity(particle.get_type())

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

    material = Germanium

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
        velocity = material.get_particle_velocity(rand_type)

        random_vx, random_vy, random_vz = create_random_spherical_vel(velocity)
        random_freq = np.random.uniform(MAX_FREQ/2.0, 3 * MAX_FREQ/2.0)

        ptcle = Particle(random_x, random_y, random_z, random_vx, random_vy, random_vz,
                         "Particle " + str(i), rand_type, random_freq)
        particle_array.append(ptcle)

    # Box with initial starting configuration. Material parameters defined in UtilityMethods.py
    box = Box(material, box_width, box_height, box_depth, particle_array, colour_dict)

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
                                  interval=500)

    plt.grid()
    plt.show()

run(10, 1e-7, 1e-7, 1e-7, 4000)
