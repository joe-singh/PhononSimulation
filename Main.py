"""
Phonon Simulation

Author: Jyotirmai (Joe) Singh 26/6/18
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d
from AnharmonicDecay import *
from IsotopicScattering import *
from BoundaryInteractions import *
from Box import Box

# Max characteristic phonon frequency. Set at 10 THz
MAX_FREQ = 5e12

PI = np.pi


def simulate_step(frames, box, points, colours, title):
    """
    The simulation step. At each step, we choose a random particle 
       and calculate the timestep required for all the processes: 
       
       1. Isotopic scattering
       2. Anharmonic Decay (L->L+T and L->T+T) if we pick a 
          longitudinal phonon
       
       In addition, we calculate another time, which is the time 
       required for the particle to hit the boundary with its current
       velocity vector. Whichever of these times is smallest is the 
       process that happens next, and we advance global time by the 
       minimum time. This way, we increase efficiency by using 
       Discrete Event Selection (DES) and not discrete time steps. 
       
       After picking which process will happen, it is simulated and the 
       animation display is updated. The title is also updated with the
       new global time.
    
    :param frames: Argument used by matplotlib. Current frame number.
    :param box: The box object in which we are simulating our particles. 
    :param points: The array of points used to tell matplotlib about particle locations.
    :param colours: The array controlling what colour each particle is. 
    :param title: The title object of the output plot, used to update title with time.
    :return: None
    """

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

    r = np.random.random()
    t_isotopic = -np.log(r) / isotopic_rate
    t_anharmonic_LLT = -np.log(r) / anharmonic_LLT_rate
    t_anharmonic_LTT = -np.log(r) / anharmonic_LTT_rate

    t_boundary = boundary_info[0]
    x_boundary = boundary_info[1]
    y_boundary = boundary_info[2]
    z_boundary = boundary_info[3]

    smallest_time = min(t_boundary, t_isotopic, t_anharmonic_LTT, t_anharmonic_LLT)
    print("t_boundary: %f, t_anharmonic_LTT: %f, t_anharmonic_LLT: %f, t_isotopic: %f" %
          (t_boundary, t_anharmonic_LTT, t_anharmonic_LLT, t_isotopic))

    if smallest_time == t_isotopic:
        print("ISOTOPIC")
        phonon_isotope_scatter(particle, t_isotopic, box, points, colours)

    elif smallest_time == t_anharmonic_LLT:
        print("BULK ANHARMONIC DECAY LLT")
        anharmonic_decay_LLT(particle, box, t_anharmonic_LLT, points, colours)

    elif smallest_time == t_anharmonic_LTT:
        print("BULK ANHARMONIC DECAY LTT")
        anharmonic_decay_LTT(particle, box, t_anharmonic_LTT, points, colours)
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
        # the velocity vector according to the appropriate process. Simulate surface effects.
        boundary_interaction(particle, box, points, colours)

    # Update position.

    x_points = box.get_x_array()
    y_points = box.get_y_array()
    z_points = box.get_z_array()

    Delta_V = get_magnitude(particle.get_vx(), particle.get_vy(), particle.get_vz()) \
              - box.get_material().get_particle_velocity(particle.get_type())

    if abs(Delta_V) > 1e-6:
        print("Velocities not being conserved properly! Delta_V: %f" % Delta_V)
        print("Original particle type is: %f" % particle.get_type())
        os._exit(1)

    # Final check to make sure no particle has jumped outside.
    if not check_no_particle_outside_box(box):
        os._exit(1)

    data = (x_points, y_points, z_points)

    points._offsets3d = data
    title.set_text('Phonon Simulation: Time={0:.4f} ns'.format(particle.get_t() * 10e9))
    return


def run(num_particles, box_width, box_height, box_depth, num_steps=4000):
    """
    The main method that starts the whole process. Initialises box with 
    all phonons and creates matplotlib animation.
    
    :param num_particles: The number of initial phonons.
    :param box_width: The width of the box (max x-coordinate)
    :param box_height: The height of the box (max y-coordinate)
    :param box_depth: The depth of the box (max z-coordinate)
    :param num_steps: The number of steps for the animation. 
    :return: None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    particle_array = []
    colour_dict = {}

    material = Silicon

    # Generating all initial particles.
    for i in range(num_particles):
        random_x = np.random.uniform(0, box_width)
        random_y = np.random.uniform(0, box_height)
        random_z = np.random.uniform(0, box_depth)
        # Map particle indices to colours

        # Initial distribution of phonons governed by the distribution in
        # http://cdms.berkeley.edu/Dissertations/mpyle.pdf page 182.
        # 54.1% Slow Transverse, 36.3% Fast Transverse, 9.6% Longitudinal
        rand_type = (np.random.choice(3, 1, p=[0.541, 0.363, 0.096]) + 1)[0]
        print(rand_type)
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
                                  interval=100)

    plt.grid()
    plt.show()

run(100, 1e-7, 1e-7, 1e-7, 4000)
