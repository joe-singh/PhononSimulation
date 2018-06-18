"""File containing methods to simulate anharmonic decay."""

from Particle import Particle
from Box import Box
import numpy as np
from UtilityMethods import *

# TODO: PUT THE SCARY INTEGRAL STUFF IN HERE LATER.
def get_anharmonic_rate(particle):

    # Make the anharmonic rate 0 if this is not a longitudinal phonon
    if particle.get_type() != 3:
        return 0.0

    return 1.62e-54 * (particle.get_f() ** 5)

def anharmonic_decay(particle, box, t, points, colours, title):

    # Firstly advance time
    box.update_time(particle.get_t() + t)

    random_type = np.random.randint(1, 4)
    curr_x, curr_y, curr_z = particle.get_x(), particle.get_y(), particle.get_z()

    # Create new particle with 0 momentum/frequency. We will give it
    # correct frequency in a bit.
    new_particle = Particle(curr_x, curr_y, curr_z, 0, 0, 0,
                            "Particle " + str(box.get_num_particles()),
                            random_type, 0, t=particle.get_t())

    # Give new and old particle correct properties. Using old particle
    # with 0 velocity/wavevector/frequency means this process is effectively
    # like breaking one particle into two.
    particle.simulate_momentum_conservation(new_particle)

    # Put new particle in.
    box.add_particle(new_particle)

    # Propagate both particles.
    propagate(particle, box, t, title)
    propagate(new_particle, box, t, title)

    old_position = np.array([curr_x, curr_y, curr_z])
    new_position = np.array([particle.get_x(), particle.get_y(), particle.get_z()])

    print("Time: %f" % t)
    print("Delta = " + str(new_position - old_position))

    # Update particle creation on display.
    colours[box.get_num_particles() - 1] = new_particle.get_type()

    colour_array = get_colour_array(colours.values())
    points._facecolor3d = colour_array
    points._edgecolor3d = colour_array

    x_points = box.get_x_array()
    y_points = box.get_y_array()
    z_points = box.get_z_array()

    data = (x_points, y_points, z_points)

    event_str = particle.get_name() + ": Interaction Event occurred at %s" % particle.get_t() \
                + ".  " + particle.get_name() + " splits to produce " + new_particle.get_name() \
                + " at (" + str(particle.get_x()) + ", " + str(particle.get_y()) + ") with velocity (" \
                + str(particle.get_vx()) + ", " + str(particle.get_vy()) + ")."

    particle.add_event(event_str)
    new_particle.add_event(event_str)
    print(event_str)

    points._offsets3d = data
    title.set_text('Phonon Simulation: time={0:.8f}'.format(particle.get_t()))

    return