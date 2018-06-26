import numpy as np
import re

class Box:

    def __init__(self, height, width, depth, particles=[], colours={}):
        self.height = height
        self.width = width
        self.particles = particles
        self.colours = colours
        self.depth = depth
        self.corners = [[0, 0, 0], [0, self.height, 0],
                        [self.width, 0, 0], [self.width, self.height, 0],
                        [0, 0, self.depth], [0, self.height, self.depth],
                        [self.width, 0, self.depth], [self.width, self.height, self.depth]]

    def get_particle(self, i):
        try:
            return self.particles[i]
        except IndexError:
            print("Incorrect index, check number is between 0 and n-1, n = number of particles.")

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_depth(self):
        return self.depth

    def get_x_array(self):
        return np.array([particle.get_x() for particle in self.particles])

    def get_y_array(self):
        return np.array([particle.get_y() for particle in self.particles])

    def get_z_array(self):
        return np.array([particle.get_z() for particle in self.particles])

    def get_corners(self):
        return self.corners

    def get_num_particles(self):
        return len(self.particles)

    """Assumes that particle is given a name with the number at the end."""
    def get_particle_no(self, string):
        m = re.search(r'\d+$', string)
        return int(m.group()) if m else None

    def add_particle(self, particle):
        self.particles.append(particle)

    def update_time(self, time):
        for particle in self.particles:
            particle.set_t(time)

