import numpy as np


class Box:

    def __init__(self, height, width, depth, particles=[], colours={}):
        self.height = height
        self.width = width
        self.particles = particles
        self.colours = colours
        self.depth = depth

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

    def get_num_particles(self):
        return len(self.particles)

    def add_particle(self, particle):
        self.particles.append(particle)

    def update_time(self, time):
        for particle in self.particles:
            particle.set_t(time)

