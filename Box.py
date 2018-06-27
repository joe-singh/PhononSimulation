"""
Box class.

Author: Jyotirmai (Joe) Singh 26/6/18
"""

import numpy as np
import re

class Box:

    def __init__(self, material, width, height, depth, particles=[], colours={}):
        """
        Initialiser method. 
        
        :param material: The material from which the box is constructed. 
        :param width: The width of the box (max x) 
        :param height: The height of the box (max y) 
        :param depth: The depth of the box (max z)
        :param particles: A list of all the phonons contained in the box. 
        :param colours: A dictionary mapping phonon types to colours. 
        """
        self.material = material
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
        """
        Get particle number i.
        
        :param i: Particle index. 
        :return: ith particle.
        """
        try:
            return self.particles[i]
        except IndexError:
            print("Incorrect index, check number is between 0 and n-1, n = number of particles.")

    def get_width(self):
        """
        :return: Box width.
        """
        return self.width

    def get_height(self):
        """
        :return: Box height.
        """
        return self.height

    def get_depth(self):
        """
        :return: Box depth.  
        """
        return self.depth

    def get_x_array(self):
        """
        :return: An array where the ith entry is the x coordinate of the ith particle.  
        """
        return np.array([particle.get_x() for particle in self.particles])

    def get_y_array(self):
        """
        :return: An array where the ith entry is the y coordinate of the ith particle.  
        """
        return np.array([particle.get_y() for particle in self.particles])

    def get_z_array(self):
        """
        :return: An array where the ith entry is the z coordinate of the ith particle.  
        """
        return np.array([particle.get_z() for particle in self.particles])

    def get_corners(self):
        """
        :return: A list of the box's corners. 
        """
        return self.corners

    def get_num_particles(self):
        """
        :return: The number of particles in the box. 
        """
        return len(self.particles)

    def get_particle_no(self, string):
        """
        Returns a particle number given its name string. 
        Assumes a format which ends with the designating number
        at the end. The designating number is the index of the 
        particle in the particle list. 
        
        :param string: The particle name.
        :return: The particle at the position of the number from 
                 the end of the string. 
        """
        m = re.search(r'\d+$', string)
        return int(m.group()) if m else None

    def add_particle(self, particle):
        """
        Adds particle to box. 
        
        :param particle: Particle to be added.
        """
        self.particles.append(particle)

    def update_time(self, time):
        """
        Update box time. Simultaneously updates 
        for all particles to maintain consistency. 
        
        :param time: Time to update particles with. 
        """
        for particle in self.particles:
            particle.set_t(time)

    def get_material(self):
        """
        Get material of box. 
        """
        return self.material
