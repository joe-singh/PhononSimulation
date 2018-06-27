"""
Particle class to simulate phonon particles. 

Author: Jyotirmai (Joe) Singh 26/6/18
"""
import numpy as np
import os
PI = np.pi


def calculate_k_vector(vx, vy, vz, w):
    """
    Calculate a k vector given a velocity vector,
    following the dispersion relation w = vk
    
    :param vx: x velocity 
    :param vy: y velocity
    :param vz: z velocity
    :param w: angular frequency
    :return: k vector, calculated by multiplying w / |v| by 
             the unit vector in the direction defined by (vx, vy, vz)
    """
    v = np.array([vx, vy, vz])
    v_mag = np.linalg.norm(v)

    if v_mag == 0:
        # No wave vector if no phonon velocity. Not
        # physical just a lame trick to simulate
        # decaying into two particles.
        return np.array([0.0, 0.0, 0.0])

    v_unit = v / v_mag
    return (w / v_mag) * v_unit


class Particle:
    """Particle class"""

    def __init__(self, x, y, z, vx, vy, vz, name, type, frequency, t=0, event_times=[]):
        """
        Initialiser method.
        
        :param x: Initial x coordinate.
        :param y: Initial y coordinate.
        :param z: Initial z coordinate.
        :param vx: Initial x velocity.
        :param vy: Initital y velocity.
        :param vz: Initial z velocity.
        :param name: Phonon name.
        :param type: Phonon type - 1 = Slow Transverse, 2 = Fast Transverse, 3 = Longitudinal. 
        :param frequency: Phonon frequency.
        :param t: Current time. 
        :param event_times: List containing records of important events that happen to this phonon,
                            such as border hits, isotopic scatters, and anharmonic decays. Stored as 
                            strings.
        """
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.vz = vz
        # Tracking times of events, will have the latest time
        self.t = t
        self.event_times = event_times
        self.name = name
        self.type = type
        self.z = z
        self.freq = frequency
        self.w = 2 * PI * self.freq
        self.k = calculate_k_vector(vx, vy, vz, self.w)

    def get_name(self):
        """
        :return: The phonon's name. 
        """
        return self.name

    def get_x(self):
        """
        :return: x coordinate.
        """
        return self.x

    def get_y(self):
        """
        :return: y coordinate.
        """
        return self.y

    def get_z(self):
        """
        :return: z coordinate.
        """
        return self.z

    def get_v(self):
        """
        :return: The magnitude of the particle's velocity.
        """
        return (self.vx ** 2 + self.vy ** 2 + self.vz ** 2) ** .5

    def get_vx(self):
        """
        :return: x velocity. 
        """
        return self.vx

    def get_vy(self):
        """
        :return: y velocity. 
        """
        return self.vy

    def get_vz(self):
        """
        :return: z velocity. 
        """
        return self.vz

    def get_t(self):
        """
        :return: Current particle time. 
        """
        return self.t

    def get_f(self):
        """
        :return: Phonon frequency. 
        """
        return self.freq

    def get_k(self):
        """
        :return: Phonon k-vector. 
        """
        return self.k

    def get_kx(self):
        """
        :return: x component of k-vector. 
        """
        return self.k[0]

    def get_ky(self):
        """
        :return: y component of k-vector. 
        """
        return self.k[1]

    def get_kz(self):
        """
        :return: z component of k-vector. 
        """
        return self.k[2]

    def get_w(self):
        """
        :return: Phonon angular frequency, 2 * PI * f 
        """
        return self.w

    def set_w(self, w):
        """
        Sets phonon angular frequency, and adjusts normal
        frequency accordingly.
        
        :param w: New angular frequency. 
        :return: None
        """
        self.w = w
        self.freq = w / (2 * PI)

    def set_k(self, k):
        """
        Sets k-vector. 
        
        :param k: New k-vector. 
        """
        self.k = k

    def set_x(self, x):
        """
        Set x coordinate.
        
        :param x: New x coordinate. 
        """
        self.x = x

    def set_y(self, y):
        """
        Set y coordinate.
        
        :param y: New y coordinate. 
        """
        self.y = y

    def set_z(self, z):
        """
        Set z coordinate.
        
        :param z: New z coordinate. 
        """
        self.z = z

    def set_vx(self, vx):
        """
        Set x velocity.
        
        :param vx: New x velocity.
        """
        self.vx = vx

    def set_vy(self, vy):
        """
        Set y velocity.
        
        :param vy: New x velocity.
        """
        self.vy = vy

    def set_vz(self, vz):
        """
        Set z velocity.
        
        :param vz: New z velocity.
        """
        self.vz = vz

    def set_velocity(self, vx, vy, vz):
        """
        Set new velocity. 
        
        :param vx: New x velocity. 
        :param vy: New y velocity.
        :param vz: New z velocity.
        """
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.k = calculate_k_vector(vx, vy, vz, self.freq)

    def set_k(self, kx, ky, kz):
        """
        Set new k-vector. 

        :param vx: New k-vector x component. 
        :param vy: New k-vector y component.
        :param vz: New k-vector z component.
        """
        self.k[0] = kx
        self.k[1] = ky
        self.k[2] = kz

    def calculate_new_k(self):
        """
        Recalculates k-vector based on current velolcity and 
        angular frequency.
        """
        self.k = calculate_k_vector(self.vx, self.vy, self.vz, self.w)

    def set_t(self, t):
        """
        Set current phonon time. 
        
        :param t: Current time.  
        """
        self.t = t

    def set_f(self, f):
        """
        Sets the new phonon frequency and 
        adjusts the angular frequency accordingly. 
        
        :param f: New frequency.
        """
        self.freq = f
        self.w = 2 * PI * f

    def get_type(self):
        """
        :return: The phonon type, as specified aboce in the init method.  
        """
        return self.type

    def set_type(self, type):
        """
        Set new phonon type. 
        
        :param type: New type, integer between 1 and 3 as 
                     documented in init method. 
        """
        self.type = type

    def add_event(self, event):
        """
        Add event information to particle log. 
        
        :param event: String containing event information. 
        """
        self.event_times.append(event)

    def get_info(self):
        """
        Prints a readout of the particle's current state, giving position and
        velocity coordinates together with frequency. 
        """
        print(self.name + "at (" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"
              + " with speed: (" + str(self.vx) + ", " + str(self.vy) + ", " +str(self.vz) +
              ") and frequency %f" % self.freq)

    def advance(self, t):
        """
        Moves a particle forward in time for a time t 
        at its current velocity. 
        
        :param t: Time for which particle will advance. 
        :return: The new x, y, and z coordinates. 
        """
        self.x = self.x + t * self.vx
        self.y = self.y + t * self.vy
        self.z = self.z + t * self.vz
        return self.x, self.y, self.z


