"""Particle File"""
import numpy as np
import os
PI = np.pi

def calculate_k_vector(vx, vy, vz, w):
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

    def __init__(self, x, y, z, vx, vy, vz, name, type, frequency, t=0, event_times=[]):
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
        return self.name

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

    def get_v(self):
        return (self.vx ** 2 + self.vy ** 2 + self.vz ** 2) ** .5

    def get_vx(self):
        return self.vx

    def get_vy(self):
        return self.vy

    def get_vz(self):
        return self.vz

    def get_t(self):
        return self.t

    def get_f(self):
        return self.freq

    def get_k(self):
        return self.k

    def get_kx(self):
        return self.k[0]

    def get_ky(self):
        return self.k[1]

    def get_kz(self):
        return self.k[2]

    def get_w(self):
        return self.w

    def set_w(self, w):
        self.w = w
        self.freq = w / (2 * PI)

    def set_k(self, k):
        self.k = k

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_z(self, z):
        self.z = z

    def set_vx(self, vx):
        self.vx = vx

    def set_vy(self, vy):
        self.vy = vy

    def set_vz(self, vz):
        self.vz = vz

    def set_velocity(self, vx, vy, vz):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.k = calculate_k_vector(vx, vy, vz, self.freq)

    def set_k(self, kx, ky, kz):
        self.k[0] = kx
        self.k[1] = ky
        self.k[2] = kz

    def calculate_new_k(self):
        self.k = calculate_k_vector(self.vx, self.vy, self.vz, self.w)

    def set_t(self, t):
        self.t = t

    def set_f(self, f):
        self.freq = f
        self.w = 2 * PI * f

    def get_type(self):
        return self.type

    def set_type(self, type):
        self.type = type

    def add_event(self, t):
        self.event_times.append(t)

    def get_info(self):
        print(self.name + "at (" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"
              + " with speed: (" + str(self.vx) + ", " + str(self.vy) + ", " +str(self.vz) +
              ") and frequency %f" % self.freq)

    def advance(self, t):
        self.x = self.x + t * self.vx
        self.y = self.y + t * self.vy
        self.z = self.z + t * self.vz
        return self.x, self.y, self.z

    """Changes frequency, wavevector, and velocity to obey conservation of 
       energy and momentum, respecting the dispersion relation w = vk. """

    def simulate_momentum_conservation(self, other):
        k_before = self.get_k() + other.get_k()

        # Energy conservation, match omegas.
        w_total_before = self.get_w() + other.get_w()
        w_1_after = np.random.uniform(0, w_total_before)
        w_2_after = w_total_before - w_1_after

        self.set_w(w_1_after)
        other.set_w(w_2_after)

        # Momentum conservation, match wavevectors
        k_total_before_x = self.get_kx() + other.get_kx()
        k_total_before_y = self.get_ky() + other.get_ky()
        k_total_before_z = self.get_kz() + other.get_kz()

        k_1_after_x = k_total_before_x * np.random.random()
        k_2_after_x = k_total_before_x - k_1_after_x

        k_1_after_y = k_total_before_y * np.random.random()
        k_2_after_y = k_total_before_y - k_1_after_y

        k_1_after_z = k_total_before_z * np.random.random()
        k_2_after_z = k_total_before_z - k_1_after_z

        v_1_x = (k_1_after_x / w_1_after) ** -1
        v_2_x = (k_2_after_x / w_2_after) ** -1

        v_1_y = (k_1_after_y / w_1_after) ** -1
        v_2_y = (k_2_after_y / w_2_after) ** -1

        v_1_z = (k_1_after_z / w_1_after) ** -1
        v_2_z = (k_2_after_z / w_2_after) ** -1

        # print("V_1_x after: %f" % v_1_x)
        k_after = np.array([k_1_after_x + k_2_after_x, k_1_after_y + k_2_after_y, k_1_after_z + k_2_after_z])

        # Checks to make sure momentum conservation is working properly.

        try:
            assert w_total_before == w_1_after + w_2_after
        except AssertionError:
            if not abs(w_total_before - w_1_after - w_2_after) < 1e-6:
                print("Omegas not matching up")
                os._exit(1)

        Delta_k = np.array([abs(k_before[i] - k_after[i]) for i in range(3)])

        for delta in Delta_k:
            if delta > 1e-6:
                print("Delta_K: " + str(Delta_k))
                os._exit(1)

        self.set_velocity(v_1_x, v_1_y, v_1_z)
        self.set_k(k_1_after_x, k_1_after_y, k_1_after_z)

        other.set_velocity(v_2_x, v_2_y, v_2_z)
        other.set_k(k_2_after_x, k_2_after_y, k_2_after_z)
