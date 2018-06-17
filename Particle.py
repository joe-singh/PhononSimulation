"""Particle File"""

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

    def get_name(self):
        return self.name

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

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

    def set_t(self, t):
        self.t = t

    def set_f(self, f):
        self.freq = f

    def get_type(self):
        return self.type

    def set_type(self, type):
        self.type = type

    def add_event(self, t):
        self.event_times.append(t)

    def get_info(self):
        print(self.name + "at (" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")" \
              + " with speed: (" + str(self.vx) + ", " + str(self.vy) + ") and frequency %f" % self.freq)
