"""
Material class to simulate phonon related properties of media. 

Author: Jyotirmai (Joe) Singh 10/7/18
"""
class Material:

    def __init__(self, name, isotope_scatter_rate, anharmonic_decay_rate, v_longitudinal, v_transverse,
                 Beta, Gamma, Lambda, Mu, density, LLT_ratio, P_Al_absorb):

        self._name = name
        self._isotope_scatter_rate = isotope_scatter_rate
        self._anharmonic_decay_rate = anharmonic_decay_rate
        self._v_longitudinal = v_longitudinal
        self._v_transverse = v_transverse
        self._beta = Beta
        self._gamma = Gamma
        self._lambda = Lambda
        self._mu = Mu
        self._density = density
        self._LLT_ratio = LLT_ratio
        self._LTT_ratio = 1 - self._LLT_ratio
        self._p_al_absorb = P_Al_absorb

    def get_particle_velocity(self, particle_type):

        if particle_type == 1 or particle_type == 2:
            return self._v_transverse

        elif particle_type == 3:
            return self._v_longitudinal

    def get_name(self):
        return self._name

    def get_isotope_scatter_rate(self):
        return self._isotope_scatter_rate

    def get_anharmonic_decay_rate(self):
        return self._anharmonic_decay_rate

    def get_LTT_ratio(self):
        return self._LTT_ratio

    def get_LLT_ratio(self):
        return self._LLT_ratio

    def get_LLT_rate(self):
        return self._LLT_ratio * self._anharmonic_decay_rate

    def get_LTT_rate(self):
        return self._LTT_ratio * self._anharmonic_decay_rate

    def get_transverse_vel(self):
        return self._v_transverse

    def get_longitudinal_vel(self):
        return self._v_longitudinal

    def get_beta(self):
        return self._beta

    def get_gamma(self):
        return self._gamma

    def get_lambda(self):
        return self._lambda

    def get_mu(self):
        return self._mu

    def get_density(self):
        return self._density

    def get_sensor_absorb_probability(self):
        return self._p_al_absorb