from UtilityMethods import *

def isotopic_scatter_rate(particle):
    return 3.67e-41 * (particle.get_f() ** 4)

def phonon_isotope_scatter(particle, t, title, box):

    # advance time:
    particle.set_t(particle.get_t() + t)
    box.update_time(particle.get_t())
    # simulate change of trajectory due to scatter
    curr_vx = particle.get_vx()
    curr_vy = particle.get_vy()
    curr_vz = particle.get_vz()
    x = particle.get_x()
    y = particle.get_y()
    z = particle.get_z()

    v_mag = get_magnitude(curr_vx, curr_vy, curr_vz)
    new_vx, new_vy, new_vz = create_random_spherical_vel(v_mag)
    particle.set_velocity(new_vx, new_vy, new_vz)

    event_str = particle.get_name() + ": Interaction Event occurred at %s" % particle.get_t() \
                + ". Isotopic scatter at (" + str(x) + ", " + str(y) + ", " + str(z) + ")." \
                + "New velocity: (" + str(new_vx) + ", " + str(new_vy) + ", " + str(new_vz) + ")."
    particle.add_event(event_str)
    print(event_str)

    # After scatter, simulate moving forward. Need to be careful about hitting boundary

    propagate(particle, box, t, title)
    return