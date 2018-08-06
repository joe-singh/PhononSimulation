# Read output trace for data thief
import numpy as np
from numpy import pi as pi
import matplotlib.pyplot as plt
import scipy.optimize as opt
from  scipy.stats import chisquare
import sys
from decimal import Decimal

fname = sys.argv[1]
reference = sys.argv[2]
xlabel = sys.argv[3]
ylabel = sys.argv[4] 

x, y = [], [] 

def linear_fit(x, a, b):
    return a*x + b

def quadratic_fit(x, a, b, c): 
    return a*(x**2) + b*x + c

# NOTE CONVERT TO FREQUENCY
# Angular Freq in Hz
def T_to_w(T):
    return 2 * pi * 90e9 * T   

def isotopic_bulk_rate(T):
    return (2.43e-42) * (T_to_w(T) / (2 * pi)) ** 4

def anharmonic_bulk_rate(T):
    return (7.41e-56) * (T_to_w(T) / (2 * pi)) ** 5

def total_bulk_rate(T):
    return isotopic_bulk_rate(T) + anharmonic_bulk_rate(T)

def quartic_fit(x, a, b, c, d, e):
    return a*(x**4) + b*(x**3) + c*(x**2) + d*x + e 

def cubic_fit(x, a, b, c, d):
    return a*(x**3) + b*(x**2) + c*(x) + d

def quintic_fit(x, a): 
    return a*(x**5)

with open(fname, 'r') as filestream:
    lines = filestream.readlines()[2:]
    for line in lines:
        currentline = line.split(",") 
        x.append(float(currentline[0])) 
        y.append(float(currentline[1])) 

def apply_fit(x, y, fit, colour, plt):
     optimised_pars, pcov = opt.curve_fit(fit, x, y)
     #print("Parameters for " + label) 
     for i in range(len(optimised_pars)):
         print("Parameter number %i: %.2Ef" % (i, Decimal(optimised_pars[i])))
     print('\n') 	
     
     plt.plot(freqs, fit(freqs, *optimised_pars), c=colour, linestyle='--')


x, y = np.array(x), np.array(y) 
# for val in x:
#     print(val)
bulk_rates = np.zeros(len(x))

for i in range(len(x)):
    bulk_rates[i] = total_bulk_rate(x[i]) 

# Since data is in inverse scattering length multiply by velocity to get inverse seconds. 
total_rates = (5.93e5) * y 
surface_rates = total_rates - bulk_rates
perfect_diffusion_rate = np.empty(len(x))

# Extracting data from the Klistner straight line appears to suggest
# that the rough sample inverse mean free path is 2 / cm so we multiply 
# this by the Debye speed of phonons in silicon in cm/s to get a per second rate.  
perfect_diffusion_rate.fill(5.93e5 * 2)
 
#plt.figure(figsize=(10,7)) 
f = plt.figure(1)
plt.scatter(x, total_rates, s=4.0, label='Extracted datapoint Total Rate', c='black') 
plt.scatter(x, surface_rates, s=4.0, label='Surface rate', c='cyan')
plt.scatter(x, bulk_rates, s=4.0, label='Bulk Rate', c='g')

optimised_pars_4, pcov_4 = opt.curve_fit(quartic_fit, x, surface_rates) 
plt.plot(x, quartic_fit(x, *optimised_pars_4), c='r')

plt.title(reference)
plt.legend(loc='best')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid()
plt.savefig(reference + ".png")

print("Quartic eqn: %f x^4 + %f x^3 + %f x^2 + %f x + %f" % (optimised_pars_4[0], optimised_pars_4[1], optimised_pars_4[2], optimised_pars_4[3], optimised_pars_4[4]))

#plt.show()

g = plt.figure(2)
total_diffusion_probability = surface_rates / perfect_diffusion_rate  
freqs = np.zeros(len(x))

# Frequencies in GHz
for i in range(len(x)):
    freqs[i] = T_to_w(x[i]) / (2*pi * 1e9) 

# All distances in cm
G_0 = 1/(65e-6) 
L_c = 1 
sad_280GHz = G_0 * L_c / (5.93e5) 
print("SAD 280GHZ %f " %  sad_280GHz)
sad = sad_280GHz * (freqs /(280)) ** 5
p_specular = 1 - total_diffusion_probability
p_lambertian = total_diffusion_probability - sad 
one = np.ones(len(x))


f_3_5K = T_to_w(3.5) / (2*pi)

sad_3_5K = sad_280GHz * (f_3_5K/280e9)**5

print("3.5K probability is %f" % sad_3_5K)

plt.scatter(freqs, p_lambertian, s=4.0, label="Lambertian Scattering Probability", c="cyan")
plt.scatter(freqs, p_specular, s=4.0, label="Specular Reflection Probability", c="r")
plt.scatter(freqs, sad, s=4.0, label="SAD Probability", c="b")
plt.scatter(freqs, total_diffusion_probability, s=3.0, label="Total Diffusive Probability", c="g")
plt.axvline(80, label='80 GHz')
plt.plot(freqs, sad_280GHz * (freqs/280)**5, c='blue')
apply_fit(freqs, total_diffusion_probability, quartic_fit, "g", plt)
apply_fit(freqs, p_lambertian, quartic_fit, "cyan", plt) 
apply_fit(freqs, p_specular, quartic_fit, "r", plt) 



def lambertian(f): 
    return  -2.98e-11 * (f**4) + 1.71e-8 * (f**3) - 2.47e-6* (f**2) + 7.83e-4*f + 5.88e-2 

def total_diff(f):
    return  -2.9e-13 * f**4 - 3.1e-9 * f**3 + 3.21e-6 * f**2 + 2.03e-4 * f + 7.16e-2
plt.plot(freqs, np.piecewise(freqs, [freqs < 366.2, freqs >= 366.2], [lambda f:-2.98e-11 * (f**4) + 1.71e-8 * (f**3) - 2.47e-6* (f**2) + 7.83e-4*f + 5.88e-2, 0.318]), label='Physical Lambertian Probability', c='cyan')
plt.plot(freqs, np.piecewise(freqs, [freqs < 366.2, freqs >= 366.2], [lambda f: total_diff(f), lambda f: 0.318 + 1.51e-14 * f**5]), c='g', label='Physical Total Diffusive Probability')
plt.plot(freqs, np.piecewise(freqs, [freqs < 366.2, freqs >= 366.2], [lambda f: 1 - total_diff(f), lambda f: 1 - (0.318 + 1.51e-14 * f**5)]), c='r', label='Physical Specular Probability')
plt.plot(freqs, one, c="black")
plt.xlabel("Frequency / GHz")
plt.ylabel("Probability")
plt.legend(loc="best") 
plt.ylim(0, 1) 
plt.xlim(0, 540)
plt.grid()
plt.title("Total Diffusive Surface Interaction Probability")
plt.show() 

