# Read output trace for data thief
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys

fname = sys.argv[1]
reference = sys.argv[2]
xlabel = sys.argv[3]
ylabel = sys.argv[4] 

x, y = [], [] 

def linear_fit(x, a, b):
    return a*x + b

def quadratic_fit(x, a, b): 
    return a*(x**2) + b

# Looking for proportional to T^4 maybe with some offset
def quartic_fit(x, a, b):
    return a*(x**4) + b

with open(fname, 'r') as filestream:
    lines = filestream.readlines()[2:]
    for line in lines:
        currentline = line.split(",") 
        x.append(float(currentline[0])) 
        y.append(float(currentline[1])) 

x, y = np.array(x), np.array(y) 

#plt.figure(figsize=(10,7)) 
plt.scatter(x, y, s=4.0, label='Extracted datapoint', c='black') 


optimised_pars_1, pcov_1 = opt.curve_fit(linear_fit, x, y) 
optimised_pars_2, pcov_2 = opt.curve_fit(quadratic_fit, x, y) 
optimised_pars_4, pcov_4 = opt.curve_fit(quartic_fit, x, y) 
plt.plot(x, quartic_fit(x, *optimised_pars_4), label='quartic', c='r')
plt.plot(x, quadratic_fit(x, *optimised_pars_2), label='quadratic', c='b')
plt.plot(x, linear_fit(x, *optimised_pars_1), label='linear', c='g')

plt.title(reference + ' with applied fits')
plt.legend(loc='best')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid()
plt.savefig(reference + ".png")

print("Linear eqn: %f x + %f " % (optimised_pars_1[0], optimised_pars_1[1]))
print("Quadratic eqn: %f x^2 + %f " % (optimised_pars_2[0], optimised_pars_2[1]))
print("Quartic eqn: %f x^4 + %f" % (optimised_pars_4[0], optimised_pars_4[1]))

plt.show()
