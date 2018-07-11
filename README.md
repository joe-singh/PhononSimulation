# PhononSimulation
Simulation of phonons in a crystal using Discrete Event Simulation. The isotopic approximation (phonon speed independent
of direction) is assumed. 

Run using python3 Main.py

Change parameters in run call at bottom of Main.py to change configuration (i.e. number of initial particles)

The current processes simulated are:

1. Bulk Isotopic Scattering
2. Bulk Anharmonic Decay
3. Boundary Lambertian Scattering
4. Boundary Surface Anharmonic Decay

To do: Update to include specular scattering off boundary surfaces.

The sources cited throughout the code can be found here:

Invited Review Article: Physics and Monte Carlo Techniques as
Relevant to Cryogenic, Phonon and Ionization Readout of CDMS
Radiation-Detectors
Steven W. Leman, https://arxiv.org/pdf/1109.1193.pdf

Tamura, Phys. Rev. B 31, #4
Tamura, Phys. Rev. B 48, #14

Sources of Loss Processes in Phonon Generation and Detection Experiments
with Superconducting Tunneling Junctions
H.J. Trumpp and W. Eisenmenger, Z. Physik B 28, 159-171 (1977)


Phonon scattering at silicon crystal surfaces
Tom Klitsner and R. O. Pohl, Physical Review B Volume 36, #12 October 1987-II
