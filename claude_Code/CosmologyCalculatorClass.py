######################################################################
# This file contains a calculator for basic cosmological calculations #
# similar to Ned Wright's calulator:                                  # 
# http://www.astro.ucla.edu/~wright/CosmoCalc.html                    #
# or http://arxiv.org/pdf/astro-ph/0609593v2.pdf                      # 
# aka A Cosmology Calculator for the World Wide Web                   #
# See also, http://arxiv.org/pdf/astro-ph/9905116v4.pdf               #
# for a review of the formulas used.                                  #
# Or Distance measures in Cosmology, David W. Hogg                    #
#######################################################################

from math import *
import scipy.integrate as integrate
import numpy
import matplotlib.pyplot as plt

class CosmoCalc(object):
    # Initializer fixes constant parameters
    # And the vairable parameters e.g. self.H_0
    def __init__(self, H_0, O_M, O_V, T_CMB):
        # Hubble constant H_0 [km s^-1 Mpc^-1]
        self.H_0 = H_0
        # Hubble parameter h [ dimensionless ]
        self.h = self.H_0 / 100
        # Matter density
        self.O_M = O_M
        # Vacuum density
        self.O_V = O_V
        # Radiation density depends only on H_0
        self.O_R = 4.165E-1/self.H_0**2
        # Fixing c [m/s]
        self.c = 299792458.0
        # Total density
        self.O_tot = self.O_M + self.O_V + self.O_R
        # Hubble distance [Mpc]
        self.D_H = self.c / (1000.0 * H_0)
        # Hubble time
        self.t_H = 1.0 / H_0
        # Boltzmann Constant [m^2 kg s^-2 K^-1]
        self.k_b = 1.3806488 * 10E-23
        # Ratio of spin degeneracy factors of the 
        # 1S triplet and singlet state
        self.ratio_g1g0 = 3
        # Planck Constant h [m^2 kg s^-1]
        self.h_planck = 6.62606957 * 10E-34
        # T_* = hc/k_b Lambda_21cm [K]
        self.T_star = self.h_planck * self.c / (self.k_b * 0.21)
        # CMB temperature [K]
        self.T_CMB = T_CMB
        # T_gamma is usually set to T_CMB [K]
        self.T_gamma = self.T_CMB 
        # Spontaneous decay-rate of the spin-flip transition [s^-1]
        self.A_10 = 2.85 * 10E-15
        # Baryon density
        self.O_b = 0.044
        
        # Collisional Coupling scattering rates [cm^3 s^-1]
        # between H and H 
        self.kappa_HH = {'1' : 1.38 * 10E-13,
                         '2' : 1.43 * 10E-13, 
                         '5' : 4.65 * 10E-13,
                         '10' : 2.88 * 10E-12, 
                         '20' : 1.78 * 10E-11,
                         '50' : 6.86 * 10E-11, 
                         '100' : 1.19 * 10E-10,
                         '200' : 1.75 * 10E-10, 
                         '500' : 2.66 * 10E-10,
                         '1000' : 3.33 * 10E-10, 
                         '2000' : 0,
                         '3000' : 0, 
                         '5000' : 0,
                         '7000' : 0, 
                         '10000' : 0,
                         '15000' : 0, 
                         '20000' : 0}  
        # between H and p
        self.kappa_Hp = {'1' : 0.4028,
                         '2' : 0.4517, 
                         '5' : 0.4301,
                         '10' : 0.3699, 
                         '20' : 0.3172,
                         '50' : 0.3047, 
                         '100' : 0.3379,
                         '200' : 0.4043, 
                         '500' : 0.5471,
                         '1000' : 0.7051, 
                         '2000' : 0.9167,
                         '3000' : 1.070, 
                         '5000' : 1.301,
                         '7000' : 1.480, 
                         '10000' : 1.695,
                         '15000' : 1.975, 
                         '20000' : 2.201 }        
        # between H and e
        self.kappa_He = {'1' : 0.239,
                         '2' : 0.337, 
                         '5' : 0.503,
                         '10' : 0.746, 
                         '20' : 1.05,
                         '50' : 1.63, 
                         '100' : 2.26,
                         '200' : 3.11, 
                         '500' : 4.59,
                         '1000' : 5.92, 
                         '2000' : 7.15,
                         '3000' : 7.71, 
                         '5000' : 8.17,
                         '7000' : 8.32, 
                         '10000' : 8.37,
                         '15000' : 8.29, 
                         '20000' : 8.11 }  
        # number densities
        # TODO: Get right values here
        # of Hydrogen [cm^-3]
        self.n_H = 0.1
        # of Protons [cm^-3]
        self.n_p = 0.1
        # of electrons [cm^-3]
        self.n_e = 0.1
        

#######################################################################
    
    # Helper function, denominator for various integrals
    # E(z) 
    def E(self, z):
        return sqrt(self.O_V + self.O_R * (1+z)**4 +
                self.O_M * (1+z)**3 + (1-self.O_tot) * (1+z)**2)

    # Helper function, Z = int (dz/E, 0, z)
    def Z(self, z):
        integral = integrate.quad(lambda x: 1.0/self.E(x), 0, z)
        return integral[0]
    
    # Curvature term in the metric:
    #   sinh x, x or sin x 
    def S_k(self, x):
        if (self.O_V + self.O_M < 1.0):
            return sinh(x) 
        elif (self.O_V + self.O_M == 1.0):
            return x
        else:
            return sin(x)   
    
    # Hubble Time [s * Mpc/km]: 
    #   t_H = 1/H_0
    def hubble_time(self):
        return self.t_H

    # Hubble Distance [Mpc]: 
    #   D_H = c/H_0
    def hubble_dist(self):
        return self.D_H

    # Comoving distance (line of sight) [Mpc], 
    #   aka. Comoving radial distance (D_now):
    #   D_C = D_H int(dz / E(z),0,z) = D_H * Z = cZ/H_0
    def comoving_radial_dist(self, z):
        return self.hubble_dist() * self.Z(z)
    def D_C(self, z):
        return self.comoving_radial_dist(z)
    def D_now(self, z):
        return self.comoving_radial_dist(z)


    # Comoving distance (transverse) [Mpc],
    #   aka. Proper motion distance:
    #   D_M = D_H/sqrt(1-O_tot) S_k(sqrt(1-O_tot) D_C/D_H)
    def comoving_dist_transverse(self, z):
        O_k = 1-self.O_M-self.O_V
        return self.D_H / sqrt(abs(1-O_k)) * \
                self.S_k(sqrt(abs(1-O_k)) * \
                self.D_C(z)/self.D_H)
    def D_M(self, z):
        return self.comoving_dist_transverse(z)

    # Angular diameter distance [Mpc]:
    #   D_A = D_M / (1+z)
    # 
    # Second Form:
    # Angular diam dist between 2 objects at redshifts z & z2 [Mpc]
    #   formula only holds for O_tot <= 1:
    #   D_A12 = 1/(1+z_2) * (D_M2 sqrt(1+(1-O_tot) D_M1^2/D_H^2) -
    #                       D_M1 sqrt(1+(1-O_tot) D_M2^2/D_H^2) )
    def angular_diam_dist(self, z, z2 = None):
        O_k = 1-self.O_M-self.O_V
        root = sqrt(abs(1-self.O_tot))
        
        if z2 is None:
            return self.D_H * self.S_k((root) * self.Z(z))/ \
                    ((1+z) * root)
        elif (self.O_V + self.O_M <= 1.0):
            return 1.0 / (1.0 + z2) * \
                    ( self.D_M(z2) * sqrt(1 + (1 - O_k) *\
                    self.D_M(z)**2 / self.D_H**2) - \
                    self.D_M(z) * sqrt(1 + (1 - O_k) *\
                    self.D_M(z2)**2 / self.D_H**2) )
        else:
            print "Error: D_A12 formula invalid for O_tot > 1.0"
            return -1
    def D_A(self, z, z2 = None):
        return self.angular_diam_dist(z, z2)

    # Luminosity distance [Mpc]:
    #   D_L = (1 + z)^2 * D_A
    def luminosity_dist(self, z):
        return self.D_A(z) * (1+z)**2
    def D_L(self, z):
        return self.luminosity_dist(z)

    # Comoving Volume [Mpc^3]:
    #   V_C = int D_H (1+z)^2 * D_A^2 / E(z) dOmega dz
    def comoving_volume(self, z):
        vol = 4*pi*self.D_H
        integral = integrate.quad(lambda x: (1+x)**2 * \
                self.D_A(x)**2/self.E(x), 0, z)
        return vol*integral[0]
    def V_C(self, z):
        return self.comoving_volume(z)
    
    # Age of the Universe at redshift z [s * Mpc/km]:
    #   t(z) = t_H int(dz/((1+z) E(z)), z, inf)
    #def age_of_universe(self, z):
    #    age = integrate.quad(lambda a: 1.0/self.D(a), 0.0, 1.0/(1.0+z))
     #   return self.convert_to_gy(age[0]/H_0)
    def age_of_universe(self, z):
        age = integrate.quad(lambda x: 1.0/((1+x)*self.E(x)) \
                , z, numpy.inf)
        return age[0] * self.t_H

    # Light travel time [s * Mpc/km]:
    #   ltt = t(0) - t(z)
    def light_travel_time(self, z):
        return self.age_of_universe(0) - self.age_of_universe(z)

    # Distance based on light travel time [Mpc]
    # D_ltt = c * (t(0) - t(z))
    def D_ltt(self, z):
        return self.light_travel_time(z) * self.c / 1000.0


    ############################ Plotting ############################

    def plot_distances(self):
        normalization = self.H_0/self.c*1000
        x = [i/100.0 for i in range(0,1000)]
        y1 = [self.D_A(i) * normalization for i in x]
        y2 = [self.D_L(i) * normalization for i in x]
        y3 = [self.D_now(i) * normalization for i in x]
        y4 = [self.D_ltt(i) * normalization for i in x]

        plot1 = plt.loglog(x, y1, basex = 10, basey = 10, label = 'D_A')
        plot2 = plt.loglog(x, y2, basex = 10, basey = 10, label = 'D_L')
        plot3 = plt.loglog(x, y3, basex = 10, basey = 10, label = 'D_now')
        plot4 = plt.loglog(x, y4, basex = 10, basey = 10, label = 'D_ltt')
        plt.legend(loc = 'upper left')
        plt.title(r'Cosmological Distances vs Redshift for $\Omega_{tot} =$ %s' % (self.O_M + self.O_V))
        plt.xlabel('Redshift z')
        plt.ylabel(r'$H_0 D / c$')
        plt.ylim(0.01)
        plt.grid(True)
        #plt.show()
        plt.savefig('cosmo_dist.png')
    ####################### 21cm Stuff  ##########################
    
    # Total Collisional Coupling Coefficient
    def x_c(self, T_k):
        norm = self.T_star / (self.T_gamma * self.A_10)
        temp_key = str(T_k)
        res = self.n_H * self.kappa_HH[temp_key] \
                + self.n_p * self.kappa_Hp[temp_key] \
                + self.n_e * self.kappa_He[temp_key]
        return res * norm



