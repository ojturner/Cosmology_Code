###########################################
# This file handles user input and output #
# for the Cosmology Calculator Class      #
###########################################

from CosmologyCalculatorClass import CosmoCalc
import argparse

############## Parsing input ##############

descr = 'This program uses the Cosmology Calculator Class \
         to calculate various cosmological results including \
         the age of the universe and various cosmological distances.\n \
         The following parameters can be entered manually, H_0, O_M, O_V,\
         z and T_CMB. Default values are assumed if nothing is entered.'

parser = argparse.ArgumentParser(description=descr)
parser.add_argument('--version', action='version', 
        version='%(prog)s v0.1')
parser.add_argument('--H_0', metavar = 'H_0', 
        type = float, default = 70.0,
        help = 'Hubble Constant [km/s/Mpc]')
parser.add_argument('--O_M', metavar = 'O_M', 
        type = float, default = 0.3, help = 'Matter density')
parser.add_argument('--O_V', metavar = 'O_V', 
        type = float, default = 0.7, help = 'Vacuum density')
parser.add_argument('--z', metavar = 'z', 
        type = float, default = 3, help = 'redshift')
parser.add_argument('--T_CMB', metavar = 'T_CMB', 
        type = float, default = 2.75, help = 'CMB temperature')

args = parser.parse_args()
z = args.z

################# Output ################## 

# TODO: Maybe include this in the class
def convert_to_gy(age):
    return age * 10**10 * 3.08568 / (365.25 * 24 * 3600)

calc = CosmoCalc(args.H_0, args.O_M, args.O_V, args.T_CMB)

print "\nFor a Universe with H0 = %s, Omega_M = %s, Omega_V = %s, z = %s and T_CMB = %s K:\n" % (calc.H_0, calc.O_M, calc.O_V, z, calc.T_CMB)
print "It is now %s Gyr since the Big Bang." % \
        convert_to_gy(calc.age_of_universe(0))
print "The age at redshift z was %s Gyr." % \
        convert_to_gy(calc.age_of_universe(z))
print "The light travel time was %s Gyr." % \
        convert_to_gy(calc.light_travel_time(z))
print "The comoving radial distance is %s MPc." % \
        calc.comoving_radial_dist(z)
vol = calc.comoving_volume(z) / 10**9
print "The comoving volume within redshift z is %s Gpc^3." % \
        vol
print "The angular size distance D_A is %s MPc." % \
        calc.angular_diam_dist(z)
print "The luminosity distance D_L is %s MPc." % \
        calc.luminosity_dist(z)
