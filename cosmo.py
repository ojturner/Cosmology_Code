#import the relevant modules
import os, sys, numpy as np, random, matplotlib.mlab as mlab, matplotlib.pyplot as plt, math, matplotlib.cm as cm, scipy.interpolate as inter
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *
from matplotlib.colors import LogNorm
from numpy import fft
from scipy import integrate

#temporarily add my toolkit.py file to the PYTHONPATH
sys.path.append('/Users/owenturner/Documents/PhD/SUPA_Courses/AdvancedDataAnalysis/Homeworks')

#and import the toolkit
import toolkit

##########################################################################################
#Subroutine to return the Comoving distance, Angular diameter distance, 
#Luminosity distance to a given redshift, comoving volume within that redshift 
#and age of the universe given a set of input parameters governing the cosmology. 
#Note that this subroutine works generally for flat (Om_k = 0), open (Om_k > 0) and closed
#(Om_k < 0) cosmologies, taking Om_k = 1 - Om_M - Om_V. Uses trapezoidal quadrature
##########################################################################################
#
#Inputs: 
#		z - The chosen redshift value 
#		H_0 - The value of the hubble parameter
#		Om_M - The value of the matter density parameter
#		Om_V - The value of the vacuum density parameter
#
#Output:
#		dict - A dictionary containing the values D_C, D_A, D_L, V_C and t_0
#
#Usage: 
#		my_results = cosmoDistanceTrap(3, 70, 0.3, 0.7)
#
##########################################################################################

def cosmoDistanceTrap(z, H_0, Om_M, Om_V):


	#Define the speed of light as a constant value
	c = 2.9979245E8

	#Convert the input values to floats 
	z = float(z)
	H_0 = float(H_0)
	Om_M = float(Om_M)
	Om_V = float(Om_V)

	#Make sure that the units for H_0 are SI
	H_0 = H_0/(3.08567758E19)

	#Define the Hubble distance
	D_H = c/H_0
	D_H = D_H / 3.08567758E22

	#Define the flatness parameter
	Om_K = 1.0 - Om_M - Om_V

	#Divide z by the number of points at which we want to evaluate the function (try 1000)
	dz = z/3500
	increment = 0

	#Create an array to host the 100 values of z at which comoving distance is evaluated
	z_array = []
	for i in range(3500):
		z_array.append(increment)
		increment += dz
	

	#Evaluate the comoving distance function at each value in z_array
	y_array = []
	for value in z_array:
		y_array.append(D_H / np.sqrt(Om_M*(1 + value)**3 + Om_K*(1 + value)**2 + Om_V))


	#Now perform the trapezoidal integral of the y_array using the built in python function
	#Return the answer in Mpc, this is the line of sight comoving distance D_C
	D_C = (np.trapz(y_array, dx=dz)) 

	#Now that the integral is done, all the other distance measures are derived from this
	#Transverse comoving distance:
	#There are different formulae for D_M depending upon which type of universe we provide
	#Be careful to supply the arguments to sinh and sin in the correct units

	if Om_K > 0:

		D_M =  (D_H/np.sqrt(Om_K))*np.sinh(np.sqrt(Om_K)*(D_C/D_H)) 

	elif Om_K == 0:

		D_M = D_C 

	else: 

		D_M = (D_H/np.sqrt(abs(Om_K)))*np.sin(np.sqrt(abs(Om_K))*(D_C/D_H)) 

	#Angular Diameter Distance
	D_A = D_M / (1 + z)	
	
	#Luminosity Distance
	D_L = D_M * (1 + z)

	#Comoving Volume, also dependent on the flatness parameter
	

	if Om_K > 0:

		term1 = (D_M/D_H)*np.sqrt(1 + (Om_K*(D_M/D_H)**2))
		
		term2 = (1/np.sqrt(Om_K))*np.arcsinh(np.sqrt(Om_K)*(D_M/D_H))
		
		diff = term1 - term2
		
		term3 = 4*np.pi*(D_H**3)/(2*Om_K)
		
		V_C = term3	* diff
		

	elif Om_K == 0:

		V_C = (4*np.pi/3)*D_M**3

	else: 

		
		term1 = (D_M/D_H)*np.sqrt(1 + (Om_K*(D_M/D_H)**2))
		
		
		term2 = (1/np.sqrt(abs(Om_K)))*np.arcsin(np.sqrt(abs(Om_K))*(D_M/D_H))
		
		
		diff = term1 - term2
		
		
		term3 = 4*np.pi*(D_H**3)/(2*Om_K)
		
		
		V_C = term3	* diff

	#Return values in a dictionary
	return {'D_C' : D_C, 'D_A' : D_A, 'D_L' : D_L, 'V_C' : V_C / 1.0E9}


##########################################################################################
#Subroutine to return the Comoving distance, Angular diameter distance, 
#Luminosity distance to a given redshift, comoving volume within that redshift 
#and age of the universe given a set of input parameters governing the cosmology. 
#Note that this subroutine works generally for flat (Om_k = 0), open (Om_k > 0) and closed
#(Om_k < 0) cosmologies, taking Om_k = 1 - Om_M - Om_V. Uses Simpson's rule
##########################################################################################
#
#Inputs: 
#		z - The chosen redshift value 
#		H_0 - The value of the hubble parameter
#		Om_M - The value of the matter density parameter
#		Om_V - The value of the vacuum density parameter
#
#Output:
#		dict - A dictionary containing the values D_C, D_A, D_L, V_C and t_0
#
#Usage: 
#		my_results = cosmoDistanceTrap(3, 70, 0.3, 0.7)
#
##########################################################################################

def cosmoDistanceSimp(z, H_0, Om_M, Om_V):


	#Define the speed of light as a constant value
	c = 2.9979245E8

	#Convert the input values to floats 
	z = float(z)
	H_0 = float(H_0)
	Om_M = float(Om_M)
	Om_V = float(Om_V)

	#Make sure that the units for H_0 are SI
	H_0 = H_0/(3.08567758E19)

	#Define the Hubble distance
	D_H = c/H_0
	D_H = D_H / 3.08567758E22

	#Define the flatness parameter
	Om_K = 1.0 - Om_M - Om_V

	#Divide z by the number of points at which we want to evaluate the function (try 1000)
	dz = z/3500
	increment = 0

	#Create an array to host the 100 values of z at which comoving distance is evaluated
	z_array = []
	for i in range(3500):
		z_array.append(increment)
		increment += dz
	

	#Evaluate the comoving distance function at each value in z_array
	y_array = []
	for value in z_array:
		y_array.append(D_H / np.sqrt(Om_M*(1 + value)**3 + Om_K*(1 + value)**2 + Om_V))
	

	#Now perform the trapezoidal integral of the y_array using the built in python function
	#Return the answer in Mpc, this is the line of sight comoving distance D_C
	D_C = (integrate.simps(y_array, dx=dz)) 

	#Now that the integral is done, all the other distance measures are derived from this
	#Transverse comoving distance:
	#There are different formulae for D_M depending upon which type of universe we provide
	#Be careful to supply the arguments to sinh and sin in the correct units

	if Om_K > 0:

		D_M =  (D_H/np.sqrt(Om_K))*np.sinh(np.sqrt(Om_K)*(D_C/D_H)) 

	elif Om_K == 0:

		D_M = D_C 

	else: 

		D_M = (D_H/np.sqrt(abs(Om_K)))*np.sin(np.sqrt(abs(Om_K))*(D_C/D_H)) 

	#Angular Diameter Distance
	D_A = D_M / (1 + z)	
	
	#Luminosity Distance
	D_L = D_M * (1 + z)

	#Comoving Volume, also dependent on the flatness parameter
	

	if Om_K > 0:

		term1 = (D_M/D_H)*np.sqrt(1 + (Om_K*(D_M/D_H)**2))
		
		term2 = (1/np.sqrt(Om_K))*np.arcsinh(np.sqrt(Om_K)*(D_M/D_H))
		
		diff = term1 - term2
		
		term3 = 4*np.pi*(D_H**3)/(2*Om_K)
		
		V_C = term3	* diff
		

	elif Om_K == 0:

		V_C = (4*np.pi/3)*D_M**3

	else: 

		
		term1 = (D_M/D_H)*np.sqrt(1 + (Om_K*(D_M/D_H)**2))
		
		
		term2 = (1/np.sqrt(abs(Om_K)))*np.arcsin(np.sqrt(abs(Om_K))*(D_M/D_H))
		
		
		diff = term1 - term2
		
		
		term3 = 4*np.pi*(D_H**3)/(2*Om_K)
		
		
		V_C = term3	* diff
	

	#Return values in a dictionary
	return {'D_C' : D_C, 'D_A' : D_A, 'D_L' : D_L, 'V_C' : V_C / 1.0E9}

	##########################################################################################
#Subroutine to return the Comoving distance, Angular diameter distance, 
#Luminosity distance to a given redshift, comoving volume within that redshift 
#and age of the universe given a set of input parameters governing the cosmology. 
#Note that this subroutine works generally for flat (Om_k = 0), open (Om_k > 0) and closed
#(Om_k < 0) cosmologies, taking Om_k = 1 - Om_M - Om_V. Uses Simpson's rule
##########################################################################################
#
#Inputs: 
#		z - The chosen redshift value 
#		H_0 - The value of the hubble parameter
#		Om_M - The value of the matter density parameter
#		Om_V - The value of the vacuum density parameter
#
#Output:
#		dict - A dictionary containing the values D_C, D_A, D_L, V_C and t_0
#
#Usage: 
#		my_results = cosmoDistanceTrap(3, 70, 0.3, 0.7)
#
##########################################################################################

def cosmoDistanceRomb(z, H_0, Om_M, Om_V):


	#Define the speed of light as a constant value
	c = 2.9979245E8

	#Convert the input values to floats 
	z = float(z)
	H_0 = float(H_0)
	Om_M = float(Om_M)
	Om_V = float(Om_V)

	#Make sure that the units for H_0 are SI
	H_0 = H_0/(3.08567758E19)

	#Define the Hubble distance
	D_H = c/H_0
	D_H = D_H / 3.08567758E22

	#Define the flatness parameter
	Om_K = 1.0 - Om_M - Om_V

	#Divide z by the number of points at which we want to evaluate the function (try 1000)
	dz = z/2049
	increment = 0

	#Create an array to host the 100 values of z at which comoving distance is evaluated
	z_array = []
	for i in range(2049):
		z_array.append(increment)
		increment += dz
	

	#Evaluate the comoving distance function at each value in z_array
	y_array = []
	for value in z_array:
		y_array.append(D_H / np.sqrt(Om_M*(1 + value)**3 + Om_K*(1 + value)**2 + Om_V))
	

	#Now perform the trapezoidal integral of the y_array using the built in python function
	#Return the answer in Mpc, this is the line of sight comoving distance D_C
	D_C = (integrate.romb(y_array, dx=dz)) 

	#Now that the integral is done, all the other distance measures are derived from this
	#Transverse comoving distance:
	#There are different formulae for D_M depending upon which type of universe we provide
	#Be careful to supply the arguments to sinh and sin in the correct units

	if Om_K > 0:

		D_M =  (D_H/np.sqrt(Om_K))*np.sinh(np.sqrt(Om_K)*(D_C/D_H)) 

	elif Om_K == 0:

		D_M = D_C 

	else: 

		D_M = (D_H/np.sqrt(abs(Om_K)))*np.sin(np.sqrt(abs(Om_K))*(D_C/D_H)) 

	#Angular Diameter Distance
	D_A = D_M / (1 + z)	
	
	#Luminosity Distance
	D_L = D_M * (1 + z)

	#Comoving Volume, also dependent on the flatness parameter
	

	if Om_K > 0:

		term1 = (D_M/D_H)*np.sqrt(1 + (Om_K*(D_M/D_H)**2))
		
		term2 = (1/np.sqrt(Om_K))*np.arcsinh(np.sqrt(Om_K)*(D_M/D_H))
		
		diff = term1 - term2
		
		term3 = 4*np.pi*(D_H**3)/(2*Om_K)
		
		V_C = term3	* diff
		

	elif Om_K == 0:

		V_C = (4*np.pi/3)*D_M**3

	else: 

		
		term1 = (D_M/D_H)*np.sqrt(1 + (Om_K*(D_M/D_H)**2))
		
		
		term2 = (1/np.sqrt(abs(Om_K)))*np.arcsin(np.sqrt(abs(Om_K))*(D_M/D_H))
		
		
		diff = term1 - term2
		
		
		term3 = 4*np.pi*(D_H**3)/(2*Om_K)
		
		
		V_C = term3	* diff

	#Return values in a dictionary
	return {'D_C' : D_C, 'D_A' : D_A, 'D_L' : D_L, 'V_C' : V_C / 1.0E9}

##########################################################################################
#Subroutine to return the Comoving distance, Angular diameter distance, 
#Luminosity distance to a given redshift, comoving volume within that redshift 
#and age of the universe given a set of input parameters governing the cosmology. 
#Note that this subroutine works generally for flat (Om_k = 0), open (Om_k > 0) and closed
#(Om_k < 0) cosmologies, taking Om_k = 1 - Om_M - Om_V. Uses Simpson's rule
##########################################################################################
#
#Inputs: 
#		z - The chosen redshift value 
#		H_0 - The value of the hubble parameter
#		Om_M - The value of the matter density parameter
#		Om_V - The value of the vacuum density parameter
#
#Output:
#		dict - A dictionary containing the values D_C, D_A, D_L, V_C and t_0
#
#Usage: 
#		my_results = cosmoDistanceTrap(3, 70, 0.3, 0.7)
#
##########################################################################################

def cosmoDistanceQuad(z, H_0, Om_M, Om_V):


	#Define the speed of light as a constant value
	c = 2.9979245E8

	#Convert the input values to floats 
	z = float(z)
	H_0 = float(H_0)
	Om_M = float(Om_M)
	Om_V = float(Om_V)

	#Make sure that the units for H_0 are SI
	H_0 = H_0/(3.08567758E19)

	#Define the Hubble distance
	D_H = c/H_0
	D_H = D_H / 3.08567758E22

	#Define the flatness parameter
	Om_K = 1.0 - Om_M - Om_V

	f = lambda value: D_H / np.sqrt(Om_M*(1 + value)**3 + Om_K*(1 + value)**2 + Om_V)
	

	#Now perform the trapezoidal integral of the y_array using the built in python function
	#Return the answer in Mpc, this is the line of sight comoving distance D_C
	D_C = integrate.quad(f, 0, z, epsabs=1.49E-8)[0] 
	
	#Now that the integral is done, all the other distance measures are derived from this
	#Transverse comoving distance:
	#There are different formulae for D_M depending upon which type of universe we provide
	#Be careful to supply the arguments to sinh and sin in the correct units

	if Om_K > 0:

		D_M =  (D_H/np.sqrt(Om_K))*np.sinh(np.sqrt(Om_K)*(D_C/D_H)) 

	elif Om_K == 0:

		D_M = D_C 

	else: 

		D_M = (D_H/np.sqrt(abs(Om_K)))*np.sin(np.sqrt(abs(Om_K))*(D_C/D_H)) 

	#Angular Diameter Distance
	D_A = D_M / (1 + z)	
	
	#Luminosity Distance
	D_L = D_M * (1 + z)

	#Comoving Volume, also dependent on the flatness parameter
	

	if Om_K > 0:

		term1 = (D_M/D_H)*np.sqrt(1 + (Om_K*(D_M/D_H)**2))
		
		term2 = (1/np.sqrt(Om_K))*np.arcsinh(np.sqrt(Om_K)*(D_M/D_H))
		
		diff = term1 - term2
		
		term3 = 4*np.pi*(D_H**3)/(2*Om_K)
		
		V_C = term3	* diff
		

	elif Om_K == 0:

		V_C = (4*np.pi/3)*D_M**3

	else: 

		
		term1 = (D_M/D_H)*np.sqrt(1 + (Om_K*(D_M/D_H)**2))
		
		
		term2 = (1/np.sqrt(abs(Om_K)))*np.arcsin(np.sqrt(abs(Om_K))*(D_M/D_H))
		
		
		diff = term1 - term2
		
		
		term3 = 4*np.pi*(D_H**3)/(2*Om_K)
		
		
		V_C = term3	* diff
		


	#Find the age of the universe, integral with upper limit of infinity
	func = lambda value: 1 / (H_0 * (1 + value) * np.sqrt(Om_M*(1 + value)**3 + Om_K*(1 + value)**2 + Om_V))
	result = integrate.quad(func, 0, np.inf)
	t_0 = result[0]
	error = result[1]
	

	#Return values in a dictionary
	return {'D_C' : D_C, 't_0' : t_0 / (365 * 60 * 60 * 24E9), 'D_A' : D_A, 'D_L' : D_L, 'V_C' : V_C / 1.0E9}

def cosmoDistanceFixedQuad(z, H_0, Om_M, Om_V):


	#Define the speed of light as a constant value
	c = 2.9979245E8

	#Convert the input values to floats 
	z = float(z)
	H_0 = float(H_0)
	Om_M = float(Om_M)
	Om_V = float(Om_V)

	#Make sure that the units for H_0 are SI
	H_0 = H_0/(3.08567758E19)

	#Define the Hubble distance
	D_H = c/H_0
	D_H = D_H / 3.08567758E22

	#Define the flatness parameter
	Om_K = 1.0 - Om_M - Om_V

	f = lambda value: D_H / np.sqrt(Om_M*(1 + value)**3 + Om_K*(1 + value)**2 + Om_V)
	

	#Now perform the trapezoidal integral of the y_array using the built in python function
	#Return the answer in Mpc, this is the line of sight comoving distance D_C
	D_C = integrate.fixed_quad(f, 0, z, n=8)[0]
	

	#Now that the integral is done, all the other distance measures are derived from this
	#Transverse comoving distance:
	#There are different formulae for D_M depending upon which type of universe we provide
	#Be careful to supply the arguments to sinh and sin in the correct units

	if Om_K > 0:

		D_M =  (D_H/np.sqrt(Om_K))*np.sinh(np.sqrt(Om_K)*(D_C/D_H)) 

	elif Om_K == 0:

		D_M = D_C 

	else: 

		D_M = (D_H/np.sqrt(abs(Om_K)))*np.sin(np.sqrt(abs(Om_K))*(D_C/D_H)) 

	#Angular Diameter Distance
	D_A = D_M / (1 + z)	
	
	#Luminosity Distance
	D_L = D_M * (1 + z)

	#Comoving Volume, also dependent on the flatness parameter
	

	if Om_K > 0:

		term1 = (D_M/D_H)*np.sqrt(1 + (Om_K*(D_M/D_H)**2))
		
		term2 = (1/np.sqrt(Om_K))*np.arcsinh(np.sqrt(Om_K)*(D_M/D_H))
		
		diff = term1 - term2
		
		term3 = 4*np.pi*(D_H**3)/(2*Om_K)
		
		V_C = term3	* diff
		

	elif Om_K == 0:

		V_C = (4*np.pi/3)*D_M**3

	else: 

		
		term1 = (D_M/D_H)*np.sqrt(1 + (Om_K*(D_M/D_H)**2))
		
		
		term2 = (1/np.sqrt(abs(Om_K)))*np.arcsin(np.sqrt(abs(Om_K))*(D_M/D_H))
		
		
		diff = term1 - term2
		
		
		term3 = 4*np.pi*(D_H**3)/(2*Om_K)
		
		
		V_C = term3	* diff
		


	#Find the age of the universe, integral with upper limit of infinity
	func = lambda value: 1 / (H_0 * (1 + value) * np.sqrt(Om_M*(1 + value)**3 + Om_K*(1 + value)**2 + Om_V))
	result = integrate.quad(func, 0, np.inf)
	t_0 = result[0]
	error = result[1]
	

	#Return values in a dictionary
	return {'D_C' : D_C, 't_0' : t_0 / (365 * 60 * 60 * 24E9), 'D_A' : D_A, 'D_L' : D_L, 'V_C' : V_C / 1.0E9}

dist = cosmoDistanceTrap(3, 70, 0.28, 0.72)
print dist 
dist2 = cosmoDistanceSimp(3, 70, 0.28, 0.72)
print dist2
dist3 = cosmoDistanceRomb(3, 70, 0.28, 0.72)
print dist3
dist4 = cosmoDistanceQuad(3, 70, 0.28, 0.72)
print dist4
dist5 = cosmoDistanceFixedQuad(3, 70, 0.28, 0.72)
print dist5

#Print out some graphs of the results for a series of evenly spaced redshift values 
#First set the redshift values to compute the distance measures at 
z_vec = np.arange(0, 5, 0.1)
D_C_vec = []
D_A_vec = []
D_L_vec = []
t_0_vec = []
V_C_vec = []

#Loop round and compute the distance values at the different redshift values 
for z in z_vec: 

	results_vec = cosmoDistanceQuad(z, 70, 0.3, 0.7)
	D_C_vec.append(results_vec['D_C'])
	D_A_vec.append(results_vec['D_A'])
	D_L_vec.append(results_vec['D_L'])
	t_0_vec.append(results_vec['t_0'])
	V_C_vec.append(results_vec['V_C'])

#Make the plots 
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
ax1.plot(z_vec, D_C_vec, color='b')
ax1.set_title('$D_{C}$ vs. Redshift')
ax2.plot(z_vec, D_A_vec, color='g')
ax2.set_title('$D_{A}$ vs. Redshift')
ax3.plot(z_vec, D_L_vec, color='r')
ax3.set_title('$D_{L}$ vs. Redshift')
ax4.plot(z_vec, V_C_vec, color='black')
ax4.set_title('$V_{C}$ vs. Redshift')

# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
#f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)	
show()










