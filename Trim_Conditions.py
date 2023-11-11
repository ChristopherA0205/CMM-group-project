'''This module will store everything related to the trim conditions which will be directly 
applied to the simulation code'''

# User-provided parameters for initial conditions and control input changes

velocity_0 = 100 # Initial velocity (m/s)
gamma_0 = 0 # Initial flight path angle (radians)

pitchTime = 100 # Time in seconds after simulation start at which the values are changed
climbTime = 300 # Duration of climb in seconds

elevatorChange = 10 # in percent
thrustChange = 0 # in percent

initialAltitude = 2000 # Altitude at t=0

# Define the range of velocities and flight path angles
V_min = 50
V_max = 210 #In order to get an upper bound of 200, since step size is 10
gamma_min = 0
gamma_max = 1

V_step = 10
gamma_step = 0.1

alpha0 = 0.0164
thrust0 = 2755.17
theta0 = 0.01646
wb0 = 1.646
delta0 = -0.0520
q0 = 0
ub0 = 99.986