"""
Trim_Conditions.py

This module stores parameters related to the trim conditions of an aircraft simulation.
Trim conditions are the steady-state values of various flight parameters when the aircraft is in stable flight.
These parameters are crucial for setting up initial conditions and control inputs for the flight simulation.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part A3:                                           
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initial velocity of the aircraft in meters per second (m/s).
# This is the speed at which the aircraft is moving at the start of the simulation.
velocity_0 = 100 

# Initial flight path angle in radians.
# This angle defines the initial ascent or descent angle of the aircraft relative to the horizon.
gamma_0 = 0 

# Time at which the aircraft starts its pitch maneuver, measured in seconds from the start of the simulation.
# This is the moment when the aircraft begins to change its flight path angle.
pitchTime = 100 

# Duration of the aircraft's climb phase in seconds.
# This is the period during which the aircraft will be ascending.
climbTime = 300 

# Percentage change in the elevator deflection.
# This change is applied at the start of the pitch maneuver to control the aircraft's pitch.
elevatorChange = 10 

# Percentage change in the thrust.
# This specifies how much the thrust will increase or decrease during the climb phase.
thrustChange = 0 

# Initial altitude of the aircraft in meters at the start of the simulation.
initialAltitude = 2000 

# Define the range of velocities (V_min to V_max) and flight path angles (gamma_min to gamma_max) for the simulation.
# These ranges are used to study the aircraft's behavior under different flight conditions.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part B1:                                           
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Minimum and maximum velocities in meters per second (m/s) for the simulation range.
V_min = 50
V_max = 210 # Set to 210 to include 200 in the range with a step size of 10.

# Minimum and maximum flight path angles in radians for the simulation range.
gamma_min = 0
gamma_max = 2

# Step sizes for iterating over the velocity and flight path angle ranges.
V_step = 10
gamma_step = 0.1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part B2:                                           
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''Part B2 uses most of the already defined parameters in A3, thus no new initial conditions or time constraints need 
to be defined'''
