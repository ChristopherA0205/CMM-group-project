import pandas as pd
import numpy as np

'''
This module will be used to store general constants and data sets used frequently throughout the code.
This module will not contain constants directly relatie to the trim conditions to avoid confusion between
constants and initial conditions.
'''


# Defining physical constants related to gravity, air properties, and aircraft characteristics
 
gravity = 9.81  # Earth's gravitational acceleration (m/s^2)
air_density = 1.0065  # Density of air at sea level at 15 degrees Celsius (kg/m^3)
wing_surface = 20.0  #Total surface area of the airplane's wings (m^2)
cbar = 1.75  # Average aerodynamic chord of the wing (m)
mass = 1300.0  # Total mass of the aircraft (kg)
inertia_yy = 7000  # Moment of inertia around the y-axis (pitching) (kg*m^2)


# Data for alpha, CD, CL, and CM
alpha_data = {
    'alpha': np.deg2rad([-16, -12, -8, -4, -2, 0, 2, 4, 8, 12]),  # Alpha angle data converted from degrees to radians
    'CD': [0.115, 0.079, 0.047, 0.031, 0.027, 0.027, 0.029, 0.034, 0.054, 0.089],  # Drag coefficient data
    'CL': [-1.421, -1.092, -0.695, -0.312, -0.132, 0.041, 0.218, 0.402, 0.786, 1.186],  # Lift coefficient data
    'CM': [0.0775, 0.0663, 0.053, 0.0337, 0.0217, 0.0073, -0.009, -0.0263, -0.0632, -0.1235]  # Moment coefficient data
}
alpha_df = pd.DataFrame(alpha_data)  # Creating DataFrame for alpha-related data

# Data for delta_el, CL_el, and CM_el
delta_el_data = {
    'delta_el': np.deg2rad([-20, -10, 0, 10, 20]),  # Delta angle data converted from degrees to radians
    'CL_el': [-0.051, -0.038, 0.0, 0.038, 0.052],  # Lift coefficient data related to elevator angle
    'CM_el': [0.0842, 0.0601, -0.0001, -0.0601, -0.0843]  # Moment coefficient data related to elevator angle
}
delta_el_df = pd.DataFrame(delta_el_data)  # Creating DataFrame for delta_el-related data