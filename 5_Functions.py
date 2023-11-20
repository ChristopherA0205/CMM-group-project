
'''This module is used to store small aerodynamic functions which are used repeatedly in the code, this excludes
any simulation or graphical displays, as those are included in the main code file.'''

import 4_Constants_and_Data as c
import numpy as np
from scipy import optimize


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part A1:                                         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''The curve fitting function used to solve for the coefficients in part A1 , constants are imported from the constants.py
file. Optimize from the scipy library is used to create a line of best fit and solve for the coefficients.
'''

def Curve_Fitting(x_data, y_data, func, initial_guess, precision=5):

    params, _ = optimize.curve_fit(func, x_data, y_data, initial_guess)
    
    return [round(param, precision) for param in params]

[CL0, CLa] = Curve_Fitting(c.alpha_df['alpha'], c.alpha_df['CL'], lambda x, a, b: a + b * x, [0.04, 0.1])
[CLde] = Curve_Fitting(c.delta_el_df['delta_el'], c.delta_el_df['CL_el'], lambda x, a: x * a, [0.003])
[CM0, CMa] = Curve_Fitting(c.alpha_df['alpha'], c.alpha_df['CM'], lambda x, a, b: a + b * x, [0.0, -0.06])
[CMde] = Curve_Fitting(c.delta_el_df['delta_el'], c.delta_el_df['CM_el'], lambda x, a: x * a, [-0.005])
[CD0, K] = Curve_Fitting(c.alpha_df['CL'], c.alpha_df['CD'], lambda x, a, b: a + b * x**2, [0.02, 0.04])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part A2:                                           
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''Defining aerodynamic coefficient functions as linear combinations of angles and constants obtained from curve fitting
These functions will be used repeatedly in the code to calculate theta, delta and other aerodynamic parameters
'''

# Function to calculate Lift Coefficient (CL) based on angle of attack (alpha) and elevator deflection (delta)
def CL(alpha, delta): 
    return CL0 + CLa * alpha + CLde * delta

# Function to calculate Moment Coefficient (CM) based on alpha and delta
def CM(alpha, delta): 
    return CM0 + CMa * alpha + CMde * delta

# Function to calculate Drag Coefficient (CD) based on alpha and delta
def CD(alpha, delta): 
    return CD0 + K * CL(alpha, delta)**2

# Defining forces and moment functions dependent on velocity (V), angles of attack (alpha), and elevator deflection (delta)

# Function to calculate Lift force
def Lift(alpha, delta, V): 
    return 0.5 * c.air_density * V**2 * c.wing_surface *  CL(alpha, delta)

# Function to calculate Drag force
def Drag(alpha, delta, V): 
    return 0.5 * c.air_density * V**2 * c.wing_surface * CD(alpha, delta)

# Function to calculate Moment around the center of gravity
def Moment(alpha, delta, V): 
    return 0.5 * c.air_density * V**2 * c.wing_surface * c.cbar * CM(alpha, delta)

# Function to estimate Engine Thrust required
def Engine_Thrust(alpha, delta, theta, V): 
    return Drag(alpha, delta, V) * np.cos(alpha) - Lift(alpha, delta, V) * np.sin(alpha) + c.mass * c.gravity * np.sin(theta)
 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part A3:                                           
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    
def AircraftDynamics ( t, y, delta, thrust):
    
    """
    Defines the differential equations for the aircraft's motion.
    t: time
    y: state vector [q, theta, ub, wb, xe, ze]
    delta: elevator deflection angle
    thrust: engine thrust
    Returns the time derivatives of the state vector.
    """
    
    
    q, theta, ub, wb, xe, ze = y # Unpacking the state vector
    alpha = np.arctan2(wb, ub) # Calculating angle of attack
    velocity = np.sqrt(ub**2 + wb**2) # Calculating velocity
 
# Differential equations for aircraft motion
   
    dq_dt = (Moment(alpha, delta, velocity)/c.inertia_yy)
    dtheta_dt = q
 
    dub_dt = (Lift(alpha, delta, velocity) * np.sin(alpha) - Drag(alpha, delta, velocity) * np.cos(alpha) - c.mass * q * wb - c.mass * c.gravity * np.sin(theta) + thrust) / c.mass
    dwb_dt = (-Lift(alpha, delta, velocity) * np.cos(alpha) - Drag(alpha, delta, velocity) * np.sin(alpha) + c.mass * q * ub + c.mass * c.gravity * np.cos(theta)) / c.mass
 
    dxe_dt = ub * np.cos(theta) + wb * np.sin(theta)
    dze_dt = - ub * np.sin(theta) + wb * np.cos(theta)
 
    return dq_dt, dtheta_dt, dub_dt, dwb_dt, dxe_dt, dze_dt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part B                                           
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
 
''' No new equations worth adding to this file were introduced as the code takes a turn to engineering design
 tasks rather than computing paramaters using trim conditions. The code becomes simulation heavy, which is 
 integrated into the Main_Code.py file'''
    
    
    
    
    
