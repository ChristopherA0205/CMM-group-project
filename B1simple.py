#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:14:51 2023

@author: ben
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:43:41 2023

@author: jackrait
"""

#Importing Libraries and Modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import newton


# Define alpha and delta_el values
alpha = np.deg2rad(np.array([-16,-12,-8,-4,-2,0,2,4,8,12]))
delta_el = np.deg2rad(np.array([-20,-10,0,10,20]))

# Define aerodynamic coefficients for the wing for different alpha values
CD_wing = np.array([0.115, 0.079, 0.047, 0.031, 0.027, 0.027, 0.029, 0.034, 0.054, 0.089])
CL_wing = np.array([-1.421, -1.092, -0.695, -0.312, -0.132, 0.041, 0.218, 0.402, 0.786, 1.186])
CM_wing = np.array([0.0775, 0.0663, 0.053, 0.0337, 0.0217, 0.0073, -0.009, -0.0263, -0.0632, -0.1235])

# Define aerodynamic coefficients for the elevator for different delta_el values
CL_el = np.array([-0.051, -0.038, 0, 0.038, 0.052])
CM_el = np.array([0.0842, 0.0601, -0.0001, -0.0601, -0.0843])


def linear_function(x, m, c):
    return m*x + c



# Function to plot a line of best fit for the given data and return slope and intercept
def plot_best_fit(x, y, x_label, y_label, title):
    # Fit the data with the linear function
    params, _ = curve_fit(linear_function, x, y)
    slope = params[0]
    intercept = params[1]
    line_equation = f'y = {slope:.2f}x + {intercept:.2f}'
    
    # Plot the data and the line of best fit
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, label='Data Points')
    plt.plot(x, slope * x + intercept, label=f'Line of Best Fit: {line_equation}', color='red')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()
    
    return slope, intercept

# Function to plot a quadratic fit for the given data and return the parameters of the quadratic function
def plot_quadratic_fit(x, y, x_label, y_label, title):
    # Define the quadratic function
    def quadratic_function(x, a, b, c):
        return a * x**2 + b * x + c

    # Fit the data with the quadratic function
    params, _ = curve_fit(quadratic_function, x, y)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = quadratic_function(x_fit, *params)
    
    # Plot the data and the quadratic fit
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='Data', color='blue')
    plt.plot(x_fit, y_fit, label=f'Quadratic Fit: {params[0]:.2f}x^2 + {params[1]:.2f}x + {params[2]:.2f}', color='red')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return params

# Use the functions to calculate and plot the aerodynamic coefficients for different cases
C_L_a, C_L_0 = plot_best_fit(alpha, CL_wing, '\u03B1', 'C$_{L}$$^{wing}$', "C$_{L}$$_{0}$ + C$_{L}$$_{\u03B1}$\u03B1")
print(f"Lift coefficient for Wing (linear approximation): C_L_α = {C_L_a:.4f}, C_L_0 = {C_L_0:.4f}")

C_M_a, C_M_0 = plot_best_fit(alpha, CM_wing, '\u03B1', 'C$_{M}$$^{wing}$', "C$_{M}$$_{0}$ + C$_{M}$$_{\u03B1}$\u03B1")
print(f"Pitch moment coefficient for Wing (linear approximation): C_M_α = {C_M_a:.4f}, C_M_0 = {C_M_0:.4f}")

C_L_del_E, _ = plot_best_fit(delta_el, CL_el, '\u03B4', 'C$_{L}$$^{el}$', "C$_{L}$$_{\u03B4}$$_{E}$")
print(f"Lift coefficient due to elevator deflection: C_L_δE = {C_L_del_E:.4f}")

C_M_del_E, _ = plot_best_fit(delta_el, CM_el, '\u03B4', 'C$_{M}$$^{el}$', "C$_{M}$$_{\u03B4}$$_{E}$")
print(f"Pitch moment coefficient due to elevator deflection: C_M_δE = {C_M_del_E:.4f}")

# Calculate the lift coefficient CL using the obtained values
CL = C_L_0 + C_L_a*alpha + C_L_del_E * ((-C_M_0 + C_M_a*alpha) / C_M_del_E)
# Fit the lift coefficient CL and the drag coefficient CD with a quadratic function and plot it
K, _, C_D_0 = plot_quadratic_fit(CL, CD_wing, '$C_L$', '$C_D$', 'Quadratic Curve Line of Best Fit')

# ... [Calculations & Bisection Method] ...


#inputs
# values are m = mass, g = gravity, V = Velocity, p = density of air, S = wing surface area, PA = path angle, roundval = values we want answers rounded to 
m = 1300
g = 9.81
V = 100
p = 1.0065
S = 20
W = m * g
PA = 0.05
roundval = 4

# Setting arrays of input values

 



#alpha=array degrees A=array in radians Alpha=specific alpha in radians
CL = C_L_0 + C_L_a * alpha + C_L_del_E * ((-C_M_0 + C_M_a * alpha)/C_M_del_E)
A = alpha

L = 0.5 * V ** 2 * p * S * CL
D = 0.5 * V ** 2 * p * S * CD_wing
Y = -L * np.cos(A) - D * np.sin(A) + W * np.cos(A)

# plt to find root
# Fit data points
params, _ = curve_fit(linear_function, A, Y)
slope = params[0]
intercept = params[1]
line_equation = f'y = {slope:.2f}x + {intercept:.2f}'

# Plot data points
plt.figure(1, figsize=(6, 4))
plt.scatter(A, Y, label='Data Points')
# Plot line of best fit
plt.plot(A, slope * A + intercept, label=f'Line of Best Fit: {line_equation}', color='red')
# Edit plt
plt.xlabel('Alpha in radians')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.title("find root")
plt.show()
#Assign values
Gra=slope
int=intercept




#Newton Method and Equillibrium Equation
def f(Al):
    return -0.5 * p * V**2 * S * (C_L_0 + C_L_a * Al - C_L_del_E * (C_M_0 + C_M_a * Al)/C_M_del_E)*np.cos(Al) - 0.5 * p * V**2 * S * (C_D_0 + K * (C_L_0 + C_L_a * Al - C_L_del_E * (C_M_0 + C_M_a * Al)/C_M_del_E)**2) * np.sin(Al) + m * g * np.cos(Al + PA)



initial_guess = 0.01   

print(type(p)) #float
print(type(S)) #int
print(type(C_L_0)) #float
print(type(C_L_a)) #float
print(type(A)) #float
print(type(C_L_del_E)) #float
print(type(C_M_0))#float
print(type(C_M_a))#float
print(type(C_M_del_E))#float
print(type(C_D_0))#float
print(type(K))#float
print(type(m))#float
print(type(g))#float
print(type(PA)) #float

'''
Loop needs to be limited to values that do not violate the physical constraints.
Hence, Thrust must be a positive value
We also have computed coefficients from data values, so we must stay in these known conditions
Hence, -16 rad < alpha < 12 rad and -20 rad < del_E < 20 rad
For code abbreviation lb and ub stand for lower bound and upper bound respectiely 
'''

alpha_lb = np.deg2rad(-16) ; alpha_ub = np.deg2rad(12)
del_E_lb = np.deg2rad(-20) ; del_E_ub = np.deg2rad(20)
thrust_lb = 0

Vmin = 0
Vmax = 500
Gmin = 0
Gmax = 5


#nested loop to run through all possible combinations of inputs

# Initializing empty lists for T (thrust) and δE (elevator angle)
T_values = []
Del_E_values = []

for V in np.linspace(Vmin, Vmax, 500):
    for gamma in np.linspace(Gmin, Gmax, 5):
        
        Alpha_next = newton(f, initial_guess)  
       
        Del_E = -(C_M_0 + C_M_a * Alpha_next) / C_M_del_E
        Theta = Alpha_next + gamma
        C_D = C_D_0 + K * (C_L_0 + C_L_a * Alpha_next + C_L_del_E * Del_E) ** 2
        D = 0.5 * p * V ** 2 * S * C_D
        C_L_1 = C_L_0 + C_L_a * Alpha_next + C_L_del_E * Del_E
        L1 = 0.5 * p * V ** 2 * S * C_L_1
        Thrust = (-(L1 // m) * np.sin(Alpha_next) + (D // m) * np.cos(Alpha_next) + (W / m) * np.sin(Theta)) * m
        print(f'Alpha: {Alpha_next}')
        print(f'Thrust: {Thrust}')
        print(f'gamma: {gamma}')
        print(f'Del_E: {Del_E}')
        print(f'Velocity: {V}')
        print(f'Theta: {Theta}')
        print('- - - - - - - - - ')
        

        # Check and apply physical constraints for T, α, and δE based on experimental data
        if alpha_lb <= Alpha_next <= alpha_ub and del_E_lb <= Del_E <= del_E_ub and Thrust >0:
               T_values.append(Thrust)
               Del_E_values.append(Del_E)
               
print(Del_E_values)
print(T_values)

# Plot T and δE against V and γ
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(T_values, label='Thrust')
plt.xlabel('Index')
plt.ylabel('Thrust')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Del_E_values, label='Elevator Angle')
plt.xlabel('Index')
plt.ylabel('Elevator Angle')
plt.legend()

plt.show()


# Values that should be brought in from vehicle file and/or tidied up from equations

cbar = 1.75
C_M = C_M_0 + C_M_a * Alpha + C_M_del_E * Del_E
pitch_mom = (1/2) * p * (V**2) * cbar * S * C_M
inertia_yy = 7000

# Degrees of Freedom Equations 

u_b = V * np.cos(Alpha)
w_b = V * np.sin(Alpha)
q = 0
d_u_b = (L1/m) * np.sin(Alpha) - (D/m) * np.cos(Alpha) - q * w_b - (W/m) * np.sin(Theta) + (Thrust/m)
d_w_b = -(L1/m) * np.cos(Alpha) - (D/m) * np.sin(Alpha) + q * u_b + (W/m) * np.cos(Theta)
d_q = pitch_mom/inertia_yy
d_x_e = u_b * np.cos(Theta) + w_b * np.sin(Theta)
d_z_e = -u_b * np.sin(Theta) + w_b * np.cos(Theta)