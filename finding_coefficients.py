# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:43:45 2023

@author: keirb
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import aero_table

# import arrays from aero_table file

alpha = aero_table.alpha
delta_el = aero_table.delta_el

CD_wing = aero_table.CD
CL_wing = aero_table.CL
CM_wing = aero_table.CM
CL_el = aero_table.CL_el
CM_el = aero_table.CM_el

# FINDING CL_WING COEFFICIENTS (SUBPLOT 1)
#calculate line of best fit 
slope, intercept, r_value, p_value, std_err = stats.linregress(alpha, CL_wing)
line_equation = f'y = {slope:.2f}x + {intercept:.2f}'
#plt data points
plt.scatter(alpha, CL_wing, label='Data Points')
#plt line of best fit
plt.plot(alpha, slope * alpha + intercept, label=f'Line of Best Fit: {line_equation}', color='red')
#edit plto
plt.xlabel('\u03B1')
plt.ylabel('C$_{L}$$^{wing}$')
plt.legend()
plt.grid(True)
plt.title("C$_{L}$$_{0}$ + C$_{L}$$_{\u03B1}$\u03B1")
plt.show()
#Assign Values
C_L_a=slope
C_L_0=intercept


# FINDING CM WING COEFFICIENTS (SUBPLOT 2)
#calculate line of best fit
slope, intercept, r_value, p_value, std_err = stats.linregress(alpha, CM_wing)
line_equation = f'y = {slope:.2f}x + {intercept:.2f}'
#plt data points
plt.figure(1, figsize=(6, 4))
plt.scatter(alpha, CM_wing, label='Data Points')
#plt line of best fit
plt.plot(alpha, slope * alpha + intercept, label=f'Line of Best Fit: {line_equation}', color='red')
#edit plt
plt.xlabel('\u03B1')
plt.ylabel('C$_{M}$$^{wing}$')
plt.legend()
plt.grid(True)
plt.title("C$_{M}$$_{0}$ + C$_{M}$$_{\u03B1}$\u03B1")
plt.show()
#Assign Values
C_M_a=slope
C_M_0=intercept


# FINDING CL_EL COEFFICIENT (SUBPLOT 3)
#calculate line of best fit
slope, intercept, r_value, p_value, std_err = stats.linregress(delta_el, CL_el)
line_equation = f'y = {slope:.2f}x + {intercept:.2f}'
#plt data points
plt.figure(1, figsize=(6, 4))
plt.scatter(delta_el, CL_el, label='Data Points')
#plt line of best fit
plt.plot(delta_el, slope * delta_el + intercept, label=f'Line of Best Fit: {line_equation}', color='red')
#edit plt
plt.xlabel('\u03B4')
plt.ylabel('C$_{L}$$^{el}$')
plt.legend()
plt.grid(True)
plt.title("C$_{L}$$_{\u03B4}$$_{E}$")
plt.show()
#Assign Values
C_L_del_E=slope


# FINDING CM_EL COEFFICIENT (SUBPLOT 4)
#calculate line of best fit
slope, intercept, r_value, p_value, std_err = stats.linregress(delta_el, CM_el)
line_equation = f'y = {slope:.2f}x + {intercept:.2f}'
#plt data points
plt.figure(1, figsize=(6, 4))
plt.scatter(delta_el, CM_el, label='Data Points')
#plt line of best fit
plt.plot(delta_el, slope * delta_el + intercept, label=f'Line of Best Fit: {line_equation}', color='red')
#edit plt
plt.xlabel('\u03B4')
plt.ylabel('C$_{M}$$^{el}$')
plt.legend()
plt.grid(True)
plt.title("C$_{M}$$_{\u03B4}$$_{E}$")
plt.show()
#Assign Values
C_M_del_E=slope

# CALCULATING CL VALUE
CL=C_L_0+C_L_a*alpha+C_L_del_E*((-C_M_0+C_M_a*alpha)/C_M_del_E)

# FINDING CD COEFFICIENTS (SUBPLOT 5)
def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c
#calculate line of best fit
params, covariance = curve_fit(quadratic_function, CL, CD_wing)
x_fit = np.linspace(min(CL), max(CL), 100)
y_fit = quadratic_function(x_fit, *params)
#plot data and line of best fit
plt.figure(figsize=(8, 6))
plt.scatter(CL, CD_wing, label='Data', color='blue')
plt.plot(x_fit, y_fit, label=f'Quadratic Fit: {params[0]:.2f}x^2 + {params[1]:.2f}x + {params[2]:.2f}', color='red')
plt.title('Quadratic Curve Line of Best Fit')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
#Assign Values
K=params[0]
C_D_0=params[2]

