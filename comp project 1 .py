# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 22:19:21 2023

@author: brodi
@edit: ben
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit


alpha = np.array([-16,-12,-8,-4,-2,0,2,4,8,12])
delta_el = np.array([-20,-10,0,10,20])

CD_wing = np.array([0.115000000000000, 0.079000000000000, 0.047000000000000, 0.031000000000000, 0.027000000000000, 0.027000000000000, 0.029000000000000, 0.034000000000000, 0.054000000000000, 0.089000000000000])
CL_wing = np.array([-1.421000000000000,-1.092000000000000,-0.695000000000000,-0.312000000000000,-0.132000000000000, 0.041000000000000, 0.218000000000000, 0.402000000000000, 0.786000000000000, 1.186000000000000])
CM_wing = np.array([0.077500000000000, 0.066300000000000, 0.053000000000000, 0.033700000000000, 0.021700000000000, 0.007300000000000,-0.009000000000000,-0.026300000000000,-0.063200000000000,-0.123500000000000])
CL_el = np.array([-0.051000000000000,-0.038000000000000, 0, 0.038000000000000, 0.052000000000000])
CM_el = np.array([0.084200000000000, 0.060100000000000,-0.000100000000000,-0.060100000000000,-0.084300000000000])


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


# Subplot 2
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

# Subplot 3
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


# Subplot 4
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

CL=C_L_0+C_L_a*alpha+C_L_del_E*((-C_M_0+C_M_a*alpha)/C_M_del_E)

# Subplt 5
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

#inputs
# values are m = mass, g = gravity, V = Velocity, p = density of air, S = wing surface area, PA = path angle, roundval = values we want answers rounded to 
m=1300
g=9.81
V=100
p=1.0065
S=20
W=m*g
PA=0.05
roundval = 4

#alpha=array degrees A=array in radians Alpha=specific alpha in radians
CL=C_L_0+C_L_a*alpha+C_L_del_E*((-C_M_0+C_M_a*alpha)/C_M_del_E)
A = np.deg2rad(alpha)

L=0.5*V**2*p*S*CL
D=0.5*V**2*p*S*CD_wing
Y=-L*np.cos(A)-D*np.sin(A)+W*np.cos(A)
def my_function(alpha):
    return -L*np.cos(A)-D*np.sin(A)+W*np.cos(A)

y = my_function(alpha)

# plt to find root
#calculate line of best fit
slope, intercept, r_value, p_value, std_err = stats.linregress(A, Y)
line_equation = f'y = {slope:.2f}x + {intercept:.2f}'
#plt data points
plt.figure(1, figsize=(6, 4))
plt.scatter(A, Y, label='Data Points')
#plt line of best fit
plt.plot(A, slope * A + intercept, label=f'Line of Best Fit: {line_equation}', color='red')
#edit plt
plt.xlabel('Alpha in radians')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.title("find root")
plt.show()
#Assign values
Gra=slope
int=intercept

#bisection method to find root
def bisection(f, a, b, N):
    if f(a)*f(b) >=0:
        print("a and b do not bound a root")
        return None
    a_n = a 
    b_n = b
    for n in range(1, N+1):
        m_n= (a_n + b_n)/2
        f_m_n= f(m_n)
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n<0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        else:
            print("Bisection method fails.")
            return None
    return (a_n + b_n)/2
    
f = lambda x: Gra*x+int
Alpha = bisection(f,0,20,250)

#calculate variables to find T
Del_E=-(C_M_0+C_M_a*Alpha)/C_M_del_E
Theta=Alpha+PA
C_D=C_D_0+K*(C_L_0+C_L_a*Alpha+C_L_del_E*Del_E)**2
D=0.5*p*V**2*S*C_D
C_L_1=C_L_0+C_L_a*Alpha+C_L_del_E*Del_E
L1=0.5*p*V**2*S*C_L_1

T=(-(L1/m)*np.sin(Alpha)+(D/m)*np.cos(Alpha)+(W/m)*np.sin(Theta))*m

Alpha = round(Alpha, roundval)
Del_E = round(Del_E, roundval)
T = round(T, roundval)
Theta = round(Theta,roundval)

print(f"For Flight path angle {PA} rad, and velocity {V} ms,")
print(f"Your Angle of Attack is {Alpha} rad")
print(f"The elevator angle is {Del_E} rad")
print(f"Your Thrust force is {T} N")
print(f"Your Pitch angle is {Theta} rad")







