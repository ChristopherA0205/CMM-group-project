# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:44:48 2023

@author: keirb
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import finding_coefficients

#importing coefficients found with experimental data

C_L_a = finding_coefficients.C_L_a
C_L_0 = finding_coefficients.C_L_0
C_M_a = finding_coefficients.C_M_a
C_M_0 = finding_coefficients.C_M_0
C_L_del_E = finding_coefficients.C_L_del_E
C_M_del_E = finding_coefficients.C_M_del_E
CL = finding_coefficients.CL
K = finding_coefficients.K
C_D_0 = finding_coefficients.C_D_0
alpha = finding_coefficients.alpha
CD_wing = finding_coefficients.CD_wing


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
print(f"Your Pitch angle is {Theta} is")