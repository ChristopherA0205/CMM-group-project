# Importing Libraries and Modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

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

# Function to plot a line of best fit for the given data and return slope and intercept
def plot_best_fit(x, y, x_label, y_label, title):
    # Calculate the line of best fit
    slope, intercept, _, _, _ = stats.linregress(x, y)
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
C_M_a, C_M_0 = plot_best_fit(alpha, CM_wing, '\u03B1', 'C$_{M}$$^{wing}$', "C$_{M}$$_{0}$ + C$_{M}$$_{\u03B1}$\u03B1")
C_L_del_E, _ = plot_best_fit(delta_el, CL_el, '\u03B4', 'C$_{L}$$^{el}$', "C$_{L}$$_{\u03B4}$$_{E}$")
C_M_del_E, _ = plot_best_fit(delta_el, CM_el, '\u03B4', 'C$_{M}$$^{el}$', "C$_{M}$$_{\u03B4}$$_{E}$")

# Calculate the lift coefficient CL using the obtained values
CL = C_L_0 + C_L_a*alpha + C_L_del_E * ((-C_M_0 + C_M_a*alpha) / C_M_del_E)
# Fit the lift coefficient CL and the drag coefficient CD with a quadratic function and plot it
K, _, C_D_0 = plot_quadratic_fit(CL, CD_wing, '$C_L$', '$C_D$', 'Quadratic Curve Line of Best Fit')

# ... [Calculations & Bisection Method] ...


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
A = alpha

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


# Values that should be brought in from vehicle file and/or tidied up from equations
cbar = 1.75
C_M = C_M_0 + C_M_a*Alpha + C_M_del_E*Del_E
pitch_mom = (1/2)*p*(V**2)*cbar*S*C_M
inertia_yy = 7000   # need to make this neater/import from vehicle

# Degrees of Freedom Equations 

u_b = V*np.cos(Alpha)
w_b = V*np.sin(Alpha)
q = 0
d_u_b = (L1/m)*np.sin(Alpha)-(D/m)*np.cos(Alpha)-q*w_b-(W/m)*np.sin(Theta)+(T/m)
d_w_b = -(L1/m)*np.cos(Alpha)-(D/m)*np.sin(Alpha)+q*u_b+(W/m)*np.cos(Theta)
d_q = pitch_mom/inertia_yy
d_x_e = u_b*np.cos(Theta) + w_b*np.sin(Theta)
d_z_e = -u_b*np.sin(Theta) + w_b*np.cos(Theta)
