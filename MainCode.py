# Importing necessary libraries for numerical calculations and plotting

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import pandas as pd
# Defining physical constants related to gravity, air properties, and aircraft characteristics
 
gravity = 9.81  # Earth's gravitational acceleration (m/s^2)
air_density = 1.0065  # Density of air at sea level at 15 degrees Celsius (kg/m^3)
wing_surface = 20.0  #Total surface area of the airplane's wings (m^2)
cbar = 1.75  # Average aerodynamic chord of the wing (m)
mass = 1300.0  # Total mass of the aircraft (kg)
inertia_yy = 7000  # Moment of inertia around the y-axis (pitching) (kg*m^2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                     #Part A1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Data for alpha, CD, CL, and CM
alpha_data = {
    'alpha': np.deg2rad([-16, -12, -8, -4, -2, 0, 2, 4, 8, 12]),  # Converted to radians
    'CD': [0.115, 0.079, 0.047, 0.031, 0.027, 0.027, 0.029, 0.034, 0.054, 0.089],
    'CL': [-1.421, -1.092, -0.695, -0.312, -0.132, 0.041, 0.218, 0.402, 0.786, 1.186],
    'CM': [0.0775, 0.0663, 0.053, 0.0337, 0.0217, 0.0073, -0.009, -0.0263, -0.0632, -0.1235]
}
alpha_df = pd.DataFrame(alpha_data)

# Data for delta_el, CL_el, and CM_el
delta_el_data = {
    'delta_el': np.deg2rad([-20, -10, 0, 10, 20]),  # Converted to radians
    'CL_el': [-0.051, -0.038, 0.0, 0.038, 0.052],
    'CM_el': [0.0842, 0.0601, -0.0001, -0.0601, -0.0843]
}
delta_el_df = pd.DataFrame(delta_el_data)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                 

# Curve fitting to empirical data to find relationships between aerodynamic coefficients and angles
 

# Curve fitting
# Curve fitting using DataFrame
[CL0, CLa], _ = optimize.curve_fit(lambda x, a, b: a + b * x, alpha_df['alpha'], alpha_df['CL'], [0.04, 0.1])
[CLde], _ = optimize.curve_fit(lambda x, a: x * a, delta_el_df['delta_el'], delta_el_df['CL_el'], [0.003])
[CM0, CMa], _ = optimize.curve_fit(lambda x, a, b: a + b * x, alpha_df['alpha'], alpha_df['CM'], [0.0, -0.06])
[CMde], _ = optimize.curve_fit(lambda x, a: x * a, delta_el_df['delta_el'], delta_el_df['CM_el'], [-0.005])
[CD0, K], _ = optimize.curve_fit(lambda x, a, b: a + b * x**2, alpha_df['CL'], alpha_df['CD'], [0.02, 0.04])

 

# Printing the coefficients for verification (useful in a standalone script for debugging)

print(f"CL0 = {CL0}")
print(f"CLa = {CLa}")
print(f"CLde = {CLde}")
print(f"CM0 = {CM0}")
print(f"CMa = {CMa}")
print(f"CMde = {CMde}")
print(f"CD0 = {CD0}")
print(f"K = {K}") 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                                Part A2
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Defining aerodynamic coefficient functions as linear combinations of angles and constants obtained from curve fitting

 
# Coefficients
def CL(alpha, delta): return CL0 + CLa * alpha + CLde * delta
def CM(alpha, delta): return CM0 + CMa * alpha + CMde * delta
def CD(alpha, delta): return CD0 + K * CL(alpha, delta)**2

# Defining forces and moment functions that depend on velocity, angles of attack, and elevator deflection

def Lift(alpha, delta, V): return 0.5 * air_density * V**2 * wing_surface * CL(alpha, delta)
def Drag(alpha, delta, V): return 0.5 * air_density * V**2 * wing_surface * CD(alpha, delta)
def Moment(alpha, delta, V): return 0.5 * air_density * V**2 * wing_surface * cbar * CM(alpha, delta)
def Engine_Thrust(alpha, delta, theta, V): return Drag(alpha, delta, V) * np.cos(alpha) - Lift(alpha, delta, V) * np.sin(alpha) + mass * gravity * np.sin(theta)
 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Defining the differential equations governing the aircraft's motion


def AircraftDynamics ( t, y, delta, thrust):
    q, theta, ub, wb, xe, ze = y
 
    alpha = np.arctan2(wb, ub)
    velocity = np.sqrt(ub**2 + wb**2)
 
    dq_dt = (Moment(alpha, delta, velocity)/inertia_yy)
    dtheta_dt = q
 
    dub_dt = (Lift(alpha, delta, velocity) * np.sin(alpha) - Drag(alpha, delta, velocity) * np.cos(alpha) - mass * q * wb - mass * gravity * np.sin(theta) + thrust) / mass
    dwb_dt = (-Lift(alpha, delta, velocity) * np.cos(alpha) - Drag(alpha, delta, velocity) * np.sin(alpha) + mass * q * ub + mass * gravity * np.cos(theta)) / mass
 
    dxe_dt = ub * np.cos(theta) + wb * np.sin(theta)
    dze_dt = - ub * np.sin(theta) + wb * np.cos(theta)
 
    return dq_dt, dtheta_dt, dub_dt, dwb_dt, dxe_dt, dze_dt

    # These equations are derived from the aircraft's equations of motion

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Function to calculate the trim conditions
def calculate_trim_conditions(trimVelocity, trimGamma):
    def alpha_trim_func(alpha, trimVelocity, trimGamma):
        delta = -(CM0 + CMa * alpha) / CMde
        return (-Lift(alpha, delta, trimVelocity) * np.cos(alpha) - Drag(alpha, delta, trimVelocity) * np.sin(alpha) + mass * gravity * np.cos(alpha + trimGamma))

    initial_guess = 0.01  # Provide an initial guess
    alpha = optimize.newton(alpha_trim_func, initial_guess, args=(trimVelocity, trimGamma))

    delta = -(CM0 + CMa * alpha) / CMde
    theta = alpha + trimGamma
    ub = trimVelocity * np.cos(alpha)
    wb = trimVelocity * np.sin(alpha)
    thrust = Engine_Thrust(alpha, delta, theta, trimVelocity)

    return alpha, delta, theta, ub, wb, thrust

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                     Part A3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Function to display the results of the simulation graphically

def display_simulation_results(Data, initialAltitude=0):
    t = Data.t
    attributes = [
        ('q', Data.y[0], "q Angular Velocity vs Time", "q [rad/s]"),
        ('theta', Data.y[1], "${\Theta}$ Pitch Angle vs Time", "${\Theta}$ [$^{0}$]"),
        ('ub', Data.y[2], "$u_{B}$ Body Axis Velocity vs Time", "$u_{B}$ [m/s]"),
        ('wb', Data.y[3], "$w_{B}$ Body Axis Velocity vs Time", "$w_{B}$ [m/s]"),
        ('xe', Data.y[4], "$x_{E}$ Horizontal Position vs Time", "$x_{e}$ [m]"),
        ('altitude', Data.y[5] * -1 + initialAltitude, "h Altitude vs Time", "Altitude h [m]")
    ]

    fig, ax = plt.subplots(3, 2, figsize=(12, 10))

    # Loop through each attribute and create its subplot
    
    for i, (attr_name, attr_values, title, ylabel) in enumerate(attributes):
        row, col = divmod(i, 2)
        ax[row, col].plot(t, attr_values)
        ax[row, col].set_title(title, fontsize=12)
        ax[row, col].set_ylabel(ylabel, rotation='horizontal')
        ax[row, col].set_xlabel("t [s]")

    plt.tight_layout()
    plt.show()

    # Plots are created for each state variable against time using matplotlib

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Function to simulate the aircraft's response to control input changes during flight

def SystemControl(t, y, pitchTime, climbTime, trimParams, trimParams2, elevatorChange=0, thrustChange=0):
    # Determine if we are in the climb phase or not
    if pitchTime <= t < pitchTime + climbTime:
        # During climb
        alpha, delta, theta, ub, wb, thrust = trimParams2
        # Apply elevator and thrust changes if specified
        delta += delta * (elevatorChange / 100)
        thrust += thrust * (thrustChange / 100)
    else:
        # Before or after climb
        alpha, delta, theta, ub, wb, thrust = trimParams

    return AircraftDynamics(t, y, delta, thrust)



# This function modifies control inputs at specified times during the flight

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Function to run the simulation using the initial conditions and user-defined parameters

def run_simulation(trimVelocity, trimGamma, t_end, pitchTime, climbTime, elevatorChange, thrustChange, initialAltitude):
    trimConditions = calculate_trim_conditions(trimVelocity, trimGamma)
    trimConditions2 = calculate_trim_conditions(trimVelocity, trimGamma)  # Modify as needed for climb phase

    y = integrate.solve_ivp(
        lambda t, y: SystemControl(t, y, pitchTime, climbTime, trimConditions, trimConditions2, elevatorChange, thrustChange),
        [0, t_end], 
        [0, trimConditions[2], trimConditions[3], trimConditions[4], 0, 0], 
        t_eval=np.linspace(0, t_end, t_end * 50)
    )
    display_simulation_results(y, initialAltitude)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 #                                             Part B1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# User-provided parameters for initial conditions and control input changes

velocity_0 = 100 # Initial velocity (m/s)
gamma_0 = 0 # Initial flight path angle (radians)

pitchTime = 100 # Time in seconds after simulation start at which the values are changed
climbTime = 300 # Duration of climb in seconds

elevatorChange = 10 # in percent
thrustChange = 0 # in percent

initialAltitude = 2000 # Altitude at t=0

# Starting the simulation with the defined parameters

run_simulation(velocity_0, gamma_0, 300, pitchTime, climbTime, elevatorChange, thrustChange, initialAltitude)

# Define the range of velocities and flight path angles
V_min = 50
V_max = 210 #In order to get an upper bound of 200, since step size is 10
gamma_min = 0
gamma_max = 1

V_step = 10
gamma_step = 0.1

V_values = np.arange(V_min, V_max, V_step)
gamma_values = np.arange(gamma_min, gamma_max, gamma_step)

# Initialize arrays to store the results
T_values = np.empty((len(V_values), len(gamma_values)))
δE_values = np.empty((len(V_values), len(gamma_values)))

# Calculate trim conditions for each combination of velocity and flight path angle
for i, V in enumerate(V_values):
    for j, γ in enumerate(gamma_values):
        _, delta_trim, _, _, _, thrust_trim = calculate_trim_conditions(V, γ)
        T_values[i, j] = thrust_trim
        δE_values[i, j] = np.rad2deg(delta_trim)


def plot_subplot(ax, x_data, y_data_series, x_label, y_label, title, legend_labels):
    for y_data, label in zip(y_data_series, legend_labels):
        ax.plot(x_data, y_data, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

def plot_trim_results(V_values, gamma_values, T_values, δE_values):
    plt.figure(figsize=(12, 12))
    γ_degrees = np.rad2deg(gamma_values)  # Pre-calculate for reuse

    # Thrust vs. Velocity subplot
    ax1 = plt.subplot(2, 2, 1)
    plot_subplot(ax1, V_values, T_values.T, 'Velocity (V)', 'Thrust (T)',
                 'Thrust vs. Velocity', [f'γ = {deg:.1f}°' for deg in γ_degrees])

    # Elevator Angle vs. Velocity subplot
    ax2 = plt.subplot(2, 2, 4)
    plot_subplot(ax2, V_values, δE_values.T, 'Velocity (V)', 'Elevator Angle (δE) (degrees)',
                 'Elevator Angle vs. Velocity', [f'γ = {deg:.1f}°' for deg in γ_degrees])

    # Elevator Angle vs. Flight Path Angle subplot
    ax3 = plt.subplot(2, 2, 2)
    plot_subplot(ax3, γ_degrees, δE_values, 'Flight Path Angle (γ) (degrees)', 'Elevator Angle (δE) (degrees)',
                 'Elevator Angle vs. Flight Path Angle', [f'V = {V} m/s' for V in V_values])

    # Thrust vs. Flight Path Angle subplot
    ax4 = plt.subplot(2, 2, 3)
    plot_subplot(ax4, γ_degrees, T_values, 'Flight Path Angle (γ) (degrees)', 'Thrust (T)',
                 'Thrust vs. Flight Path Angle', [f'V = {V} m/s' for V in V_values])

    plt.tight_layout()
    plt.show()

plot_trim_results(V_values, gamma_values, T_values, δE_values)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#                                     Part B2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def display_sim2(Data, initialAltitude):
    t = Data.t
    labels = [
        ('$u_B$ Body Axis Velocity vs Time', '$u_B$ [m/s]'),
        ('$w_B$ Body Axis Velocity vs Time', '$w_B$ [m/s]'),
        ('$\\theta$ Pitch Angle vs Time', '$\\theta$ [rad]'),
        ('q Angular Velocity vs Time', 'q [rad/s]'),
        ('$x_E$ Horizontal Position vs Time', '$x_e$ [m]'),
        ('Altitude vs Time', 'Altitude h [m]')
    ]
    variables = [Data.y[2], Data.y[3], Data.y[1], Data.y[0], Data.y[4], -Data.y[5]]
    
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    
    for i, axi in enumerate(ax.flat):
        axi.plot(t, variables[i])
        axi.set_title(labels[i][0], fontsize=12)
        axi.set_ylabel(labels[i][1], rotation='horizontal')
        axi.set_xlabel("t [s]")
    
    plt.tight_layout()
    plt.show()


def find_climb_time(trimVelocity, trimGamma, t_end, initialAltitude, maxAltitude, pitchTime, climbVelocity, climbGamma, climbTimeGuess=0, climbStep=0.5):
    trimParams = calculate_trim_conditions(trimVelocity, trimGamma)
    trimParams2 = calculate_trim_conditions(climbVelocity, climbGamma)
    
    climbTime = climbTimeGuess
    finalAltitude = initialAltitude  # Start at initial altitude

    # Loop until the aircraft reaches the target altitude or the end of simulation time
    while finalAltitude < maxAltitude and climbTime < t_end - pitchTime:
        # Simulate with current climbTime guess
        y = integrate.solve_ivp(SystemControl, [0, t_end], [0, 0.01646, 99.986, 1.646, 0, -initialAltitude], t_eval=np.linspace(0, t_end, 1000), args=(pitchTime, climbTime, trimParams, trimParams2))
        
        # Check the final altitude reached
        finalAltitude = -y.y[5][-1]
        
        # If the desired altitude is not reached, increase the climbTime
        if finalAltitude < maxAltitude:
            climbTime += climbStep
        else:
            # If the desired altitude is reached, exit the loop
            break

    display_sim2(y, initialAltitude)
    
    print(f"Climb Duration: {climbTime}s")
    return climbTime


climb_duration = find_climb_time(trimVelocity=105, trimGamma=0, t_end=700, initialAltitude=1000, maxAltitude=2000, pitchTime=10, climbVelocity=105, climbGamma=np.deg2rad(2), climbTimeGuess=200, climbStep=1)
