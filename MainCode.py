# Importing necessary libraries for numerical calculations and plotting

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import pandas as pd
import Trim_Conditions as trim
import Constants_and_Data as c
import Functions as F
# Defining physical constants related to gravity, air properties, and aircraft characteristics
 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                             #Part A1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Print the coefficients
print(f"CL0 = {F.CL0}, CLa = {F.CLa}, CLde = {F.CLde}, CM0 = {F.CM0}, CMa = {F.CMa}, CMde = {F.CMde}, CD0 = {F.CD0}, K = {F.K}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                            Part A2
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Function to calculate the trim conditions
def calculate_trim_conditions(trimVelocity, trimGamma):
    def alpha_trim_func(alpha, trimVelocity, trimGamma):
        delta = -(F.CM0 + F.CMa * alpha) / F.CMde
        return (-F.Lift(alpha, delta, trimVelocity) * np.cos(alpha) - F.Drag(alpha, delta, trimVelocity) * np.sin(alpha) + c.mass * c.gravity * np.cos(alpha + trimGamma))

    initial_guess = 0.01  # Provide an initial guess
    alpha = optimize.newton(alpha_trim_func, initial_guess, args=(trimVelocity, trimGamma))

    delta = -(F.CM0 + F.CMa * alpha) / F.CMde
    theta = alpha + trimGamma
    ub = trimVelocity * np.cos(alpha)
    wb = trimVelocity * np.sin(alpha)
    thrust = F.Engine_Thrust(alpha, delta, theta, trimVelocity)

    return alpha, delta, theta, ub, wb, thrust

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                     Part A3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def DisplaySimulation_A3(Data, initialAltitude=0):
    t = Data.t  #Time array from simulation data
   
    # Array of tuples containing variable names, values, titles for plots, and y-axis labels
    attributes = [
        ('q', Data.y[0], "q Angular Velocity vs Time", "q [rad/s]"),
        ('theta', Data.y[1], "${\Theta}$ Pitch Angle vs Time", "${\Theta}$ [$^{0}$]"),
        ('ub', Data.y[2], "$u_{B}$ Body Axis Velocity vs Time", "$u_{B}$ [m/s]"),
        ('wb', Data.y[3], "$w_{B}$ Body Axis Velocity vs Time", "$w_{B}$ [m/s]"),
        ('xe', Data.y[4], "$x_{E}$ Horizontal Position vs Time", "$x_{e}$ [m]"),
        ('altitude', Data.y[5] * -1 + initialAltitude, "h Altitude vs Time", "Altitude h [m]")
    ]

    fig, ax = plt.subplots(3, 2, figsize=(12, 10)) # Creating subplots for each attribute

    # Loop through each attribute and create its subplot
    
    for i, (attr_name, attr_values, title, ylabel) in enumerate(attributes):
        row, col = divmod(i, 2)
        ax[row, col].plot(t, attr_values)
        ax[row, col].set_title(title, fontsize=12)
        ax[row, col].set_ylabel(ylabel, rotation='horizontal')
        ax[row, col].set_xlabel("t [s]")

    plt.tight_layout()
    plt.show()  #Dispalying all the plots 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Function to simulate the aircraft's response to control input changes during flight

def SystemControl(t, y, pitchTime, climbTime, trimParams, trimParams2, elevatorChange=0, thrustChange=0):
    
    """
    Adjusts the aircraft's control parameters based on the phase of the flight.
    t: current time
    y: state vector
    pitchTime: time at which pitch maneuver starts
    climbTime: duration of the climb phase
    trimParams: trim conditions before and after climb
    trimParams2: trim conditions during climb
    elevatorChange: percentage change in elevator deflection
    thrustChange: percentage change in thrust
    """
    
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

    return F.AircraftDynamics(t, y, delta, thrust)

# This function modifies control inputs at specified times during the flight

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Function to run the simulation using the initial conditions and user-defined parameters

def run_simulation(trimVelocity, trimGamma, t_end, pitchTime, climbTime, elevatorChange, thrustChange, initialAltitude):
    
    """
    Runs the aircraft simulation over a specified time range and control input changes.
    trimVelocity: initial trim velocity
    trimGamma: initial trim flight path angle
    t_end: end time of the simulation
    pitchTime: time at which pitch maneuver starts
    climbTime: duration of the climb phase
    elevatorChange: percentage change in elevator deflection during climb
    thrustChange: percentage change in thrust during climb
    initialAltitude: initial altitude of the aircraft
    """
    # Calculate initial trim conditions for the flight   
    trimConditions = calculate_trim_conditions(trimVelocity, trimGamma)
    trimConditions2 = calculate_trim_conditions(trimVelocity, trimGamma)  
    
    # Solve the initial value problem for the aircraft's motion
    y = integrate.solve_ivp(
        lambda t, y: SystemControl(t, y, pitchTime, climbTime, trimConditions, trimConditions2, elevatorChange, thrustChange),
        [0, t_end], 
        [0, trimConditions[2], trimConditions[3], trimConditions[4], 0, 0], 
        t_eval=np.linspace(0, t_end, t_end * 50)
    )
    # Display the results of the simulation  
    DisplaySimulation_A3(y, initialAltitude)

# Function to run the simulation using the initial conditions and user-defined parameters

run_simulation(trim.velocity_0, trim.gamma_0, 300, trim.pitchTime, trim.climbTime, trim.elevatorChange, trim.thrustChange, trim.initialAltitude)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                               Part B1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Generate arrays of velocity and flight path angle values within the specified range
V_values = np.arange(trim.V_min, trim.V_max, trim.V_step)
gamma_values = np.arange(trim.gamma_min, trim.gamma_max, trim.gamma_step)

# Initialize arrays to store the results of thrust and elevator deflection for each combination
T_values = np.empty((len(V_values), len(gamma_values)))
δE_values = np.empty((len(V_values), len(gamma_values)))

# Calculate trim conditions for each combination of velocity and flight path angle
for i, V in enumerate(V_values):
    for j, γ in enumerate(gamma_values):
        _, delta_trim, _, _, _, thrust_trim = calculate_trim_conditions(V, γ)
        T_values[i, j] = thrust_trim # Store thrust trim value
        δE_values[i, j] = np.rad2deg(delta_trim) # Store elevator deflection trim value in degrees


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

def DisplaySimulation_B2(Data, initialAltitude):
    
    """
    Displays the simulation results for various aircraft state variables over time.
    Data: Simulation data containing time and state variables
    initialAltitude: The initial altitude of the aircraft
    """    
    # Labels for plotting the state variables 
    t = Data.t # Time array from the simulation data
    labels = [
        ('$u_B$ Body Axis Velocity vs Time', '$u_B$ [m/s]'),
        ('$w_B$ Body Axis Velocity vs Time', '$w_B$ [m/s]'),
        ('$\\theta$ Pitch Angle vs Time', '$\\theta$ [rad]'),
        ('q Angular Velocity vs Time', 'q [rad/s]'),
        ('$x_E$ Horizontal Position vs Time', '$x_e$ [m]'),
        ('Altitude vs Time', 'Altitude h [m]')
    ]
    # Extracting the state variables from the simulation data
    variables = [Data.y[2], Data.y[3], Data.y[1], Data.y[0], Data.y[4], -Data.y[5]]
    
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    
    for i, axi in enumerate(ax.flat):  # Looping through each variable to create its subplot
        axi.plot(t, variables[i]) # Plotting the variable over time
        axi.set_title(labels[i][0], fontsize=12) # Setting the title for each subplot
        axi.set_ylabel(labels[i][1], rotation='horizontal') # Setting the y-axis label
        axi.set_xlabel("t [s]") # Setting the x-axis label
    
    plt.tight_layout()
    plt.show()

def sim_control_level_flight(t, y, trimParams):
    delta = trimParams[1]
    thrust = trimParams[5]
    return F.AircraftDynamics(t, y, delta, thrust)

def sim_control_climb(t, y, trimParams2):
    delta = trimParams2[1]
    thrust = trimParams2[5]
    return F.AircraftDynamics(t, y, delta, thrust)

def combined_sim_control(t, y, trimParams, trimParams2, pitchTime, climbTime):
    if pitchTime <= t < pitchTime + climbTime:
        return sim_control_climb(t, y, trimParams2)
    else:
        return sim_control_level_flight(t, y, trimParams)



def Find_Climb_Time(trimVelocity, trimGamma, t_end, initialAltitude, maxAltitude, pitchTime, climbVelocity, climbGamma, climbTimeGuess=0, climbStep=0.5):
    trimParams = calculate_trim_conditions(trimVelocity, trimGamma)
    trimParams2 = calculate_trim_conditions(climbVelocity, climbGamma)

    climbTime = climbTimeGuess
    finalAltitude = initialAltitude

    while finalAltitude < maxAltitude:
        y = integrate.solve_ivp(
            lambda t, y: combined_sim_control(t, y, trimParams, trimParams2, pitchTime, climbTime),
            [0, t_end], 
            [0, trimParams[2], trimParams[3], trimParams[4], 0, -initialAltitude],
            t_eval=np.linspace(0, t_end, t_end * 50)
        )
        finalAltitude = -y.y[5][-1]
        if finalAltitude < maxAltitude:
            climbTime += climbStep

    DisplaySimulation_B2(y, initialAltitude)
    print(f"Climb Duration: {climbTime}s")
    return climbTime


    DisplaySimulation_B2(y, initialAltitude)
    
    print(f"Climb Duration: {climbTime}s")
    return climbTime

climb_duration = Find_Climb_Time(trimVelocity=105, trimGamma=0, t_end=700, initialAltitude=1000, maxAltitude=2000, pitchTime=10, climbVelocity=105, climbGamma=np.deg2rad(2), climbTimeGuess=200, climbStep=1)
