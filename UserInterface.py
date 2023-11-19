"""
UserInterface.py

This module contains the code to run the user interface, allowing a user to easily input values and commands and in return generating specified outputs.
The first part of the GUI takes inputs of velocity and flight path angle, and then returns the trim conditions at which the plane is in equilibrium.
The second part takes another series of input commands, including a starting altitude, step changes in elevator angle and/or thrust, and simulation duration. 
The GUI then outputs the plots of how some chosen parameters behave during the simulation, due to given inputs. 
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# importing necessary modules for user interface

import PySimpleGUI as sg
import MainCode as mc
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# function used to update interface window with plots

def draw_figure(canvas, figure):
   tkcanvas = FigureCanvasTkAgg(figure, canvas)
   tkcanvas.draw()
   tkcanvas.get_tk_widget().pack(side='top', fill='both', expand=1)
   return tkcanvas

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# code to set layout for the interface window

sg.theme('LightBlue7')

layout = [[sg.Text('Please input values for velocity and flight path angle:'), sg.Text(size=(15,1))],
          [sg.Text('Velocity, V (m/s):', size=(15, 1)), sg.Input(key='-VELOCITY-'), sg.Text('Flight Path angle, \u03b3 (rad):', size=(20, 1)), sg.Input(key='-FLIGHT_PATH_ANGLE-')],
          [sg.Text('Resulting trim conditions:'), sg.Text(size=(2,1)), sg.Text('Angle of Attack, \u03b1 (rad):', size=(17, 1)), sg.Text(key='-Output_1-'), sg.Text('Thrust, T (N):', size=(10, 1)), sg.Text(key='-Output_2-'), sg.Text('Elevator Angle, \u03b4 (rad):', size=(17, 1)), sg.Text(key='-Output_3-')],
          [sg.Button('Trim')],
          [sg.Text('Initial Altitude (m):', size=(30, 1)), sg.Input(key='-INITIAL_ALTITUDE-')],
          [sg.Text('Value of Elevator Angle Increase (%):', size=(30, 1)), sg.Input(key='-ELEV_INCREASE-'), sg.Text('Value of Thrust Increase (N):', size=(30, 1)), sg.Input(key='-THRUST_INCREASE-')],
          [sg.Text('Duration of Trim Condition (s):', size=(30, 1)), sg.Input(key='-TRIM_TIME-'), sg.Text('Value of Climb Duration (s):', size=(30, 1)), sg.Input(key='-CLIMB_TIME-')],
          [sg.Push(), sg.Canvas(key='-CANVAS-'), sg.Push()],
          [sg.Button('Evaluate'), sg.Button('Exit')]]

window = sg.Window('Flight Simulation', layout, resizable=True,
                   grab_anywhere=True, keep_on_top=True, finalize=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# event loop updating interface window depending on event 

while True:  # Event Loop
    event, values = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit': # event to exit the user interface
        break
    if event == 'Trim': # event for if trim button pushed
        trimVelocity = float(values['-VELOCITY-']) # setting input values as variables
        trimGamma = float(values['-FLIGHT_PATH_ANGLE-'])
        alpha, delta, theta, ub, wb, thrust = mc.calculate_trim_conditions(trimVelocity, trimGamma) # running function from main code
        window['-Output_1-'].update(round(alpha, 4)) # updating layout with outputs
        window['-Output_2-'].update(round(thrust, 4))
        window['-Output_3-'].update(round(delta, 4))
    if event == 'Evaluate': # event for if evaluate button is pushed
        trimVelocity = int(float(values['-VELOCITY-']))
        trimGamma = int(float(values['-FLIGHT_PATH_ANGLE-']))
        initialAltitude = int(float(values['-INITIAL_ALTITUDE-']))
        elevatorChange = int(float(values['-ELEV_INCREASE-']))
        thrustChange = int(float(values['-THRUST_INCREASE-']))
        pitchTime = int(float(values['-TRIM_TIME-']))
        climbTime = int(float(values['-CLIMB_TIME-']))
        t_end = pitchTime + climbTime
        y = mc.run_simulation(trimVelocity, trimGamma, t_end, pitchTime, climbTime, elevatorChange, thrustChange, initialAltitude)
        simulation_figure = mc.DisplaySimulation_A3(y, initialAltitude)
        tkcanvas = draw_figure(window['-CANVAS-'].TKCanvas, simulation_figure) # update the layout with the plots
        
window.close() # close window after event loop broken
