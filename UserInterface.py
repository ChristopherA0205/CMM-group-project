


import PySimpleGUI as sg
import final_comp_project as fcp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib


matplotlib.use('TkAgg')


sg.theme('LightBlue7')


layout = [[sg.Text('Please input values for velocity and flight path angle:'), sg.Text(size=(15,1))],
          [sg.Text('Velocity, V:', size=(10, 1)), sg.Input(key='-VELOCITY-'), sg.Text('Flight Path angle, \u03b3:', size=(15, 1)), sg.Input(key='-FLIGHT_PATH_ANGLE-')],
          [sg.Text('Resulting trim conditions:'), sg.Text(size=(2,1)), sg.Text('Angle of Attack, \u03b1 (rad):', size=(17, 1)), sg.Text(key='-Output_1-'), sg.Text('Thrust, T (N):', size=(10, 1)), sg.Text(key='-Output_2-'), sg.Text('Elevator Angle, \u03b4 (rad):', size=(17, 1)), sg.Text(key='-Output_3-')],
          [sg.Button('Trim')],
          [sg.Text('Initial Altitude (m):', size=(30, 1)), sg.Input(key='-INITIAL_ALTITUDE-')],
          [sg.Text('Value of Elevator Angle Increase (%):', size=(30, 1)), sg.Input(key='-ELEV_INCREASE-'), sg.Text('Value of Thrust Increase (N):', size=(30, 1)), sg.Input(key='-THRUST_INCREASE-')],
          [sg.Text('Duration of Trim Condition (s):', size=(30, 1)), sg.Input(key='-TRIM_TIME-'), sg.Text('Value of Climb Duration (s):', size=(30, 1)), sg.Input(key='-CLIMB_TIME-')],
          [sg.Canvas(key='-CANVAS-')],
          [sg.Button('Evaluate'), sg.Button('Exit')]]



def draw_figure(canvas, figure):
   tkcanvas = FigureCanvasTkAgg(figure, canvas)
   tkcanvas.draw()
   tkcanvas.get_tk_widget().pack(side='top', fill='both', expand=1)
   return tkcanvas



window = sg.Window('Flight Simulation', layout, resizable=True)

while True:  # Event Loop
    event, values = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'Trim':
        trimVelocity = float(values['-VELOCITY-'])
        trimGamma = float(values['-FLIGHT_PATH_ANGLE-'])
        alpha, delta, theta, ub, wb, thrust = fcp.find_trim_conditions(trimVelocity, trimGamma)
        window['-Output_1-'].update(alpha)
        window['-Output_2-'].update(thrust)
        window['-Output_3-'].update(delta)
    if event == 'Evaluate':
        trimVelocity = int(float(values['-VELOCITY-']))
        trimGamma = int(float(values['-FLIGHT_PATH_ANGLE-']))
        initialAltitude = int(float(values['-INITIAL_ALTITUDE-']))
        elevatorChange = int(float(values['-ELEV_INCREASE-']))
        thrustChange = int(float(values['-THRUST_INCREASE-']))
        pitchTime = int(float(values['-TRIM_TIME-']))
        climbTime = int(float(values['-CLIMB_TIME-']))
        t_end = pitchTime + climbTime
        y = fcp.run_simulation(trimVelocity, trimGamma, t_end, pitchTime, climbTime, elevatorChange, thrustChange, initialAltitude)
        simulation_figure = fcp.display_simulation_results(y, initialAltitude)
        print("Simulation Data:", y)
        tkcanvas = draw_figure(window['-CANVAS-'].TKCanvas, simulation_figure)
        
        
        
        
window.close()
