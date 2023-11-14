'''
  ____                              _  _____   
 / ___| _ __   ___   _   _  _ __   / ||___  |  
| |  _ | '__| / _ \ | | | || '_ \  | |   / /   
| |_| || |   | (_) || |_| || |_) | | |  / /    
 \____||_|    \___/  \__,_|| .__/  |_| /_/     
                           |_|                 
 _____  _  _         _      _    
|  ___|| |(_)  __ _ | |__  | |_  
| |_   | || | / _` || '_ \ | __| 
|  _|  | || || (_| || | | || |_  
|_|    |_||_| \__, ||_| |_| \__| 
              |___/              
 ____   _                    _         _                
/ ___| (_) _ __ ___   _   _ | |  __ _ | |_   ___   _ __ 
\___ \ | || '_ ` _ \ | | | || | / _` || __| / _ \ | '__|
 ___) || || | | | | || |_| || || (_| || |_ | (_) || |   
|____/ |_||_| |_| |_| \__,_||_| \__,_| \__| \___/ |_|   

                               ________|________
                              "-----------------"
                                    || | ||
                                    ||---||
                                    ' ___ '
                          ---------'  .-.  '---------
          _________________________'  '-'  '_________________________
           ''''''-|---|----/  \==][^',_m_,'^][==/  \----|---|-''''''
                           \__/        ~        \__/
                                                         

Hello and welcome to the group 17 flight simulator! 

This file will explain to you step by step what each component of the code does, and how to make the most out of it.  
The first file you should open is 'UserInterface.py'.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                           
                            
                                                         User Guide                            
                             
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                       
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      


•	First open and then execute the file ‘UserInterface.py’, making sure that ‘MainCode.py’ is in the appropriate 
directory, to guarantee the expected operation of the User Interface (UI). Prior to running the script, please ensure that 
the 'PySimpleGUI' module is installed in your Python environment, as this is a requirement for the code to run.
   
•	Input values of ‘Velocity’ and ‘Flight Path Angle’ and press the ‘trim’ button to calculate dynamic equilibrium 
of the aircraft.

•	To simulate the response of the aircraft to changes, input the values shown (including ‘Initial Altitude’, 
the step changes in ‘Thrust’ and ‘Elevator Angle’, and the simulation duration before and after the step change’) 
and press the ‘evaluate’ button.

This simulation tool is designed to model the flight dynamics of a small aircraft. When the 'trim' function is executed, 
it takes the user-defined inputs and calculates the parameters at which the aircraft is in perfect dynamic equilibrium, 
also known as the 'trim condition'. Subsequently, a handful of these values are printed on the graphical interface to 
show that the calculations were executed as expected. 

The next section of the UI runs a simulation of how the flight dynamics change over time, starting at the previously 
determined 'trim condition' and incorporating a user-defined 'step-change'. Upon initiating the evaluation, the 
software will process your input, yielding aerodynamic data. These visualizations detail key flight dynamics,
such as altitude and pitch angle over time, allowing for in-depth analysis and design optimization.

IMPORTANT: Please find all of the main code which allows the UI to work in the MainCode.py file

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                           
                            
                                                  Imports, Files and Modules                            
                             
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                       
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

UserInterface.py: The first file you should open. It launches the graphical UI used for testing and simulations.
All the code in this file runs by the simulations in the MainCode.py, which is then imported to this file for the user to test.

MainCode.py: The file where all the computations are done, then imported to the UserInterface.py file.

Constants_and_Data.py: This file was created in order to store all the constants used in the code. The module is then imported
as 'c' for ease of use. Additionally large data sets, such as the expieramental data used to calculate the coefficients
are stored here. This file does not contain data specific for trim conditions.

Functions.py: This file is used to store small functions such as CD, CM and CL which will be used repeatedly in the code
to calculate aerodynamic parameters. This module is imported as 'L'. Large simulation functions are included in the main
code. 

Trim_Conditions.py: Although containing constants, this module is specific for trim condition values in order to import
them in the later calculations of late part A and part B. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                                                                                           
                                     
                                     Code Breakdown: For your interest, not needed to test the code.                          
                            
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                           
                            
                                                          Section A
                
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
The code in this section computes the coefficients of lift (CL), drag (CD), and pitching moment (CM) from a given  
set of experimental data. It uses the scipy.optimize module to perform curve fitting. Upon defining the aerodynamic 
coefficients, the code calculates the trim conditions for a specified velocity (V) and flight path angle (γ).   
It determines the required angle of attack (α), thrust (T), and elevator angle (δE) that achieve a steady-state 
flight condition with no acceleration. 

The simulation initiates from the aircraft's trim conditions, then computes and displays the response of the 
system to time-dependent control inputs such as changes in thrust (T) and elevator angle (δE). The graphs are
plotted using the matplotlib.pyplot library.  
 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                           
                            
                                                          Section B
                
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
 
Part B1 conducts trim analysis over a range of velocities (V) and flight path angles (γ) to determine the necessary 
thrust (T) and elevator angle (δE) adjustments for stable flight. Graphical outputs illustrate the dependencies of 
T and δE on V and γ, adhering to physical constraints such as a positive thrust value and viable control angles as well
as constraints due to the range of the experimental data used to calculate aerodynamic coefficients.

Part B2 simulates an aircraft's ascent from an altitude of 1000m to 2000m. This section  calculates the required trim 
conditions at different stages of the climb, maintaining constant velocity while adjusting the flight path angle and 
commands accordingly. A velcocity of 105m/s was tested, as V = 100 + U  where U is the day of birth of the oldest 
member of the group. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                           
                            
                                                          Section C
                
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

Section C provides a user-friendly interface for simulating flight conditions. The interface allows the user to input desired 
flight trim parameters—velocity (V) and flight path angle (γ) and displays the resulting angle of attack (α), thrust (T), and 
elevator angle (δE). Users can also input step changes in thrust and elevator angle, along with a total simulation time, to 
observe the aircraft's time response via plots of certain parameters against time. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

A message from the group: 
    
    Thank you for using our flight simulator, we hope you have a great experience! :) 


                                              
Faisal Bindakhil
Keir Barbary 
Ben Ihde 
Jack Rait 
Christopher Aitken
Daniel Brodie 
Innes Cameron                                               
                                              
                                              
                                              
                                                                                                                           '''






  
