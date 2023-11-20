
                               ________|________
                              "-----------------"
                                    || | ||
                                    ||---||
                                    ' ___ '
                          ---------'  .-.  '---------
          _________________________'  '-'  '_________________________
           ''''''-|---|----/  \==][^',_m_,'^][==/  \----|---|-''''''
                           \__/        ~        \__/


**Hello and welcome to the group 17 flight simulator!**

This file will explain to you step by step what each component of the code does, and how to make the most out of it.  
The first file you should open is 'B_UserInterface.py'. Files are accompanied with alphabets to organize them within the
directory.

------------------------------------------------------------------------------------------------------------------------                        
                            
                                                         User Guide                            
                             
------------------------------------------------------------------------------------------------------------------------  

•	Prior to running the script, *please ensure* that the 'PySimpleGUI' module is installed in your 
  Python environment, as this is a requirement for the code to run.
  
•	Once installed, open and then execute the file ‘B_UserInterface.py’, making sure that ‘C_MainCode.py’ is in the 
  appropriate directory, to guarantee the expected operation of the User Interface (UI). 
  
•	Input specified values and press the ‘**trim**’ button to calculate dynamic equilibrium of the aircraft.

•	To simulate the response of the aircraft to changes, input the values show and press the ‘**evaluate**’ button.

#### List of values that are required in the UI
| Input | Description and Limitations | Units | Options (example) |
| ------ | --------------------------- | ----- | ----------------- |
| `Velocity` | Velocity of the aircraft must be positive, with an approximate maximum value of 200. | `m/s` | 100 |
| `Flight Path Angle` | Angle between the earth axis and trajectory, with an approximate value between -0.25 and 1. | `radians` | 0.05 |
| `Initial Altitude` | Cruising altitude of a small aircraft should be somewhere between 1000 and 3000. | `metres` | 2000 |
| `Value of Elevator Angle Increase` | Percentage change in Elevator Angle, can be positive or negative but magnitude should be within a about 50% of initial value. | `%` | 10 |
| `Value of Thrust Increase` | Thrust step change can be positive or negative, with a magnitude of less than about 1000. | `N` | -20 |
| `Duration of Trim Condition` | How long the aircraft will stay at the trim condition calculated prior. | `seconds` | 100 |
| `Value of Climb Duration` | How long the simulation will last for after the step change is incorporated. | `seconds` | 300 |


This simulation tool is designed to model the flight dynamics of a small aircraft. When the 'trim' function is executed, 
it takes the user-defined inputs and calculates the parameters at which the aircraft is in perfect dynamic equilibrium, 
also known as the 'trim condition'. Subsequently, a handful of these values are printed on the graphical interface to 
show that the calculations were executed as expected. 

The next section of the UI runs a simulation of how the flight dynamics change over time, starting at the previously 
determined 'trim condition' and incorporating a user-defined 'step-change'. Upon initiating the evaluation, the 
software will process your input, yielding aerodynamic data. These visualizations detail key flight dynamics,
such as altitude and pitch angle over time, allowing for in-depth analysis and design optimization.

**IMPORTANT**: Please find all of the main code which allows the UI to work in the C_MainCode.py file

------------------------------------------------------------------------------------------------------------------------                  
                            
                                                  Imports, Files and Modules                      
  
------------------------------------------------------------------------------------------------------------------------

*B_UserInterface.py*: The first file you should open. It launches the graphical UI used for testing and simulations.
All the code in this file runs by the simulations in the MainCode.py, which is then imported to this file for the user to test.

*C_MainCode.py*: The file where all the computations are done, then imported to the UserInterface.py file.

*D_Constants_and_Data.py*: This file was created in order to store all the constants used in the code. The module is then imported
as 'c' for ease of use. Additionally large data sets, such as the expieramental data used to calculate the coefficients
are stored here. This file does not contain data specific for trim conditions.

*E_Functions.py*: This file is used to store small functions such as CD, CM and CL which will be ued repeatedly in the code
to calculate aerodynamic parameters. This module is imported as 'F'. Large simulation functions are included in the main
code. 

*F_Trim_Conditions.py*: Although containing constants, this module is specific for trim condition values in order to import
them in the later calculations of late part A and part B. 

------------------------------------------------------------------------------------------------------------------------
    
    Thank you for using our flight simulator, we hope you have a great experience! :) 


                                              
  Project Members: Faisal Bindakhil,
  Keir Barbary, 
  Ben Ihde, 
  Jack Rait, 
  Christopher Aitken,
  Daniel Brodie, 
  and Innes Cameron
  ------------------------------------------------------------------------------------------------------------------------
