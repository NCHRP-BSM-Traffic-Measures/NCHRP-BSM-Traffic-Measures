# NCHRP 03-137: Safety Evaluation Code Package Documentation

This folder contains the source code and the user manual for safety evaluation algorithms for the NCHRP 03-137 Project: Algorithms to Convert Basic Safety Messages into Traffic Measures. A detailed description of the methodologies can be found in the project final report. The source code contains two different parts: 1) source code for the three algorithms to convert basic safety messages (BSM) into safety measures, and 2) the source code for algorithm test. Part I is used for the Surrogate Safety measures (SSM) generation. Part II is used to identify the optimized SSM threshold and evaluate the performance of the market penetration rate (MPR) of the connected vehicles (CV). The source code was programmed in Python.

# README Outline:
* Project Description
* Prerequisites
* Usage
	* Algorithm
	* Testing
* Release Notes
* License
* Contributors and Contact Information
* Acknowledgements

# Project Description
* Project Title:  NCHRP 03-137 - Algorithms to Convert Basic Safety Messages into Traffic Measures
* Background: Connected vehicles rely on short-range messaging using Basic Safety Message (BSM) data that includes information about vehicle size, speed, position, and heading (direction). In the future, all vehicles could send and receive this information to enhance safety and mobility. Exchange of BSM data among vehicles and traffic management systems will have the potential to generate traffic information that could be used to support current or develop new traffic safety measures. This will be particularly valuable for arterial roadways and work zones in areas without instrumentation or where transportation systems management instrumentation is disrupted by construction. The objective of the NCHRP 03-137 research project is to develop and validate algorithms that will use BSM to estimate selected traffic measures that could be used for performance monitoring, traffic management (e.g., traffic signal control, lane control), and traveler information.

# Prerequisites
- Python 3.7 (or higher)

Python Package Requirements:
- pandas 1.2.4
- numpy 1.20.3
- scipy 1.6.3
- scikit-learn 0.24.2
- matplotlib 3.4.2
- shapely 1.7.1

# Usage

## 1. Algorithm:

Step 1: Run the vehicle ID inference program

Since BSM ID of a vehicle is refreshed every few minutes and the SSM algorithms require consistent independent ID for each involved vehicle to maintain the accuracy of the output, a vehicle ID inference program is developed. This program is built to link a series BSM ID according to the location and the time and assign one specific inferred vehicle ID to each vehicle in the BSM ID link. All the vehicles with the same inferred vehicle ID will be considered as the same vehicle. This program needs to be executed for every BSM data before running any SSM generation program. If the BSM data are generated from simulated vehicle trajectories and the vehicle IDs are known, this step can be skipped.

Copy the python file to the folder that stores the BSM files, then run the following script:
```
python TCA_IDgen.py
```
Input the BSM data file name: 
```
Please input the name of the trajectory file(*.csv):
```
the program will generate the output file start with ("ReGen_") in the same directory.

Step 2: Run the safety evaluation algorithm to convert BSM to SSM

The three high-priority safety measures of interest to the NCHRP panel in this project are: 
- **_Hard Braking (HB)_** 
- **_Deceleration Rate to Avoid Crash (DRAC)_**
- **_Time To Collision with Disturbance (TTCD)_**

For HB and DRAC, there are two modes: offline mode and online mode. Offline mode means the BSM data are collected completely, and the SSM analysis is processed asynchronously. Online mode means the SSM analysis is processed synchronously with BSM data collection. TTCD only has the offline mode. 

### 1.1. Hard Braking (HB) Algorithm with Offline and Online Mode
These programs can generate the offline/online version of the hard braking event records with the step length (0.1 seconds in the project) of the BSM data.
Input: BSM data from vehicles installed CV device, the threshold of defining hard braking.
Output: The Hard Braking event records, including time stamp, the involved vehicle's ID, the deceleration rate value. For offline mode's output, the continuing hard braking events will be merged into one event. Users can find the merged event ID in the "Event_Combine" column from the output file. For online mode, every hard braking record will be recorded separately. 

Copy the python files to the folder that stores the BSM files, then run the script for Hard Braking Offline Mode:
```
python HB_Exp_Offline.py
```
or Hard Braking Online Mode:
```
python HB_Exp_Online.py
```
Input the BSM data file name and the threshold value: 
```
Please input the name of the trajectory file(*.csv): 
Please input the hard braking threshold (fpss, positive float):
```
The program will generate the output file start with ("HB_") in the same directory.

### 1.2. DRAC Algorithm with Offline and Online Mode
These programs can generate the offline/online version of the DRAC event records with the step length (0.1 seconds in the project) of the BSM data.
* Input: BSM data from vehicles installed CV device, the coordinate points location of vehicles, customized time interval, and number of processors will be assigned. The threshold of defining DRAC is suggested to be set to the appropriate maximum value.
* Output: The DRAC event records, including time stamp, involved vehicles' IDs and coordinates, the deceleration rate value, and the relative heading angle of involved vehicles. For offline mode's output, the continuing DRAC events will be merged into one event. Users can find the merged event ID in the "Event_Combine" column from the output file. For online mode, every DRAC record will be recorded separately. 

Copy the python files to the folder that stores the BSM files, then run the script for DRAC Offline Mode:
```
python DRAC_Calculation_Offline.py
```
or DRAC Online Mode:
```
python DRAC_Calculation_Online.py
```
Input the BSM data file name and other required parameters: 
```
Please input the name of the trajectory file(*.csv): 
Please select the reference point style (Front bumper: 1; Centroid: 2):
Do you want to use part of the trajectory file? (Yes: 1; No: 2)
    If yes: 
	Please input the start time of the sub-interval (1 digit float):
	Please input the end time of the sub-interval (1 digit float):
Please input the number of processors you want to use: (1 ~ max)
```
The program will generate the output file start with ("DRAC_") in the same directory.

### 1.3.	TTCD_Calculation.py
These programs can generate the offline/online version of the TTCD event records with the step length (0.1 seconds in the project) of the BSM data.
Input: BSM data from vehicles installed CV device, the threshold of TTCD, the number of Monte Carlo (MC) simulation runs, the coordinate points location of vehicles, customized time interval, and number of processors will be assigned.
Output: The TTCD event records, including time stamp, both involved vehicles' ID and coordinates, the three types of TTCD values according to the relative heading angle of involved vehicles: Rear-End TTCD (CRD_RE), Crossing TTCD (CRD_CR), Lane-Changing TTCD (CRD_LC). Each TTCD value is the cumulative value from multiple MC runs. 

Copy the python file to the folder which stores the BSM files, then run the script for TTCD Offline Mode:
```
python TTCD_Calculation.py
```
The program will ask the user to input the BSM data file name and other required parameters: 
```
Please input the name of the trajectory file(*.csv): 
Please input the TTCD threshold (1 digit float):
Please input the number of Monte Carlo simulation(int):
Please select the reference point style (Front bumper: 1; Centroid: 2):
Do you want to use part of the trajectory file? (Yes: 1; No: 2)
	If yes: 
	Please input the start time of the sub-interval (1 digit float):
	Please input the end time of the sub-interval (1 digit float):
Please input the number of processors you want to use: (1 ~ max)
```
The program will generate the output file start with ("TTCD_") in the same directory.


## 2. Testing:
Algorithm testing is used to identify the optimized SSM threshold and evaluate the performance of the market penetration rate for CVs. Before processing the tests, different SSM outputs under appropriate MPR (In this project, 100/75/50/20/5 MPR are tested) need to be prepared. There are three test steps.

Step 1 Identify the optimized SSM threshold and data slicing time interval.

This method compares the Spearman correlation coefficients between event number of 100 MPR SSM data and the number of crashes generated by different SSM thresholds and the time-slicing window sizes combinations. The threshold with maximum Spearman correlation coefficient value is used as the optimized threshold and time window combination. 

1) For HB and DRAC, copy the python file to the folder which stores the SSM files, then run the script:
```
python SSM_Opt.py
```
The program will ask the user to input the 100 MPR SSM file name and other required parameters: 
```
Please input the name of the 100MPR SSM file(*.csv): 
Please select the type of the SSM (1 - Hard Braking, 2 - DRAC):
Please input the start time of the sub-interval(int):
Please input the end time of the sub-interval(int):
```
The optimization results will be printed in command lines and saved in the output.csv file.
Since the correlation is between the 100 MPR SSM data and the real world crash data, please ensure that the duration of the SSM data and the crash data are the same (e.g., if 2-hour SSM data is used, the crash data also needs to be 2-hour).

2) For TTCD, copy the python file to the folder which stores the SSM files, then run the script:
```
python SSM_TTCD_Opt.py
```
The TTCD thresholds need to be deployed independently, so there will have multiple 100MPR TTCD outputs using different thresholds (1.5/2.0/2.5 in this project). 

The program will ask the user to input the 100 MPR's TTCD file names applying different TTCD thresholds and other required parameters: 
```
Please input the name of the 100MPR TTCD-1.5 file(*.csv): 
Please input the name of the 100MPR TTCD-2.0 file(*.csv):
Please input the name of the 100MPR TTCD-2.5 file(*.csv):
Please input the start time of the sub-interval(int):
Please input the end time of the sub-interval(int):
```
The optimization results will be printed in command lines and saved in the output.csv file.
Since the correlation is between the 100 MPR SSM data and the real crash data, please ensure that the duration of the SSM data and the crash data are the same (e.g., if 2-hour SSM data is used, the crash data also needs to be 2-hour).


Step 2 Determine the minimum MPR level

Method 1 - Basic Correlation method
This method uses the correlation coefficient to determine the minimum MPR level.

Copy the python file to the folder which stores the SSM files, then run the script:
```
python SSM_CorrCoef.py
```
The program will ask the user to input the type of SSM data, optimal time interval, the optimized threshold, 100MPR – 5MPR file names, and start/end time point: 
```
Please select the type of the SSM (1 - Hard Braking, 2 - DRAC, 3 - TTCD):
Please input the optimal time interval:
Please input the optimal threshold:
Please input the name of the 100MPR SSM file(*.csv):
Please input the name of the 75MPR SSM file(*.csv):
Please input the name of the 50MPR SSM file(*.csv):
Please input the name of the 20MPR SSM file(*.csv):
Please input the name of the 5MPR SSM file(*.csv):
Please input the start time of the sub-interval(int):
Please input the end time of the sub-interval(int):
```
After that, the program will work automatically. The Spearman correlation coefficients will be plotted and saved in .png file.

Method 2 - Elbow method
This method uses the elbow method to determine the minimum MPR level.

Copy the python file to the folder which stores the SSM files, then run the script:
```
python SSM_KDE.py
```
The program will ask the user to input the number of bins, the type of SSM data, the optimized threshold, and 5 MPR - 100 MPR file names: 
```
Please input the number of bins:
Please select the type of the SSM (1 - Hard Braking, 2 - DRAC, 3 - TTCD):
Please input the optimal threshold:
Please input the name of the 100MPR SSM file(*.csv):
Please input the name of the 75MPR SSM file(*.csv):
Please input the name of the 50MPR SSM file(*.csv):
Please input the name of the 20MPR SSM file(*.csv):
Please input the name of the 5MPR SSM file(*.csv):
```
The absolute rank differences will be plotted and saved in .png file.


## Release Notes

#### Release 1.0.0 (June 1, 2021)
- Initial release

## License
https://creativecommons.org/licenses/by-sa/4.0/

## Contributors and Contact Information

- Authors: Fan Zuo, Di Yang, Jingqin Gao, Di Sha, Kaan Ozbay, New York University C2SMART University Transportation Center.
- Contact Info: 
Fan Zuo, C2SMART Center, New York University, RH 469C, 6 MetroTech Center, Brooklyn, New York. Email: fz380@nyu.edu

## Acknowledgements
This research is supported by National Cooperative Highway Research Program, Transportation Research Board of The National Academies of Sciences, Engineering, and Medicine. The source code herein was developed by New York University under NCHRP Project 03-137 (http://apps.trb.org/cmsfeed/TRBNetProjectDisplay.asp?ProjectID=4550). Noblis was the contractor for this study. 
