# NCHRP 03-137: Mobility Estimation Code Package Documentation

This folder contains the source code and the user manual for mobilitiy estimation algorithms for the NCHRP 03-137 Project: Algorithms to Convert Basic Safety Messages into Traffic Measures. A detailed description of the methodologies can be found in the project final report. The source code contains three different parts: the four algorithms to convert basic safety messages (BSM) into mobility measures and the code for algorithm testing. The source code was programmed in Python.

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
This folder contains four algorithms for the estimation of four mobility measures using BSMs: travel time, space mean speed, queue length and mean time to detect and verify incidents.

# Prerequisites
- Python 3.7 (or higher)

Python Package Requirements:
- imblearn==0.0
- Keras-Preprocessing==1.1.2
- keras-tuner==1.0.1
- numpy==1.19.1
- pandas==1.1.0
- tensorflow==2.3.0
- tensorflow-estimator==2.3.0

# Usage
Instructions for how to run the four mobility measures are included in their individual README files in each folder. Below are the usage intructions for the support modules. References to I-405 refer to the Seattle I-405 emulated Basic Safety Messages available through the ITS DataHub here https://datahub.transportation.gov/Automobiles/Seattle-I-405-Simulated-Basic-Safety-Message/ntts-pk3f. A small sample of the I-405 BSM data is in the Test folder that you can use for running the tests and understanding the structure of the TCA BSM output. 

## Support Modules

### VISSIM linkmaker
Creates a Measures Estimation Links File using modified data from a VISSIM .inp file, currently set for the I-405 data. The links file is used for the geographic binning of Basic Safety Messages based on their location, by creating a rectangular mapping of the roadway. Since VISSIM defines roadways as a series of small rectangles already, this code pulls the relevant information from the file and reforms it for BSM link assignment.

To run you will need to create and define a VISSIM links file, a VISSIM connectors file, a link widths file, a superlinks file and a full routes file. The VISSIM links file is the information from the Link section of the .inp file with extra whitespace and blank lines removed, the I-405 version is included as i405links.txt. The VISSIM connectors file is the information from the Connector section of the .inp file with extra whitespace and blank lines removed, the I-405 version is included as i405connectors.txt. Link width information can be retrieved from the VISSIM .inp file and the file should be formatted as: VISSIM link id, link width, the I-405 version is included as i405linkwidths.csv. The superlinks and full routes files follow the same format and are used for space mean speed and travel time evaluation respectively. The format for both is: 

```
superlink or route id, superlink or route origin VISSIM link id, next VISSIM link id in superlink or route, ..., superlink or route destination VISSIM link id
```

The I-405 versions are included as i405_fullroutes.csv and i405_superlinks.csv.

Once the input files are created and inserted into the code, it can be run using the command:
```
python vissimlinkmaker.py
```

### bsm_stream
Reads Basic Safety Messages from a TCA output file and turns them into a real time stream of BSMs by time step that is used by each algorithm to replicate real time analysis using offline data. Also performs geographic roadway link, route or superlink assignment based on each BSM's location. Is not run as a standalone.

# Release Notes

### Release 1.0.0 (June 1, 2021)
- Initial release

# License
https://creativecommons.org/licenses/by-sa/4.0/

# Contributions and Contact Information
 - Authors: James O'Hara, Haley Townsend, Syihan Muhammad and Meenakshy Vasudevan, Noblis.
 - Contact Name: James O'Hara	
 - Contact Information: james.ohara@noblis.org

# Acknowledgements
This research is supported by National Cooperative Highway Research Program, Transportation Research Board of The National Academies of Sciences, Engineering, and Medicine. 
