# Description
The Travel Time application estimates travel times for defined route segments for a given arterial or freeway network using Basic Safety Messages. The Travel Time application is intended to track travel times in real time.

# Usage
The Travel Time application requires five command line arguments and imports bsm_stream.py. The required arguments are: a csv file of TCA Basic Safety Messages for evaluation, a csv full routes file, a csv file of links created by linkmaker.py, a csv file of link endpoints which includes a link id number and two X,Y points one for the origin of the link and one for the destination, and a filename to save timing output for performance evaluation. The user can also optionally specify a filename to output the travel time values to, or a default filename will be used.

The full routes file defines the roadway segments that travel time is being evaluated over, built up from the smaller links defined by linkmaker.py. Routes can be variable in length between points of interest, for this project routes were defined as stretches of freeway between major interchanges. The full routes file is formatted as:

```
route or route id, route or route origin VISSIM link id, next VISSIM link id in route or route, ..., route or route destination VISSIM link id
```

To run the Travel Time application use the following command:
```
python TravelTime_BSM.py [bsm_filename | REQUIRED] [fullroutes_filename | REQUIRED] [links_filename | REQUIRED] [link_endpoints_filename | REQUIRED] [timing_output_filename | REQUIRED] --out [output_filename | OPTIONAL] 
```

For license and contributor information see the main Mobility Algorithms README.