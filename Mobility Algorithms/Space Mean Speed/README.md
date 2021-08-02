# Description
The Space Mean Speed application estimates space mean speed for defined roadway segments, called Superlinks, of a given arterial or freeway network using Basic Safety Messages. The Space Mean Speed application is intended to track speeds in real time.

# Usage
The Space Mean Speed application requires five command line arguments and imports bsm_stream.py. The required arguments are: a csv file of TCA Basic Safety Messages for evaluation, a csv full superlinks file, a csv file of links created by linkmaker.py, a csv file of link endpoints which includes a link id number and two X,Y points one for the origin of the link and one for the destination, and a filename to save timing output for performance evaluation. The user can also optionally specify a filename to output the space mean speed values to, or a default filename will be used.

The full superlinks file defines the roadway segments that Space Mean Speed is being evaluated over, built up from the smaller links defined by linkmaker.py. It is recommended that superlinks be nearly uniform in length, this project used a distance of one mile, although most superlinks were not exactly a mile because of the variable length of the smaller links. The full superlinks file is formatted as:

```
superlink or route id, superlink or route origin VISSIM link id, next VISSIM link id in superlink or route, ..., superlink or route destination VISSIM link id
```

To run the Space Mean Speed application use the following command:
```
python SpaceMeanSpeed_BSM.py [bsm_filename | REQUIRED] [fullsuperlinks_filename | REQUIRED] [links_filename | REQUIRED] [link_endpoints_filename | REQUIRED] [timing_output_filename | REQUIRED] --out [output_filename | OPTIONAL] 
```

For license and contributor information see the main Mobility Algorithms README.