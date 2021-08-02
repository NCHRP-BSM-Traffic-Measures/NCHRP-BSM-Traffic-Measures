import argparse
import numpy as np
import pandas as pd
import os
import math
import time
from bsm_stream import BSM, BSMStream

"""Perform feature extraction from TCA Basic Safety Messages for offline model training."""

def read_links(links_filename):
	"""Read links file and store in reference list"""
	links_list = []
	with open(links_filename) as in_f:
		is_header = True
		for line in in_f:
			if is_header:
				is_header = False
				continue
			row = line.strip().split(',')
			links_list.append(int(row[0]))
	return links_list

def run_incident_feature_extraction(bsm_stream, links_list, output_name):
	"""Read BSM stream and every five seconds group BSMs for that time bucket and for each link and create features: number of BSMs, median and standard deviation speed, 
	median and standard deviation acceleration, median and standard deviation brake pressure and boolean hard braking. 
	Last 100 seconds of BSM is kept so that for each link there are 20 values for each feature each representing a 5 second period.
	 Then output the grouped features to the file."""
	bsms_list = []

	with open(output_name, "w") as out_f:
		out_f.write("Link,VehicleCount,,,,,,,,,,,,,,,,,,,,MedianSpeed,,,,,,,,,,,,,,,,,,,,StdDevSpeed,,,,,,,,,,,,,,,,,,,,MedianAccel,,,,,,,,,,,,,,,,,,,,StdDevAccel,,,,,,,,,,,,,,,,,,,,HardBraking,,,,,,,,,,,,,,,,,,,,MedBrakePressure,,,,,,,,,,,,,,,,,,,,StdDevBrakePressure,,,,,,,,,,,,,,,,,,,,CurrentTime\n")
	with open(output_name, "a") as out_f:
		for tp, bsms in bsm_stream.read():
			if tp % 5 == 0:
				for link in links_list:
					#Append one empty record for each link in case there are no BSMs on that link so that the output file properly reports no data available for the link at that time
					bsms_list.append([tp - 1, np.nan, np.nan, np.nan, np.nan,  0, np.nan, int((tp - 1)/5), link])
				bsms_array = np.array(bsms_list)
				bsms_array = bsms_array[(bsms_array[:,BSM.TimeIndex] > tp - 100)]
				bsms_list = bsms_array.tolist()
				features = pd.DataFrame(bsms_array,columns=['Time','X','Y','Speed', 'Acceleration', 'HardBraking','BrakePressure','MinuteBucket', 'Link'])
				grouped_features = features.groupby(['Link','MinuteBucket']).agg({
					'Time':'count',
					'Speed': ['median','std'],
					'Acceleration': ['median','std'],
					'HardBraking': 'max',
					'BrakePressure':['median','std'],
					})
				grouped_features = grouped_features.unstack()
				grouped_features['CurrentTime'] = tp
				grouped_features.to_csv(path_or_buf = out_f, header=False, mode='a')	
			bsms_list += bsms

def main():
	"""Parse command line arguments, create BSMStream and links list then run incident feature extraction."""
	parser = argparse.ArgumentParser(description='Measures Estimation program for extracting incident detection features from BSMs.')
	parser.add_argument('bsm_filename') # CSV file of Basic Safety Messages
	parser.add_argument('links_filename') # CSV file of links in network
	parser.add_argument('--out', help = 'Output csv file (include .csv)')  
	args = parser.parse_args()

	dir_path = os.path.dirname( os.path.realpath( __file__ ) )

	bsm_stream = BSMStream(filename = args.bsm_filename, links_filename = args.links_filename, add_link = 'link', time_bucket = 5)

	links_list = read_links(args.links_filename)

	if args.out:
	    out_file = dir_path + '/' + args.out

	else:
	    out_file = dir_path + '/bsm_incidentdetection_features.csv'

	run_incident_feature_extraction(bsm_stream, links_list, out_file)


if __name__ == "__main__":
    main()