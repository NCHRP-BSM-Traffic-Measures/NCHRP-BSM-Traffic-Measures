import argparse
import numpy as np
import pandas as pd
import math
import time
import tensorflow as tf

from bsm_stream import BSM, BSMStream

"""Single pipeline every five seconds performing feature extraction from TCA Basic Safety Messages, using saved model to detect an incident and tracking detected incidents
to verify incident conditions if the same link is positive six or more consecutive times (30 seconds or more)."""

def read_links(links_filename):
    """Read links file and store in reference list"""
    links_list = []
    link_tracker = {}
    with open(links_filename) as in_f:
        is_header = True
        for line in in_f:
            if is_header:
                is_header = False
                continue
            row = line.strip().split(',')
            link = int(row[0])
            links_list.append(link)
            link_tracker[link] = {'count': 0, 
                                  'start_time': None,
                                  'last_tp_seen': 0}
    return links_list, link_tracker

def run_incident_detection(bsm_stream, links_list, link_tracker, output_name, timing_output, model_filename):
    """Read BSMs from stream, group by 5 second period and link to extract features: number of BSMs, median and standard deviation speed, 
    median and standard deviation acceleration, median and standard deviation brake pressure and boolean hard braking. 
    Last 100 seconds of BSM is kept so that for each link there are 20 values for each feature each representing a 5 second period. Then detect 
    incidents using data for current timestep, and track consecutive detection events by link. If a link has 6 or more consecutive detections 
    (30 seconds or more of incident), then output it to file."""
    bsms_list = []
    timing = []
    model = tf.keras.models.load_model(model_filename)

    with open(timing_output, "w") as out_f:
        out_f.write("Time Period,Execution Time\n")

    with open(output_name, "w") as out_f:
        out_f.write("Link, Detection Time, Verification Time\n")

    for tp, bsms in bsm_stream.read():
        timer = time.time()
        bsms_list += bsms
        if tp % 5 == 0:
            for link in links_list:
                bsms_list.append([tp, 0, 0, 0, 0,  0, 0, int(tp/5), link])
            if tp < 100:
                continue
            bsms_array = np.array(bsms_list)
            bsms_array = bsms_array[(bsms_array[:,7] > int(tp/5) - 20)]
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
            grouped_features.columns = grouped_features.columns.get_level_values(0)
            grouped_features.fillna(value = 0, inplace = True)
            try:
                prediction = model.predict(grouped_features, batch_size=1)
            except ValueError as e:
                print(tp)
                print(grouped_features)
                print(e)
                raise
            positive_links = grouped_features[prediction >= 0.5].index.tolist()
                
            for link in positive_links: 
                if link_tracker[link]['count'] == 0 or tp - link_tracker[link]['last_tp_seen'] != 5:
                    link_tracker[link]['count'] = 0
                    link_tracker[link]['start_time'] = tp
                link_tracker[link]['count'] += 1
                if link_tracker[link]['count'] > 6:
                    with open(output_name, "a") as out_f:
                        out_f.write("{},{},{}\n".format(link,link_tracker[link]['start_time'],tp))
                link_tracker[link]['last_tp_seen'] = tp

            with open(timing_output, "a") as out_f:
                out_f.write("{},{}\n".format(tp, time.time() - timer))

def main():
    """Parse command line arguments for input and output filenames and start incident detection."""
    parser = argparse.ArgumentParser(description='ME program for reading in BSMs and producing Travel Time values')
    parser.add_argument('bsm_filename') # CSV file of Basic Safety Messages
    parser.add_argument('links_filename') # CSV file of links in network
    parser.add_argument('model_filename') # Name of saved neural network model created using buildmodel.py
    parser.add_argument('timing_filename') # Timing output filename
    parser.add_argument('--out', help = 'Output csv file (include .csv)')  
    args = parser.parse_args()

    bsm_stream = BSMStream(args.bsm_filename, args.links_filename)

    links_list, link_tracker = read_links(args.links_filename)

    if args.out:
        out_file = args.out

    else:
        out_file = 'verified_incidents.csv'

    run_incident_detection(bsm_stream, links_list, link_tracker, out_file, args.timing_filename, args.model_filename)

if __name__ == "__main__":
    main()