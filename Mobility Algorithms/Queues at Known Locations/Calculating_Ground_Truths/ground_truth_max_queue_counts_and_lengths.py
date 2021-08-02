"""This py script calculates the queue count and length ground truths from vehicle trajectory files
at defined intersections by link, lane, and time point (every second). Then, it takes the max queue count and length
for each link (across all lanes) and 30 second time interval and outputs to csv file.
The outputted csv file will be used as the supervised machine learning labels for model training.

This was originally coded from simulation data for Flatbush Ave in NYC. 

**Definition of vehicle in queue:** A vehicle is in a queue when it is either stopped or traveling at a speed less than 10 ft/s (3 m/s) 
and is approaching another queued vehicle at headway of less than 20 ft.

Units used for this file are summarized below. 
speed: mph
acceleration: fpss
time: float starting at simulation start time and ending at simulation end time (7-10 AM)
X,Y: coordinates corresponding to center of FRONT bumper of vehicle in feet
vehicle length: in feet

Test with: python ground_truth_max_queue_counts_and_lengths.py sample_trajectories_file.csv sample_vehicle_length_by_type_file.csv sample_stoplines_file.csv
"""

# Load libraries
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import csv
import time
from datetime import timedelta

# define path names
supporting_files_path = "../Supporting_Files/"

# Ground Truth queue values from definition
QUEUE_START_SPEED = 0.00 
QUEUE_FOLLOWING_SPEED =  (10.0 *0.681818) #converted ft/sec to mph since speed is in mph
QUEUE_HEADWAY_DISTANCE =  20.0 #ft, in queue definition
QUEUE_DISTANCE_WITHIN_STOP_POINT = 20.0 #ft

def read_trajectories_file(trajectories_filename):
    """Read vehicle trajectories csv file and store in pandas dataframe (df)"""
    df = pd.read_csv(supporting_files_path + trajectories_filename)
    return df

def read_veh_lengths_file(veh_lengths_filename):
    """Read vehicle lengths by type csv file and store in pandas dataframe (veh_len_df)"""
    veh_len_df = pd.read_csv(supporting_files_path + veh_lengths_filename)
    return veh_len_df

def read_stoplines_file(stoplines_filename):
    """Read stop lines by lane and link csv file and store in pandas dataframe (stopline_df)"""
    stopline_df = pd.read_csv(supporting_files_path + stoplines_filename)
    return stopline_df

def distance_between(origin_x, origin_y, destination_x, destination_y):
    """Takes origin X,Y and destination X,Y and returns the distance between them"""
    return ((origin_x - destination_x)**2 + (origin_y - destination_y)**2)**.5

def format_result(result):
    """Format result of simulation time float into datetime, 
    add 7 hours for Flatbush simulation data because 0.0 corresponds to 7:00 AM,
    output is time in HH:MM:SS.microseconds"""
    seconds = int(result)
    microseconds = (result * 1000000) % 1000000
    output = timedelta(0, seconds, microseconds) + timedelta(hours=7)
    return output

def take_one_traj_per_second(df_name):
    """There are 10 vehicle trajectories every second. 
    This takes only 1 observation per second to speed up processing time without losing precision.
    We would expect the ground truth queues to be the same across tenths of second."""
    dfs = df_name.loc[np.isclose((df_name.time % 1).values, .6)]
    return dfs

def join_cols_to_traj_df(dfs_name, veh_len_df_name, stoplines_df_name):
    """Join the vehicle length and stopline X,Y coordinates to the main veh trajectories df """
    # Join vehicle length column to main veh traj df
    dfsv = dfs_name.merge(veh_len_df_name[['Type_ID','Length (ft)']], how='left', left_on='type', right_on='Type_ID')
    dfsv = dfsv.drop(['Type_ID'], axis=1)
    # Join stopline coordinates to main veh traj df
    dfsvs = pd.merge(dfsv, stoplines_df_name, how='left', left_on=['link', 'lane'], right_on=['Link_ID', 'Lane'])
    dfsvs = dfsvs.drop(['Link_ID', 'Lane'], axis=1)
    # if the links/lanes are not in the stoplines_df_file, then remove from main df
    dfsvs = dfsvs.dropna()
    return dfsvs

def run_ground_truth_queue_count_len(dfsvs_name, stoplines_df_name):
    """Calculate the ground truth queue count (# vehicles) and queue length (in feet) every second for every link and lane """
    # convert to numpy array
    dfs_arr = dfsvs_name.to_numpy()
    # get the lanes, links and time points to iterate over
    lanes = dfs_arr[:,4]
    #print(np.unique(lanes), "unique lanes")
    # 3 for links
    links = dfs_arr[:,3]
    #print(np.unique(links), "unique links")
    # 0 for times
    times = dfs_arr[:,0]
    #print(np.unique(times), "unique times")

    queue_output = []

    # iterate over each second
    for time in np.unique(times):
        # iterate over each link. Suggest starting with the smallest link for testing
        for link in np.unique(links):
            lanes_in_link = stoplines_df_name.loc[stoplines_df_name['Link_ID']==link, 'Lane'].tolist()
            for lane in lanes_in_link: 
                rows = dfs_arr[(dfs_arr[:,4]==lane) & (dfs_arr[:,3]==link) & (dfs_arr[:,0]==time)]
                # for one time point, one link, one lane worth of data
                if len(rows) > 0:
                    # initialize queue count to 0
                    queue_count = 0

                    # initialize queue length to 0
                    queue_len = 0
                    first_veh = None
                    last_veh = None
                    leader_veh = None
                    veh_count = 0

                    # initialize empty list of queued vehicle IDs
                    queued_vehicles = []
                    time_step_count = 0

                    # Starting with the vehicle closest to the end of the link
                    for veh in rows:
                        veh_count +=1

                        # if queue_count is 0, looking at first vehicle
                        if queue_count == 0:

                            # Look for a motionless vehicle within range of stopline
                            if ((veh[8] == QUEUE_START_SPEED) and
                                (veh[-1] <= QUEUE_DISTANCE_WITHIN_STOP_POINT)):
                                queue_count = 1
                                first_veh = last_veh = leader_veh = veh
                                # queue_len is distance to stopbar plus length of veh (assuming X,Y coords correspond to FRONT of vehicle)
                                # this will preserve against angle and link direction
                                # veh[-1] is distance to stopbar from front of veh
                                queue_len = veh[-1] + veh[11]
                                # append the ID, veh[1]
                                queued_vehicles.append(veh[1])

                        # Cycle through remaining vehicles on the link-lane, 
                        # determining if they are in queue behind motionless vehicle 
                        # distance between front of second vehicle and back of first vehicle and <= 20 ft
                        # veh[5] is X and veh[6] is Y, leader_veh[11] is its length
                        # Please note: this distance is an approximation.
                        elif ((queue_count>0) and (veh[8]<= QUEUE_FOLLOWING_SPEED) and ((distance_between(leader_veh[5], leader_veh[6], veh[5], veh[6])-leader_veh[11])<= QUEUE_HEADWAY_DISTANCE)):
                            queue_count +=1
                            last_veh = leader_veh = veh
                            # append vehicle id (veh[1]) to queued_vehicles list
                            queued_vehicles.append(veh[1])

                        elif queue_count >0:
                            # queue_len same as before, last veh distance to stop plus it's length
                            queue_len = last_veh[-1] + last_veh[12]
                            # in queue output include: time, link, lane, queue count and queue len
                            queue_output.append(([veh[0], veh[3], veh[4], queue_count, queue_len]))
                            break

                        # This is the last vehicle on the roadway link, print queue count and length
                        if len(rows) == veh_count:
                            if queue_count >0:
                                queue_len = last_veh[-1] + last_veh[12]
                                last_link = last_veh[3]
                                last_lane = last_veh[4]
                            else:
                                last_link = last_lane = '0'
                            queue_output.append(([veh[0], veh[3], veh[4], queue_count, queue_len]))

    #print(len(queue_output), "length of queue output") 
    headers = ['time', 'link', 'lane', 'queue_count', 'queue_len']
    queue_df = pd.DataFrame(queue_output, columns = headers)
    #print(queue_df.head())
    return queue_df

def find_max_queues_over_30_secs_per_link(queue_df_name):
    """Find the max queue count and length for each link across all lanes over 30 seconds """
    queue_df_name.time = queue_df_name.time.apply(format_result)
    # Round to nearest 30 seconds
    queue_df_name['time_30'] = queue_df_name['time'].dt.round('30S')
    # create the groupby 30 secs and link
    gb_main = queue_df_name.groupby(['time_30','link'])
    # getting max queue count and length for each 30 sec time and link grouping (across all lanes in link)
    # create an aggregated dataframe with maxes
    queue_df_max = gb_main.max()[['queue_count', 'queue_len']].add_suffix('_max').reset_index('time_30')
    #print("shape of 30sec-link grouped df with maxes:", queue_df_max.shape)
    #print(queue_df_max.head())
    return queue_df_max

def write_max_queues_to_csv(queue_df_max_name, output_file_name):
    """Write the max queues dataframe to a csv file"""
    queue_df_max_name.to_csv(output_file_name)
    
def main():
    """Parse command line arguments [trajectories_filename | REQUIRED] [veh_lengths_filename | REQUIRED] [stoplines_filename | REQUIRED] then run ground truth queue calculation (may take a while)."""
    parser = argparse.ArgumentParser(description='Script to output ground truth queue counts and lengths from vehicle trajectories.')
    parser.add_argument('trajectories_filename') # CSV file of vehicle trajectories
    parser.add_argument('veh_lengths_filename') # CSV file of vehicle lengths in ft by type
    parser.add_argument('stoplines_filename') # CSV file of stop line X,Y coordinates for each link and lane
    parser.add_argument('--out', help = 'Output csv file (include .csv)')  
    args = parser.parse_args()

    # read in the three files
    df = read_trajectories_file(args.trajectories_filename)
    veh_len_df = read_veh_lengths_file(args.veh_lengths_filename)
    stoplines_df = read_stoplines_file(args.stoplines_filename)

    # take one obs per second
    dfs = take_one_traj_per_second(df)

    # join the three dataframes
    dfsvs = join_cols_to_traj_df(dfs, veh_len_df, stoplines_df)
    
    # calculate the distance to the stopbar of each vehicle
    dfsvs['distance_to_stop'] = distance_between(dfsvs.x, dfsvs.y, dfsvs.stopline_X, dfsvs.stopline_Y)
    # make sure values sorted in order by closest distance to stop
    dfsvs = dfsvs.sort_values(by='distance_to_stop')

    # run the queue ground truth code for every second, link, and lane
    queue_df = run_ground_truth_queue_count_len(dfsvs, stoplines_df)

    # find the MAX ground truth queue counts/lengths per link and 30 secs
    queue_df_max = find_max_queues_over_30_secs_per_link(queue_df)

    if args.out:
        output_file = args.out
    else:
        output_file = "max_ground_truth_queues.csv"

    write_max_queues_to_csv(queue_df_max, output_file)


if __name__ == "__main__":
    main()


