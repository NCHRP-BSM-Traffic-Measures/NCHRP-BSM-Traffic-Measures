"""This python script contains the libraries and functions needed to run the other two ML py scripts in this folder.
"""

# load libraries necessary for both offline and live ML scripts
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime
import math
# ML Model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import imblearn
#from imblearn.over_sampling import RandomOverSampler #optional if you want to try oversampling imbalanced data
import joblib

# define path names
supporting_files_path = "../Supporting_Files/"

# Queue values from definition
QUEUE_START_SPEED = 0.00 
QUEUE_FOLLOWING_SPEED = (10.0 *0.681818) # convert ft/sec to mph
QUEUE_HEADWAY_DISTANCE = 20.0 #ft, in queue definition
QUEUE_DISTANCE_WITHIN_STOP_POINT = 20 #ft 

def read_BSMs_file(BSMs_X_filename):
    """Read BSMs csv file and store in pandas dataframe (df)"""
    df = pd.read_csv(supporting_files_path + BSMs_X_filename)
    #print(df.head())
    return df

def read_max_queues_Y_file(max_queues_Y_filename):
    """Read max queues by link and 30 secs csv file and store in pandas dataframe (y_df)"""
    y_df = pd.read_csv(supporting_files_path + max_queues_Y_filename)
    # clean up the time column so it only has hour, mins, seconds
    y_df['time'] = y_df['time_30'].str[7:15]
    y_df.drop(['time_30'], axis=1, inplace=True)
    #print(y_df.head())
    return y_df

def read_veh_lengths_file(veh_lengths_filename):
    """Read vehicle lengths by type csv file and store in pandas dataframe (veh_len_df)"""
    veh_len_df = pd.read_csv(supporting_files_path + veh_lengths_filename)
    #print(veh_len_df.head())
    return veh_len_df

def read_stoplines_file(stoplines_filename):
    """Read stop lines by lane and link csv file and store in pandas dataframe (stopline_df)"""
    stoplines_df = pd.read_csv(supporting_files_path + stoplines_filename)
    #print(stopline_df.head())
    return stoplines_df

def read_signal_timing_file(signal_timing_filename):
    """Read signal timing (% time green, amber, red) by link and 30 secs csv file and store in pandas dataframe (signals_df)"""
    signals_df = pd.read_csv(supporting_files_path + signal_timing_filename)
    # reformat trantime_30sec to string, convert timedelta to string
    signals_df['time_30'] = signals_df['transtime_30sec'].astype(str).str[7:15]
    signals_df.drop('transtime_30sec',axis='columns', inplace=True)
    #print(signals_df.head())
    return signals_df

def read_link_corners_file(link_corner_points_filename):
    """Read link corner points (X,Y) csv file and store in pandas dataframe (link_points_df)"""
    link_points_df = pd.read_csv(supporting_files_path + link_corner_points_filename)
    #print(link_points_df.head())
    return link_points_df

def create_avg_stoplines_df(stoplines_df_name):
    """Create an aggregated stoplines_avg_df with the average stopline X, Y for each link and number of lanes"""
    stopline_avg_df = stoplines_df_name.groupby('Link_ID')['stopline_X'].mean().reset_index(name='mean_X')
    stopline_avg_df['mean_Y'] = stoplines_df_name.groupby('Link_ID')['stopline_Y'].mean().reset_index(name='mean_Y').iloc[:,1]
    stopline_avg_df['n_lanes'] = stoplines_df_name.groupby('Link_ID')['Lane'].count().reset_index().iloc[:,1]
    stopline_avg_df['link_direction'] = stoplines_df_name.groupby('Link_ID')['Link_direction'].first().reset_index().iloc[:,1]
    #print(stopline_avg_df.head())
    return stopline_avg_df

def assign_BSMs_to_roadway_links(df_BSM_name, link_points_df_name):
    """This function assigns each BSM to its roadway link based on the link corner points. This assignment is an approximation.
    Please note: You will likely need to update the latter part of this function to account for your roadway geometry. It is currently designed to separate NB and SB links and filter BSMs based on their headings.
    Please note: This could take a while to run on all BSMs."""
    # Initialize a new empty column for assigned_LinkID
    df_BSM_name['assigned_linkID'] = np.nan

    # save the links in an array/list
    links = link_points_df_name.Link.unique()

    # find the min and max X,Ys from the four corner points for each link
    for idx, link in enumerate(links): 
        min_x = link_points_df_name.loc[link_points_df_name['Link']==link, ['X_lower_L','X_lower_R','X_upper_L', 'X_upper_R']].min().min()
        max_x = link_points_df_name.loc[link_points_df_name['Link']==link, ['X_lower_L','X_lower_R','X_upper_L', 'X_upper_R']].max().max()
        min_y = link_points_df_name.loc[link_points_df_name['Link']==link, ['Y_lower_L','Y_lower_R','Y_upper_L', 'Y_upper_R']].min().min()
        max_y = link_points_df_name.loc[link_points_df_name['Link']==link, ['Y_lower_L','Y_lower_R','Y_upper_L', 'Y_upper_R']].max().max()

        mask = ((df_BSM_name.X>=min_x) & (df_BSM_name.X<=max_x) & (df_BSM_name.Y>=min_y) & (df_BSM_name.Y<=max_y))

        # BSMs that fall in that link mask are assigned to that link in a new column
        df_BSM_name.loc[mask,'assigned_linkID'] = link

    # if assigned_linkID is NA then drop
    df = df_BSM_name.loc[df_BSM_name['assigned_linkID'].notna()]
    #print(df.shape, "shape of bsms assigned to links before heading filter")

    # **You will need to edit this section for your specific network geometry
    # figure out whether NB or SB for each of the six Flatbush links
    # then have two filters: one for NB and one for SB links
    # if BSM heading is NOT in the range for NB then drop
    # if BSM is NOT in the range for SB then drop
    # for TCA BSM heading, 0 degrees is due N
    SB_links = ['5677903#1', '221723366', '221731899#2.168']
    NB_links = ['349154287#5', '-139919720#0.114', '-139919720#0', '-23334090#0']

    # create a filter mask for South links
    maskS = ((df['assigned_linkID'].isin(SB_links)) & (df['Heading']>90) & (df['Heading']<270))

    # create a filter mask for North links
    maskN = ((df['assigned_linkID'].isin(NB_links)) & ((df['Heading']>270) | (df['Heading']<90))) 

    # combine the S and N masks
    df = df.loc[maskS | maskN]
    # Add this step to combine links -139919720#0.114 and -139919720#0 
    df['assigned_linkID'].replace("-139919720#0.114", "-139919720#0", inplace=True)
    #print(df.shape, "shape of corrected bsms")
    return df

def format_result(result):
    """Format result of simulation time float into datetime, 
    add 7 hours for Flatbush simulation data because 0.0 corresponds to 7:00 AM,
    output is time in HH:MM:SS.microseconds"""
    seconds = int(result)
    microseconds = (result * 1000000) % 1000000
    output = timedelta(0, seconds, microseconds) + timedelta(hours=7)
    return output

def distance_between(origin_x, origin_y, destination_x, destination_y):
    """Calculate the distance between two points. """
    return ((origin_x - destination_x)**2 + (origin_y - destination_y)**2)**.5

def min_dist_to_avg_stopbar(group_row, stopline_avg_df_name):
    """Calculate the distance between each grouping min X,Y and the avg stopline X,Y for that link.
    This distance is an approximation and depends on the direction and curvature of the links."""
    row_stop_X, row_stop_Y = stopline_avg_df_name.loc[stopline_avg_df_name['Link_ID']==group_row['assigned_linkID'],['mean_X','mean_Y']].values[0]
    direction = stopline_avg_df_name.loc[stopline_avg_df_name['Link_ID']==group_row['assigned_linkID'],['link_direction']].values[0]
    # we have to do the opposite for N direction links
    if direction =='N':
        row_dist = distance_between(group_row['max_X'], group_row['max_Y'], row_stop_X, row_stop_Y)
    else:
        row_dist = distance_between(group_row['min_X'], group_row['min_Y'], row_stop_X, row_stop_Y)
    return(row_dist)

def max_dist_to_avg_stopbar(group_row, stopline_avg_df_name):
    """Calculate the max distance between each grouping max X,Y and the avg stopline X,Y for that link."""
    row_stop_X, row_stop_Y = stopline_avg_df_name.loc[stopline_avg_df_name['Link_ID']==group_row['assigned_linkID'],['mean_X','mean_Y']].values[0]
    direction = stopline_avg_df_name.loc[stopline_avg_df_name['Link_ID']==group_row['assigned_linkID'],['link_direction']].values[0]
     # Do the opposite for N direction links
    if direction =='N':
        row_dist = distance_between(group_row['min_X'], group_row['min_Y'], row_stop_X, row_stop_Y)
    else:
        row_dist = distance_between(group_row['max_X'], group_row['max_Y'], row_stop_X, row_stop_Y)

    return(row_dist)

def to_seconds(s):
    """Convert the 30 second time string to a float."""
    hr, minute, sec = [float(x) for x in s.split(':')]
    total_seconds = hr*3600 + minute*60 + sec
    return total_seconds

def join_veh_len_to_BSM_df(df_BSM_name, veh_len_df_name):
    """Join vehicle length column to main BSM df."""
    df = df_BSM_name.merge(veh_len_df_name[['Type_ID','Length (ft)']], how='left', left_on='Type', right_on='Type_ID')
    df = df.drop(['Type_ID'], axis=1)
    #print(df.head())
    return df

def feature_engineering(df_BSM_name, stopline_avg_df_name):
    """Create grouped df with new aggregated features based on BSMs."""
    # Our main group by object (road link, 30 second time chunk)
    gb_main = df_BSM_name.groupby(['transtime_30sec','assigned_linkID'])[['BSM_tmp_ID']].count()
    # creating the base aggregated DF to add columns to
    base_df = gb_main.add_suffix('_Count').reset_index()
    gb = df_BSM_name.groupby(['transtime_30sec','assigned_linkID'])

    # get the value of the average vehicle length across all BSMs
    avg_veh_len = df_BSM_name["Length (ft)"].mean()
    median_veh_len = df_BSM_name["Length (ft)"].median()

    # count # of BSMs in 30 sec-link grouping with 0 speed
    base_df['num_BSMs_0speed'] = gb['Speed'].apply(lambda x: (x==0).sum()).reset_index(name='sum').iloc[:,2]

    # number of BSMs with speed between 0 and QUEUE_FOLLOWING_SPEED
    base_df['num_BSMs_0_to_following_speed'] = gb['Speed'].apply(lambda x: ((x>0) & (x<=QUEUE_FOLLOWING_SPEED)).sum()).reset_index(name='sum').iloc[:,2]

    # number of BSMs greater than QUEUE_FOLLOWING_SPEED
    base_df['num_BSMs_above_following_speed'] = gb['Speed'].apply(lambda x: (x>QUEUE_FOLLOWING_SPEED).sum()).reset_index(name='sum').iloc[:,2]

    # number of BSMs with vehicle length above average
    base_df['num_BSMs_len_above_avg'] = gb["Length (ft)"].apply(lambda x: (x>avg_veh_len).sum()).reset_index(name='sum').iloc[:,2]

    # number of BSMs with vehicle length equal to or below average
    base_df['num_BSMs_len_below_avg'] = gb["Length (ft)"].apply(lambda x: (x<=avg_veh_len).sum()).reset_index(name='sum').iloc[:,2]

    # get AVG vehicle length per grouping
    base_df['veh_len_avg_in_group'] = gb["Length (ft)"].mean().reset_index(name='sum').iloc[:,2]

    # get the MEDIAN vehicle length per grouping
    base_df['veh_len_med_in_group'] = gb["Length (ft)"].median().reset_index(name='sum').iloc[:,2]

    # speed standard deviation 
    base_df['speed_stddev'] = gb["Speed"].std().reset_index().iloc[:,2]

    # max speed in grouping
    base_df['speed_max'] = gb["Speed"].max().reset_index().iloc[:,2]

    # acceleration standard deviation
    # could be called "Instant_Acceleration" instead of "Avg_Acceleration"
    base_df['accel_stddev'] = gb["Avg_Acceleration"].std().reset_index().iloc[:,2]

    # number of BSMs with negative acceleration
    base_df['num_BSMs_neg_accel'] = gb["Avg_Acceleration"].apply(lambda x: (x<=0).sum()).reset_index(name='sum').iloc[:,2]

    # Max X per group
    base_df['max_X'] = gb["X"].max().reset_index(name='max').iloc[:,2]

    # Max Y per group
    base_df['max_Y'] = gb["Y"].max().reset_index(name='max').iloc[:,2]

    # Min X per group
    base_df['min_X'] = gb["X"].min().reset_index(name='max').iloc[:,2]

    # Min Y per group
    base_df['min_Y'] = gb["Y"].min().reset_index(name='max').iloc[:,2]

    # distance between Max X,Y and Min X,Y to indicate how far apart the BSMs are
    base_df['max_distance_between_BSMs'] = base_df.apply(lambda row: distance_between(row['max_X'],row['max_Y'],row['min_X'],row['min_Y']), axis=1)

    # direction matters here
    base_df['min_dist_to_stopbar'] = base_df.apply(lambda row: min_dist_to_avg_stopbar(row, stopline_avg_df_name), axis=1)

    base_df['max_dist_to_stopbar'] = base_df.apply(lambda row: max_dist_to_avg_stopbar(row, stopline_avg_df_name), axis=1)                                    

    # Create frequency of braking features
    base_df['num_braking'] = gb["brakeStatus"].apply(lambda x: (x>0).sum()).reset_index(name='sum').iloc[:,2]
    base_df['num_braking_hard'] = gb["hardBraking"].apply(lambda x: (x>0).sum()).reset_index(name='sum').iloc[:,2]
    # change it to 1/0 yes/no hard braking occurred
    base_df['hard_braking'] = 0
    mask_hardBrake = (base_df['num_braking_hard']>0)
    base_df.loc[mask_hardBrake,'hard_braking'] = 1

    # convert timedelta to string
    base_df['time_30'] = base_df['transtime_30sec'].astype(str).str[7:15]

    # avoid dropping for creating the queue_count column for previous 30 secs
    base_df.drop('transtime_30sec',axis='columns', inplace=True)
    return base_df

def handle_missing_data(df_xy_name, df_BSM_name):
    """Since python's scikit-learn will not accept rows with NA, this function replaces NAs with 0 for most columns except the veh len avg and median.
    Assumption: rows with NA for the BSM features did not see any BSMs sent from a CV in that link and time period. 
    Please note: Handling missing data is more an art than a science! You may want to handle NAs differently in your case."""
    # explore missingness first
    #print(df_xy_name.isna().sum(), "total NA")

    ## Handling NaN rows in df_xy
    #replace NaN with 0
    df_xy = df_xy_name.fillna(0)

    # get the value of the average vehicle length across all BSMs
    avg_veh_len = df_BSM_name["Length (ft)"].mean()
    median_veh_len = df_BSM_name["Length (ft)"].median()

    # replace 0 values for veh_len_avg_in_group with the average over all BSMs
    mask_veh_avg = (df_xy['veh_len_avg_in_group']==0)
    df_xy.loc[mask_veh_avg,'veh_len_avg_in_group'] = avg_veh_len

    # replace 0 values for veh_len_med_in_group with the median over all BSMs
    mask_veh_med = (df_xy['veh_len_med_in_group']==0)
    df_xy.loc[mask_veh_med,'veh_len_med_in_group'] = median_veh_len

    return df_xy

def label_encode_categorical_features(df_xy_name):
    """Label encode categorical features for Random Forest.
    Please note: encoding is also more of an art than a science. You could try different methods.""" 
    # label encode the link IDs
    df_xy_name["link"] = df_xy_name["link"].astype('category')
    df_xy_name["link_encoded"] = df_xy_name["link"].cat.codes

    # now drop the original 'link' column (you don't need it anymore)
    df_xy_name.drop(['link'],axis=1, inplace=True)

    # label encode the roadway direction
    df_xy_name["link_direction"] = df_xy_name["link_direction"].astype('category')
    df_xy_name["link_direction_encoded"] = df_xy_name["link_direction"].cat.codes
    # now drop the original 'link_direction' column (you don't need it anymore)
    df_xy_name.drop(['link_direction'],axis=1, inplace=True)

    # needs to be numeric to work in sklearn
    df_xy_name['time_float'] = df_xy_name['time'].apply(to_seconds)

    return df_xy_name

def feature_scaling_X(X_name):
    """Minmax scale the features X. 
    Please note: Feature scaling is not necessarily required for a Random Forest classifier, but other classifiers require it."""
    min_max_scaler = preprocessing.MinMaxScaler()
    #Minmax scaler
    X_minmax = min_max_scaler.fit_transform(X_name)
    return X_minmax
