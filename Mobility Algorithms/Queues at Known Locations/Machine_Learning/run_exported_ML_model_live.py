"""This python script processes 30 seconds of Basic Safety Message data, loads the pre-trained Random Forest classifier,
classifies queue counts for each link for those 30 seconds, derives queue lengths, and outputs the results. 
Sample input and output files can be found in the Supporting_Files folder. 

Test with: python run_exported_ML_model_live.py sample_30secs_BSMs_file.csv sample_exported_model_file.pkl sample_vehicle_length_by_type_file.csv sample_stoplines_file.csv sample_signal_timing_file.csv sample_link_corner_points_file.csv
"""

# load additional libraries for this script (the rest are in queue_fx.py)
import argparse
import numpy as np
import pandas as pd
# ML Model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
import glob
# python module in the Machine_Learning folder
import queue_fx

# define path names
supporting_files_path = "../Supporting_Files/"

# Queue values from definition
QUEUE_START_SPEED = 0.00 
QUEUE_FOLLOWING_SPEED = (10.0 *0.681818) # convert ft/sec to mph
QUEUE_HEADWAY_DISTANCE = 20.0 #ft, in queue definition
QUEUE_DISTANCE_WITHIN_STOP_POINT = 20 #ft 

def import_trained_RF_model(model_filename):
     """Import trained ML model pkl file and store in joblib_model_rf"""
     joblib_model_rf = joblib.load(supporting_files_path + model_filename)
     return joblib_model_rf

def join_features(base_df_name, stopline_avg_df_name, signals_df_name):
    """Join the aggregated features (created from BSMs and supporting files) to the same df.
    Stopline_avg_df is outputted from the create_avg_stoplines_df function."""

    # Rename the link and time columns
    base_df = base_df_name.rename(columns = {'assigned_linkID':'link', 'time_30':'time'}, inplace = False)

    # Bring in a few link-specific features (e.g., # lanes, direction) by joining stopline_avg_df to base_df
    # This will add two features based on link: n_lanes and link_direction
    base_df = base_df.merge(stopline_avg_df_name, how='left', 
                            left_on=['link'], 
                            right_on=['Link_ID'])

    base_df = base_df.drop(['Link_ID','mean_X', 'mean_Y'], axis=1)

    # join signals df columns to base_df
    # issue of link ID 16:15 since accidentally coded as 15:16 in signals file! whoops.
    base_df = base_df.merge(signals_df_name, how='left', 
                            left_on=['time','link'], 
                            right_on=['time_30','Link_ID'])

    base_df = base_df.drop(['Link_ID', 'time_30'], axis=1)
    #print(df_xy.columns)
    return base_df

def load_and_join_queue_count_max_previous(base_df_name):
    """Load the latest queue output file and add a new column for queue_count_max_previous"""
    y_previous_file = glob.glob(supporting_files_path + "*queue_output*" + ".csv")
    df_y_previous = pd.read_csv(y_previous_file[0])
    # join previous 30 secs data to current df
    df_x = base_df_name.merge(df_y_previous, how='left', 
                            left_on=['link'], 
                            right_on=['link'])
    
    df_x = df_x.rename(columns = {'queue_count':'queue_count_max_previous'}, inplace = False)
    df_x = df_x.rename(columns = {'time_x':'time'}, inplace = False)
    df_x.drop(['time_y', 'queue_length'], axis=1, inplace=True)
    return df_x

def derive_queue_len_from_count(df_x_name, predicted_rf_counts):
    """Use the ML classifications of queue count to estimate queue length for each link"""
    # get the index values from X
    idx = df_x_name.index.tolist()
    # replaced veh_len_avg_in_group with avg_veh_len IF Na/0
    # get the veh_len_avg_in_group 
    veh_len_avg_grp = df_x_name.iloc[idx]['veh_len_avg_in_group'].to_numpy()
    pred_q_len = predicted_rf_counts*veh_len_avg_grp
    return pred_q_len

def main():
    """Parse six command line arguments then run 30 secs worth of BSMs through the trained ML model and output queue counts and lengths."""
    parser = argparse.ArgumentParser(description='Script to output trained Supervised ML Classifier (Random Forest), for offline purposes.')
    parser.add_argument('BSMs_30secs_filename') # CSV file with 30 seconds worth of BSMs to "simulate" running live, in real-time
    parser.add_argument('model_filename') # PKL file exported from train_test_export_ML_model_offline.py
    parser.add_argument('veh_lengths_filename') # supporting CSV file of vehicle lengths in ft by type
    parser.add_argument('stoplines_filename') # supporting CSV file of stop line X,Y coordinates for each link and lane
    parser.add_argument('signal_timing_filename') # supporting CSV file of signal timing for each link every 30 seconds
    parser.add_argument('link_corners_filename') # supporting CSV file with link corner points for BSM assignment
    args = parser.parse_args()

    # read in the six files
    df = queue_fx.read_BSMs_file(args.BSMs_30secs_filename) # 30 seconds worth of BSMs
    joblib_model_rf = import_trained_RF_model(args.model_filename) # trained RF model
    veh_len_df = queue_fx.read_veh_lengths_file(args.veh_lengths_filename)
    stoplines_df = queue_fx.read_stoplines_file(args.stoplines_filename)
    signals_df = queue_fx.read_signal_timing_file(args.signal_timing_filename)
    link_points_df = queue_fx.read_link_corners_file(args.link_corners_filename)

    # create the avg stoplines X,Y df
    stopline_avg_df = queue_fx.create_avg_stoplines_df(stoplines_df)

    # format the time
    df.transtime = df.transtime.apply(queue_fx.format_result)
    # create a new column that assigns BSM to 30 second time interval
    df['transtime_30sec'] = df['transtime'].dt.round('30S')

    # Assign BSMs to links
    df = queue_fx.assign_BSMs_to_roadway_links(df, link_points_df)

    # join columns from veh len to main BSMs df 
    df = queue_fx.join_veh_len_to_BSM_df(df, veh_len_df)

    # Engineer the aggregated BSM features by assigned link and 30 secs
    base_df = queue_fx.feature_engineering(df, stopline_avg_df)

    # Join all features to base_df
    base_df = join_features(base_df, stopline_avg_df, signals_df)

    # Add a column to the features for the previous time step's queue count for each link
    # find the latest queue_output.csv file and load that data
    df_x = load_and_join_queue_count_max_previous(base_df)

    # Handle any missing values
    df_x = queue_fx.label_encode_categorical_features(queue_fx.handle_missing_data(df_x, df))
    # replace "time" with "time_float"
    df_x.drop(['time'], axis=1, inplace = True)

    # scale the features X for classification
    X = queue_fx.feature_scaling_X(df_x)

    # output model queue count classifications, one for each link
    link_names_lst = base_df['link'].tolist()
    predicted_rf = joblib_model_rf.predict(X).tolist()
    # output derived queue lengths from queue counts
    predicted_queue_lens = (derive_queue_len_from_count(df_x, predicted_rf)).tolist()
    time_lst = base_df['time'].tolist()
    headers = ['time', 'link', 'queue_count', 'queue_len']

    # output the queue counts and lengths neatly in a df to standard output (stdout)
    output_df = pd.DataFrame(columns = headers)
    output_df['time'] = time_lst
    output_df['link'] = link_names_lst
    output_df['queue_count'] = predicted_rf
    output_df['queue_len'] = predicted_queue_lens
    print(output_df)

    output_time = time_lst[0].replace(':', '')
    output_df.to_csv(supporting_files_path + output_time + "_queue_output.csv")


if __name__ == "__main__":
    main()