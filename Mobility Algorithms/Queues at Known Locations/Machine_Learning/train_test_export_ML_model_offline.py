"""This python script trains supervised machine learning Random Forest models on the complete data provided, 
ouptputs performance metrics, and exports the trained Random Forest to a pickle file.

Test with: python train_test_export_ML_model_offline.py sample_BSMs_X_file.csv sample_max_queues_Y_file.csv sample_vehicle_length_by_type_file.csv sample_stoplines_file.csv sample_signal_timing_file.csv sample_link_corner_points_file.csv
"""

# load additional libraries for this script (the rest are in queue_fx.py)
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime
from sklearn.model_selection import train_test_split
# ML Model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
# python module in the Machine_Learning folder
import queue_fx

# define path names
supporting_files_path = "../Supporting_Files/"

# Queue values from definition
QUEUE_START_SPEED = 0.00 
QUEUE_FOLLOWING_SPEED = (10.0 *0.681818) # convert ft/sec to mph
QUEUE_HEADWAY_DISTANCE = 20.0 #ft, in queue definition
QUEUE_DISTANCE_WITHIN_STOP_POINT = 20 #ft 

def format_queues(y_df_name):
    """Bin the number of vehicles in queue into pairs. 
    Assumption: This reduces the number of classes for multiclass classification by half without losing too much information."""
    # Creating a queue indicator column
    # add a column to y_df that is an indicator 1/0 for queue at intersection yes/no
    y_df_name['queue_indicator'] = 0
    mask_queue = (y_df_name['queue_count_max']>0)
    y_df_name.loc[mask_queue,'queue_indicator'] = 1
    # Creating a queue count binned column, pairs of # vehs
    # bin the queue counts into pairs as high as your max queue count observed in your training data
    binned_queues = [-np.inf,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]
    bin_labels = ["no_queue","1-2", "3-4", "5-6", "7-8", "9-10", "11-12", "13-14", "15-16",
                "17-18", "19-20", "21-22", "23-24", "25-26", "27-28"]

    # convert the categorically binned queue_count_binned column to int with .cat.codes
    y_df_name['queue_count_binned']=pd.cut(x=y_df_name['queue_count_max'], bins=binned_queues, 
                                    labels = bin_labels, include_lowest =True).cat.codes
    #print(y_df_name.head())
    return y_df_name

def join_features_and_labels(base_df_name, y_df_name, stopline_avg_df_name, signals_df_name):
    """Join the aggregated features (created from BSMs and supporting files) and their labels (queue count and length).
    Stopline_avg_df is outputted from the create_avg_stoplines_df function."""
    # join the labels (y_df) to the featuers (base_df)
    df_xy = y_df_name.merge(base_df_name, how= 'left', left_on=['time','link'],right_on=['time_30', 'assigned_linkID'])
    df_xy = df_xy.drop(['assigned_linkID', 'time_30'], axis=1)

    # Bring in a few link-specific features (e.g., # lanes, direction) by joining stopline_avg_df to base_df
    # This will add two features based on link: n_lanes and link_direction
    df_xy = df_xy.merge(stopline_avg_df_name, how='left', 
                            left_on=['link'], 
                            right_on=['Link_ID'])

    df_xy = df_xy.drop(['Link_ID','mean_X', 'mean_Y'], axis=1)

    # join signals df columns to base_df
    # issue of link ID 16:15 since accidentally coded as 15:16 in signals file! whoops.
    df_xy = df_xy.merge(signals_df_name, how='left', 
                            left_on=['time','link'], 
                            right_on=['time_30','Link_ID'])

    df_xy = df_xy.drop(['Link_ID', 'time_30'], axis=1)
    #print(df_xy.columns)
    return df_xy

def add_previous_time_queue_count_col(df_xy_name):
    """Creating a column that captures the previous 30 seconds queue_count for each link as a new feature"""
    # to datetime
    df_xy_name['time_30_dt']= pd.to_datetime(df_xy_name['time'], format="%H:%M:%S")

    # add a new column that is 30 secs prior to current time
    df_xy_name['previous_time_30sec'] = df_xy_name['time_30_dt'] - timedelta(seconds=30)

    # now remove the date from the datetime
    df_xy_name['time_30_dt'] = df_xy_name['time_30_dt'].dt.time
    df_xy_name['previous_time_30sec'] = df_xy_name['previous_time_30sec'].dt.time

    #  self inner join, left on current time, right on previous time 30sec (same link!)
    base = pd.merge(df_xy_name, df_xy_name, left_index = True, 
            left_on=['previous_time_30sec','link'],
            right_on=['time_30_dt','link'],
            how = 'inner', copy=False, suffixes=('', '_previous'))

    # columns to keep in base
    cols_keep = df_xy_name.columns.tolist()
    cols_keep.append('queue_count_max_previous')

    # keep only the original columns plus the queue_count_max_previous
    base = base.loc[:,base.columns.isin(['time','link','queue_count_max_previous'])]


    df_xy = df_xy_name.merge(base, how='left', 
                            left_on=['time','link'], 
                            right_on=['time','link'])

    df_xy.drop(['previous_time_30sec', 'time_30_dt'], axis=1, inplace=True)

    #print(df_xy.columns)
    return df_xy

def split_into_X_and_Y(df_xy_name, label_selection = 'queue_count_binned'):
    """Separate the features (X) and the labels (Y). The default label selection (Y) is queue_count_binned. 
    However, you could use queue_count_max (not binned) or queue_indicator for the classifier instead."""
     # preparing X and y
    col_lst = ['queue_count_max', 'queue_len_max', 'queue_indicator', 'queue_count_binned','time']
    X = df_xy_name.loc[:,~df_xy_name.columns.isin(col_lst)] #.to_numpy()
    #print(X.shape, "shape of features X")
    y = df_xy_name[label_selection] #.to_numpy()
    #print(y.shape, "shape of labels y")
    return X, y

def train_RF_model(X_train, X_test, y_train):
    """Train the Random Forest Classifier and make predictions on held out test data.
    Model parameters are set to those that worked well for testing and validation in this project."""
    model_rf = RandomForestClassifier(n_estimators=150, max_depth=50, random_state=0)
    model_rf.fit(X_train, y_train)
    # make predictions
    # no changes to 33% test set other than scaling
    predicted_rf = model_rf.predict(X_test)
    return predicted_rf, model_rf

def evaluate_RF_model(expected, predicted_rf):
    """Report out performance measures for the trained RF model on unseen test data.
    Measures include accuracy, weighted F1-score, confusion matrix, False Positive Rate (FPR), and False Negative Rate (FNR)."""
    # summarize the fit of the model
    print("Accuracy:", metrics.accuracy_score(expected, predicted_rf))
    # choose F1-score type
    print("Weighted F1-Score:", metrics.f1_score(expected, predicted_rf, average ='weighted'))
    #print(metrics.classification_report(expected, predicted_rf))
    print("Confusion Matrix", metrics.confusion_matrix(expected, predicted_rf))
    rf_conf_matrix = metrics.confusion_matrix(expected, predicted_rf)
    # calculating FNR and FPR
    all_0_preds = len(predicted_rf[predicted_rf ==0])
    correct_0_preds = rf_conf_matrix[0,0]
    FN = all_0_preds-correct_0_preds
    FP = sum(rf_conf_matrix[0])-rf_conf_matrix[0,0]
    TN = rf_conf_matrix[0,0]
    total_n = np.sum(rf_conf_matrix)
    TP = total_n-FN-FP-TN
    print("FN:", FN, "FP:", FP, "TP:", TP, "TN:", TN) 
    FNR = FN/(FN+TP)
    FPR = FP/(FP+TN)
    print("FPR:",FPR*100,'%')
    print("FNR:",FNR*100,'%')

def export_trained_RF_model(model_name, joblib_output_filename):
    """Export the trained RF model as a joblib pickel .pkl file"""
    print("FB joblib file:", joblib_output_filename, "MUST END IN .pkl")
    joblib.dump(model_name, joblib_output_filename)

def main():
    """Parse six command line arguments then run data cleaning, feature engineering, ML Random Forest training, and model export."""
    parser = argparse.ArgumentParser(description='Script to output trained Supervised ML Classifier (Random Forest), for offline purposes.')
    parser.add_argument('BSMs_X_filename') # CSV file of all BSMs for ML model training and testing. The BSMs will be used to create aggregated ML features (X).
    parser.add_argument('max_queues_Y_filename') # CSV file outtputed from ground_truth_max_queue_counts_and_lengths.py. These are the supervised ML labels (Y).
    parser.add_argument('veh_lengths_filename') # supporting CSV file of vehicle lengths in ft by type
    parser.add_argument('stoplines_filename') # supporting CSV file of stop line X,Y coordinates for each link and lane
    parser.add_argument('signal_timing_filename') # supporting CSV file of signal timing for each link every 30 seconds
    parser.add_argument('link_corners_filename') # supporting CSV file with link corner points for BSM assignment
    parser.add_argument('--out', help = 'Output pkl file (include .pkl)') # name of exported ML model, needs .pkl extension
    args = parser.parse_args()

    # read in the six files
    df = queue_fx.read_BSMs_file(args.BSMs_X_filename)
    y_df = format_queues(queue_fx.read_max_queues_Y_file(args.max_queues_Y_filename))
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

    # Join all features and labels
    df_xy = join_features_and_labels(base_df, y_df, stopline_avg_df, signals_df)

    # Add a column to the features for the previous time step's queue count for each link
    df_xy = add_previous_time_queue_count_col(df_xy)

    # Handle any missing values
    df_xy = queue_fx.label_encode_categorical_features(queue_fx.handle_missing_data(df_xy, df))

    X,y = split_into_X_and_Y(df_xy)

    # scale the features X for classification
    X = queue_fx.feature_scaling_X(X)

    # Split the data into training and testing sets for ML.
    # This code is for training and testing new ML models.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    # train the RF classifier and make predictions from the test set X's
    predicted_rf, model_rf = train_RF_model(X_train, X_test, y_train)

    # print model performance results
    evaluate_RF_model(y_test, predicted_rf)

    # export the trained RF model as a .pkl file
    if args.out:
        output_file = args.out
    else:
        output_file = "Exported_Trained_RF_Model.pkl"

    export_trained_RF_model(model_rf, output_file)

if __name__ == "__main__":
    main()