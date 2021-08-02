# Description
The Queue Length application detects queues at known bottleneck locations and estimates their vehicle counts and lengths for a given arterial network using Basic Safety Messages (BSMs).

*Please note: these scripts were developed using simulated data. Before attempting to run these scripts for your own network, you will need to create files similar to those included in the Supporting_Files folder. You will also need vehicle trajectories to calculate the ground truth labels OR obtain maximum queue counts and lengths per link and 30 seconds from other data.* 

# Usage
The Queue Count/Length pipeline includes code for ground truth calculations (supervised machine learning (ML) labels), offline data and model preparation, and online execution. The ground truth code is in the "Calculating_Ground_Truths" folder. The offline and live ML scripts are in the "Machine_Learning" folder. 

This pipeline also contains a "Supporting_Files" folder with sample input and ouptut files required to run the python scripts.

In the "Machine_Learning" folder, queue_fx.py contains libraries and functions needed to run both train_test_export_ML_model_offline.py and run_exported_ML_model_live.py. 

## Calculating Ground Truth Queues from Vehicle Trajectories
ground_truth_max_queue_counts_and_lengths.py script in the "Calculating_Ground_Truths" folder calculates ground truth queue counts (# of vehicles in the queue) and lengths (in feet from the stop line to the back of the last vehicle in queue) per roadway link (across all lanes) for every 30 seconds and outputs to a csv file. This csv output serves as the supervised machine learning labels (one of the inputs) for the offline model training script in the "Machine_Learning" folder, train_test_export_ML_model_offline.py. 

It requires three command line arguments: (1) the name of the vehicle trajectories file, (2) the name of the vehicle lengths by type file, and (3) the name of the stoplines file with the X,Y coordinates. It looks for these files in the "Supporting_Files" folder. 

To run the ground_truth_max_queue_counts_and_lengths.py script in the "Calculating_Ground_Truths" folder, use the following command:
```
python ground_truth_max_queue_counts_and_lengths.py [trajectories_filename | REQUIRED] [veh_lengths_filename | REQUIRED] [stoplines_filename | REQUIRED] --out [output_filename | OPTIONAL]
```

## Training and Exporting the Machine Learning Classifier (Offline)
train_test_export_ML_model_offline.py script in the "Machine_Learning" folder brings together BSM, network, and ground truth data to train, test, and export a Random Forest Classifier that determines the queue count at each defined bottleneck location every 30 seconds. The script outputs classifier performance results in the command line and exports the trained RF parameters to a pickle file (.pkl). 

It requires six command line arguments: (1) the name of the BSMs file, (2) the name of the ground truth queues file outputted from ground_truth_max_queue_counts_and_lengths.py, (3) the name of the vehicle lengths by type file, (4) the name of the stoplines file with the X,Y coordinates, (5) the name of the signal timing file with the % of time the light was green, amber, and red for each 30-second interval and link, and (6) the name of the file with the link corner points (X,Y coordinates) to use for BSM link assignment.

*Please note: the BSMs used for this project were generated from the Trajectory Conversion Algorithm (TCA) Software. The BSMs do not have any gaps or errors.*

To run the train_test_export_ML_model_offline.py script in the "Machine_Learning" folder, use the following command:
```
python train_test_export_ML_model_offline.py [BSMs_X_filename | REQUIRED] [max_queues_y_filename | REQUIRED] [veh_lengths_filename | REQUIRED] [stoplines_filename | REQUIRED] [signal_timing_filename | REQUIRED] [link_corners_filename | REQUIRED] --out [output_filename | OPTIONAL]
```

## Running the Exported ML Model in Real-Time to Classify Queue Counts and Derive Queue Lengths (Live)
python run_exported_ML_model_live.py processes a csv file with 30 seconds of BSMs, imports the trained ML Random Forest classifer outputted from train_test_export_ML_model_offline.py, classifies queue counts for that time period for each link, estimates queue lengths, and outputs results to standard output and to a csv file. 

To run the run_exported_ML_model_live.py script in the "Machine_Learning" folder, use the following command:
```
python run_exported_ML_model_live.py [BSMs_30secs_filename | REQUIRED] [model_filename | REQUIRED] [veh_lengths_filename | REQUIRED] [stoplines_filename | REQUIRED] [signal_timing_filename | REQUIRED] [link_corners_filename | REQUIRED]
```

## Contributors
For license and contributor information see the main Mobility Algorithms README.
