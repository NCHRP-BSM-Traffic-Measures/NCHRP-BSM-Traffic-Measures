# Description
The Mean Time to Detect and Verify Incidents application detects and verifies roadway incidents and measures the time to detection for a given arterial or freeway network using Basic Safety Messages. This application can be run as both an offline and online mode.

# Usage
The Mean Time to Detect and Verify Incidents pipeline begins by first creating features data for training and evaluating hyperparameter options and building the best performing model from the training and validation data. The model can then be used with additional testing or deployment data either in an offline or online mode.

## Offline Feature Extraction
offline_incident_feature_extraction.py is the program used for feature extraction of the training data for building the model for either offline or online deployment and for feature extraction of test or deployment data for the offline mode. It requires two input files: a bsm file which is a BSM output file from the Trajectory Conversion Algorithm (TCA) and a links file which is the output of vissimlinkmaker.py. You can also specify an output filename, or leave it as the default. Offline Feature Extraction does not perform any training/validation/test data splitting and must be run multiple times on seperate input files to create training, validation and test data sets.

To run the offline feature extraction use the following command:

```
python offline_incident_feature_extraction.py [bsm_filename | REQUIRED] [link_filename | REQUIRED] --out [output_filename | OPTIONAL]
```

## Evaluate and Train Model
evaluate_and_train_model.py uses Keras Tuner Hyberband, which trains multiple models on the training data, to determine the best hyperparameters to use in detecting incidents in the validation data and then saves the best performing model. It requires four command line arguments: the training feature data from offline_incident_feature_extraction.py, the test feature data from offline_incident_feature_extraction.py, a project name to save the Keras Tuner Hyperband results to, and a model name to save the best performing model to. Before running the code must be edited so that the variables label and actual are set to label rows in the data where an incident was present as True and all other rows as False.

To run the Keras Tuner model evaluation and training use the following command:
```
python evaluate_and_train_model.py [training_data | REQUIRED] [test_data | REQUIRED] [kerastuner_project_name | REQUIRED] [model_filename | REQUIRED]
```

## Run Model Offline
runmodel_offline.py runs the incident detection neural network model created in evaluate and train model in an offline using a test features file created by offline feature extraction. The primary use of this would be for model evaluation during the early stages of model development and deployment to a new area to ensure it performs as expected before setting up the live version. 

It requires two command line arguments: a csv file of test features from offline incident feature extraction and the name of the model saved by evaluate and train model. Before running the code must be edited so that the variable actual is set to label rows in the data where an incident was present as True and all other rows as False.

To run the model offline use the following command:
```
python runmodel_offline.py [test_data | REQUIRED] [model_filename | REQUIRED] --out [output_filename | OPTIONAL]
```

## Feature Extraction and Run Model Live
featureextraction_and_runmodel_live.py is a single pipeline that performs feature extraction from TCA Basic Safety Messages, uses the saved model from Evaluate and Train Model to detect an incident on a link, and tracks detected incidents to verify incident conditions if the same link is positive six or more consecutive times (30 seconds or more). This is designed to work with live streaming BSMs, or emulated streaming BSMs through bsm_stream.py. 

It requires four command line arguments: a csv file of TCA Basic Safety Messages for streaming, a links file for bsm_stream.py to use for binning the BSMs by geographic roadway link, the name of the model saved by evaluate and train mode, and a filename to save the timing output for performance evaluation. The user can also optionally specify the filname to save verified incidents to or a default file will be used.

To run the full feature extraction and incident detection and verification pipeline use the following command:
```
python featureextraction_and_runmodel_live.py [bsm_filename | REQUIRED] [links_filename | REQUIRED] [model_filename | REQUIRED] [timing_filename | REQUIRED] --out [output_filename | OPTIONAL]
```


For license and contributor information see the main Mobility Algorithms README.