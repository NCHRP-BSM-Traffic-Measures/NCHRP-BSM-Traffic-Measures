import pandas as pd
import numpy as np
import argparse
from imblearn.over_sampling import RandomOverSampler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
import kerastuner as kt

"""Use Keras Tuner Hyberband, which trains multiple models on the training data, to determine the best hyperparameters to use in detecting 
incidents in the validation data and then save the best performing model."""

def model_test_builder(hp):
    """Use Keras Tuner to create random neural network models of various sizes to evaluate and determine the model with the best performing hyperparameters."""
    model = keras.Sequential()
    model.add(normalizer)    

    for i in range(hp.Int('numLayers',min_value=1,max_value=6,step=1)):
        model.add(keras.layers.Dense(hp.Int('hidden_size_{}'.format(i), min_value = 16, max_value = 320, step = 32), activation='relu'))
        model.add(keras.layers.Dropout(hp.Float('Dropout_{}'.format(i), min_value = 0.0, max_value = 0.5, step = 0.05)))
    
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[.001,.0001,.00001])),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    return model

def final_model_builder(best_hps):
    """Use best_hps to build final model using the best performing hyperparameters as determined by Keras Tuner."""
    model = keras.Sequential()
    model.add(normalizer)

    for i in range(best_hps.get('numLayers')):
        model.add(keras.layers.Dense(best_hps.get('hidden_size_{}'.format(i)), activation='relu'))
        model.add(keras.layers.Dropout(best_hps.get('Dropout_{}'.format(i))))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=keras.optimizers.Adam(best_hps.get('learning_rate')),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    return model

def run_hyperband(project_name,X_ros, y_ros, test, actual):
    """Set up and run the Keras Tuner Hyperband which creates a bracket of many different neural network models 
    with different hyperparameters and trains, tests and evaluates them to determine the best one and stores the values in
    best_hps."""
    METRICS = [
          keras.metrics.TruePositives(name='tp'),
          keras.metrics.FalsePositives(name='fp'),
          keras.metrics.TrueNegatives(name='tn'),
          keras.metrics.FalseNegatives(name='fn'), 
          keras.metrics.BinaryAccuracy(name='accuracy'),
          keras.metrics.Precision(name='precision'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='auc')
    ]

    tuner = kt.Hyperband(model_test_builder, objective = kt.Objective('val_auc',direction='max'), max_epochs = 100, factor = 3, project_name=project_name)   

    tuner.search(X_ros,y_ros,batch_size=240,epochs=20,validation_data=(np.array(test),np.array(actual)))

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    return best_hps

def build_and_save(best_hps, X_ros, y_ros, test, actual, save_filename):
    """Using the best hyperparameters as determined by Keras Tuner, build and train a model and save it to save_filename for later use."""
    model = final_model_builder(best_hps)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', 
        verbose=1,
        patience=5,
        mode='max',
        restore_best_weights=True)
    model.fit(X_ros,y_ros,batch_size=240,epochs=100,validation_data=(np.array(test),np.array(actual)), callbacks = [early_stopping])
    model.save(save_filename)

def main():
    """Read in training and test data, add ground truth labels, run hyperband and save the best performing model."""
    parser = argparse.ArgumentParser(description='Measures Estimation program for training and saving neural network for incident detection.')
    parser.add_argument('training_data')#csv
    parser.add_argument('test_data')#csv
    parser.add_argument('project_name') #Where to save KT Hyperband information to
    parser.add_argument('save_filename') #Where to save trained best performing model to
    args = parser.parse_args()

    df_train = pd.read_csv(training_data).fillna(0)
    #Ground truth data is binary True or False whether an incident was present on the link at the current time
    labels = (df_features['Link'] >= 637) & (df_features['Link'] <= 638) & (df_features['CurrentTime'] >= 12000) & (df_features['CurrentTime'] <= 14100)

    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(df_train))

    ros = RandomOverSampler()
    X_ros, y_ros = ros.fit_sample(df_train, labels)

    df_test = pd.read_csv(testing_data).fillna(0)
    #Ground truth data is binary True or False whether an incident was present on the link at the current time
    actual = (df_test['Link'] >= 249) & (df_test['Link'] <= 250) & (df_test['CurrentTime'] >= 14640) & (df_test['CurrentTime'] <= 14940)

    best_hps = run_hyperband(args.project_name, X_ros, y_ros, df_test, actual)
    build_and_save(best_hps, X_ros, y_ros, df_test, actual, args.save_filename)

if __name__ == "__main__":
    main()