# C2SMART Lab, NYU
# NCHRP 03-137
# @file    SSM_DRAC_Opt.py
# @author  Fan Zuo
# @author  Di Yang
# @date    2020-10-18

import pandas as pd
import numpy as np
import math
import time
import glob
from scipy.stats import spearmanr
from scipy.stats import kde
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import matplotlib as mpl

def main(dataset, time_interval, threshold, start_point, end_point, ssm_type):
    """
    The main processing function of estimating the correlation coefficient.
    Returns the correlation coefficient of number of Surrogated Safety Measurements (SSM) and the number of real crashes.

    Keyword arguments:
    >>> dataset: The generated SSM from the BSM data.
    >>> time_interval: The time window that split the dataset (5min, 10min, 15min).
    >>> threshold: The threshold of identifying the unsafe movements.
    >>> start_point: The start time of the dataset (second)
    >>> end_point: The end time of the dataset (second)
    >>> ssm_type: The type of the SSM output.

    RETURN: Spearmans correlation coefficient of the SSM and the crash data

    """
    # Read the whole file using useful columns.
    if ssm_type == '1':
        df = pd.read_csv(dataset, usecols=['transtime', 'Avg_Acceleration'])
        df = df.sort_values(by=['transtime'])
        # Select rows with acceleration lower than threshold
        # If the start time is not from 0, user can use the following sentence to modify the starting time point
        df = df[df.transtime > start_point]
        # Filter the the data by threshold
        df = df[df.Avg_Acceleration < (-1 * threshold * 3.28)]
        df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

        # Define the time interval from minutes to seconds
        time_jump = time_interval * 60

        # Define the time ranges and bins, be careful of number of words in the file name
        ranges = np.arange(start_point, end_point + 1, time_jump).tolist()
        df_sim = df.groupby(pd.cut(df.transtime, ranges)).count()
        df_sim_corr = df_sim['transtime'].tolist()

    elif ssm_type == '2':
        df = pd.read_csv(dataset, usecols=['Time', 'DRAC'])
        df = df.sort_values(by=['Time'])
        # Select rows with acceleration lower than threshold
        # If the start time is not from 0, user can use the following sentence to modify the starting time point
        df = df[df.Time > start_point]
        # Filter the the data by threshold
        df = df[df.DRAC > (threshold * 3.28)]
        df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

        # Define the time interval from minutes to seconds
        time_jump = time_interval * 60

        # Define the time ranges and bins, be careful of number of words in the file name
        ranges = np.arange(start_point, end_point+1, time_jump).tolist()
        df_sim = df.groupby(pd.cut(df.Time, ranges)).count()
        df_sim_corr = df_sim['Time'].tolist()

    # Load crash data, separate them into different time interval by setting in the function variable
    if time_interval == 5:
        df_crash = pd.read_csv("CrashData_5min.csv")
        df_crash_corr = df_crash['num'].tolist()
    if time_interval == 10:
        df_crash = pd.read_csv("CrashData_10min.csv")
        df_crash_corr = df_crash['num'].tolist()
    if time_interval == 15:
        df_crash = pd.read_csv("CrashData_15min.csv")
        df_crash_corr = df_crash['num'].tolist()
    print (df_sim_corr, len(df_sim_corr))
    print (df_crash_corr, len(df_crash_corr))

    # calculate spearman's correlation
    coef, p = spearmanr(df_sim_corr, df_crash_corr)
    print('For time interval', time_interval, 'min, acceleration threshold', DRAC_threshold,
          ', spearmans correlation coefficient: %.3f' % coef)
    # interpret the significance
    alpha = 0.05
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)
    return (coef)

if __name__ == "__main__":
    program_st = time.time()
    print("*******************  Start Program  *******************")
    print("Start time %s" % (time.strftime('%X', time.localtime(program_st))))


    traj_file = input("Please input the name of the 100MPR SSM file(*.csv):")
    s_t = input("Please select the type of the SSM (1 - Hard Braking, 2 - DRAC):")
    Start_Point = float("{:.1f}".format(input("Please input the start time of the sub-interval(int): ")))
    End_Point = float("{:.1f}".format(input("Please input the end time of the sub-interval(int): ")))

    if s_t == '1':
        final_result = pd.DataFrame(columns=['2mpss', '3mpss', '4mpss', 'Optimal Threshold'])
        # Since we have 3 different time interval settings (5min, 10min, 15min), the loop is defined to iterate those value
        for i in range(5, 20, 5):
            temp_list = []
            # Loop alternative HB thresholds (2, 3, 4)
            for j in range(2, 5):
                # Load the main function to calculate the correlation coefficients for different interval+threshold combinations
                temp_list.append(main(traj_file, i, j, Start_Point, End_Point, s_t))
            print (temp_list)
            opt_threshold = str(temp_list.index(max(temp_list)))
            if opt_threshold == "0":
                temp_list.append("2mpss")
            elif opt_threshold == "1":
                temp_list.append("3mpss")
            else:
                temp_list.append("4mpss")
            final_result.loc[len(final_result), :] = temp_list

        new_index = ['5min', '10min', '15min']
        final_result.index = new_index
        print(final_result)
        final_result.to_csv("Optimal_Threshold_HB.csv")

    elif s_t == '2':
        final_result = pd.DataFrame(columns=['2.5mpss', '3mpss', '3.5mpss', 'Optimal Threshold'])
        # Since we have 3 different time interval settings (5min, 10min, 15min), the loop is defined to iterate those value
        for i in range(5, 20, 5):
            temp_list = []
            # Loop alternative DRAC thresholds (2.5, 3.0, 3.5)
            for j in range(25, 40, 5):
                # Load the main function to calculate the correlation coefficients for different interval+threshold combinations
                temp_list.append(main(traj_file, i, j * 0.1, Start_Point, End_Point, s_t))
            print (temp_list)
            # Select the maximum correlation coefficient value related threshold as the optimized result
            opt_threshold = str(temp_list.index(max(temp_list)))
            if opt_threshold == "0":
                temp_list.append("2.5mpss")
            elif opt_threshold == "1":
                temp_list.append("3mpss")
            else:
                temp_list.append("3.5mpss")
            final_result.loc[len(final_result), :] = temp_list

        new_index = ['5min', '10min', '15min']
        final_result.index = new_index
        print(final_result)
        final_result.to_csv("Optimal_Threshold_DRAC.csv")

    ed_time = time.time()
    print("End time %s (%f)" % (time.strftime('%X', time.localtime(ed_time)), (ed_time - program_st)))
    print("*******************  End Program  *******************")