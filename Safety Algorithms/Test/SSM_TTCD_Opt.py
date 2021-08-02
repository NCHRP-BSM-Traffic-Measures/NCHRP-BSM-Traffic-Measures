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

def main(dataset, time_interval, TTCD_threshold, start_point, end_point):
    """
    The main processing function of estimating the correlation coefficient.
    Returns the correlation coefficient of number of Surrogated Safety Measurements (SSM) and the number of real crashes.

    Keyword arguments:
    >>> dataset: The generated SSM from the BSM data.
    >>> time_interval: The time window that split the dataset (5min, 10min, 15min).
    >>> DRAC_threshold: The threshold of identifying the unsafe movements.
    >>> start_point: The start time of the dataset (second)
    >>> end_point: The end time of the dataset (second)

    RETURN: Spearmans correlation coefficient of the SSM and the crash data

    """
    # Read the whole file using useful columns.
    df = pd.read_csv(dataset, usecols=['Time', 'CRD_RE', 'CRD_CR', 'CRD_LC'])
    df = df.sort_values(by=['Time'])
    # Select rows with acceleration lower than threshold
    df = df[df.Time > start_point]
    print (len(df))
    df['CRD'] = df.apply(lambda row: row['CRD_RE'] + row['CRD_CR'] + row['CRD_LC'], axis=1)
    df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

    # Define the time interval from minutes to seconds
    time_jump = time_interval * 60

    # Define the time ranges and bins, be careful of number of words in the file name
    ranges = np.arange(start_point, end_point, time_jump).tolist()
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


    traj_file_15 = input("Please input the name of the 100MPR TTCD-1.5 file(*.csv):")
    traj_file_20 = input("Please input the name of the 100MPR TTCD-2.0 file(*.csv):")
    traj_file_25 = input("Please input the name of the 100MPR TTCD-2.5 file(*.csv):")
    Start_Point = float("{:.1f}".format(input("Please input the start time of the sub-interval(int): ")))
    End_Point = float("{:.1f}".format(input("Please input the end time of the sub-interval(int): ")))

    final_result = pd.DataFrame(columns=['TTCD:1.5', 'TTCD:2', 'TTCD:2.5', 'Optimal Threshold'])

    for i in range(5, 20, 5):
        temp_list = []
        for j in range(15, 30, 5):
            if j == 15:
                temp_list.append(main(traj_file_15, i, j * 0.1, Start_Point, End_Point))
            elif j == 20:
                temp_list.append(main(traj_file_20, i, j * 0.1, Start_Point, End_Point))
            elif j == 25:
                temp_list.append(main(traj_file_25, i, j * 0.1, Start_Point, End_Point))
        print (temp_list)
        opt_threshold = str(temp_list.index(max(temp_list)))
        if opt_threshold == "0":
            temp_list.append("TTCD:1.5")
        elif opt_threshold == "1":
            temp_list.append("TTCD:2")
        else:
            temp_list.append("TTCD:2.5")
        final_result.loc[len(final_result), :] = temp_list

    new_index = ['5min', '10min', '15min']
    final_result.index = new_index
    print(final_result)
    final_result.to_csv("Optimal_Threshold_TTCD.csv")

    ed_time = time.time()
    print("End time %s (%f)" % (time.strftime('%X', time.localtime(ed_time)), (ed_time - program_st)))
    print("*******************  End Program  *******************")