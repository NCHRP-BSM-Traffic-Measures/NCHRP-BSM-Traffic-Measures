# C2SMART Lab, NYU
# NCHRP 03-137
# @file    SSM_DRAC_CorrCoef.py
# @author  Fan Zuo
# @author  Di Yang
# @date    2020-10-18

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import math
import time
import multiprocessing as mp
import glob
from scipy.stats import spearmanr
from scipy.stats import kde
import matplotlib.pyplot as plt
import matplotlib as mpl


# 2 mpss = 6.56168 fpss
# 3 mpss = 9.84252 fpss
# 4 mpss = 13.1234 fpss


def main(dataset, dataset_ref, time_interval, threshold, ssm_type, start_point, end_point):
    """
    The main processing function of estimating the correlation coefficient.
    Returns the correlation coefficient of number of Surrogated Safety Measurements (SSM)
    between target and reference(100MPR).

    Keyword arguments:
    >>> dataset: The generated SSM from the BSM data. (Other MPR file, str)
    >>> dataset_ref: The generated SSM from the BSM data for comparison usage. (the 100MPR file, str)
    >>> time_interval: The time window that split the dataset, generated from SSM_DRAC_Opt.py. (5/10/15 min, int)
    >>> threshold: The threshold of identifying the unsafe movements, generated from SSM_DRAC_Opt.py. (float)
    >>> ssm_type: The type of the SSM output.
    >>> start_point: The start time of the dataset (second)
    >>> end_point: The end time of the dataset (second)

    RETURN: Spearmans correlation coefficient of  number of SSM events between target and reference data

    """
    if ssm_type == '1':
        # Read the whole file using useful columns.
        df = pd.read_csv(dataset, usecols=['transtime', 'Avg_Acceleration'])
        df = df.sort_values(by=['transtime'])
        # Select rows with acceleration lower than threshold
        # If the start time is not from 0, user can use the following sentence to modify the starting time point
        df = df[df.transtime > start_point]
        # Filter the the data by threshold, select rows with acceleration lower than threshold
        df = df[df.Avg_Acceleration < (-1 * threshold * 3.28)]
        # Define the time interval from minutes to seconds
        time_jump = time_interval * 60
        # Define the time ranges and bins, be careful of number of words in the file name
        ranges = np.arange(start_point, end_point+1, time_jump).tolist()
        df_sim = df.groupby(pd.cut(df.transtime, ranges)).count()
        df_sim_corr = df_sim['transtime'].tolist()

        # Read the whole reference file using useful columns.
        df_ref = pd.read_csv(dataset_ref, usecols=['transtime', 'X', 'Y', 'Avg_Acceleration'])
        df_ref = df.sort_values(by=['transtime'])
        # Select rows with acceleration lower than threshold
        # If the start time is not from 0, user can use the following sentence to modify the starting time point
        df_ref = df_ref[df_ref.transtime > start_point]
        # Filter the the data by threshold
        df_ref = df_ref[df_ref.Avg_Acceleration < (-1 * threshold * 3.28)]
        # Define the time interval from minutes to seconds
        df_uni_ref = df_ref
        time_jump = time_interval * 60
        # Define the time ranges and bins, be careful of number of words in the file name
        ranges = np.arange(start_point, end_point+1, time_jump).tolist()
        df_sim_ref = df_uni_ref.groupby(pd.cut(df_uni_ref.transtime, ranges)).count()
        df_sim_corr_ref = df_sim_ref['transtime'].tolist()
    elif ssm_type == '2':
        # Read the whole file using useful columns.
        df = pd.read_csv(dataset, usecols=['Time', 'DRAC'])
        df = df.sort_values(by=['Time'])
        # Select rows with acceleration lower than threshold
        # If the start time is not from 0, user can use the following sentence to modify the starting time point
        df = df[df.Time > start_point]
        # Filter the the data by threshold
        df = df[df.DRAC > (threshold * 3.28)]
        # Define the time interval from minutes to seconds
        time_jump = time_interval * 60
        # Define the time ranges and bins, be careful of number of words in the file name
        ranges = np.arange(start_point, end_point + 1, time_jump).tolist()
        df_sim = df.groupby(pd.cut(df.Time, ranges)).count()
        df_sim_corr = df_sim['Time'].tolist()

        # Read the whole reference file using useful columns.
        df_ref = pd.read_csv(dataset_ref, usecols=['Time', 'DRAC'])
        df_ref = df_ref.sort_values(by=['Time'])
        # Select rows with acceleration lower than threshold
        # If the start time is not from 0, user can use the following sentence to modify the starting time point
        df_ref = df_ref[df_ref.Time > start_point]
        # Filter the the data by threshold
        df_ref = df_ref[df_ref.DRAC > (threshold * 3.28)]
        # Define the time interval from minutes to seconds
        df_uni_ref = df_ref
        time_jump = time_interval * 60
        # Define the time ranges and bins, be careful of number of words in the file name
        ranges = np.arange(start_point, end_point + 1, time_jump).tolist()
        df_sim_ref = df_uni_ref.groupby(pd.cut(df_uni_ref.Time, ranges)).count()
        df_sim_corr_ref = df_sim_ref['Time'].tolist()
    elif ssm_type == '3':
        # Read the whole file using useful columns.
        df = pd.read_csv(dataset, usecols=['Time'])
        df = df.sort_values(by=['Time'])
        # Select rows with acceleration lower than threshold
        # If the start time is not from 0, user can use the following sentence to modify the starting time point
        df = df[df.Time > start_point]
        # Define the time interval from minutes to seconds
        time_jump = time_interval * 60
        # Define the time ranges and bins, be careful of number of words in the file name
        ranges = np.arange(start_point, end_point + 1, time_jump).tolist()
        df_sim = df.groupby(pd.cut(df.Time, ranges)).count()
        df_sim_corr = df_sim['Time'].tolist()

        # Read the whole reference file using useful columns.
        df_ref = pd.read_csv(dataset_ref, usecols=['Time'])
        df_ref = df_ref.sort_values(by=['Time'])
        # Select rows with acceleration lower than threshold
        # If the start time is not from 0, user can use the following sentence to modify the starting time point
        df_ref = df_ref[df_ref.Time > start_point]
        # Define the time interval from minutes to seconds
        df_uni_ref = df_ref
        time_jump = time_interval * 60
        # Define the time ranges and bins, be careful of number of words in the file name
        ranges = np.arange(start_point, end_point + 1, time_jump).tolist()
        df_sim_ref = df_uni_ref.groupby(pd.cut(df_uni_ref.Time, ranges)).count()
        df_sim_corr_ref = df_sim_ref['Time'].tolist()

    print (df_sim_corr, len(df_sim_corr))
    print (df_sim_corr_ref, len(df_sim_corr_ref))

    # calculate spearman's correlation
    coef, p = spearmanr(df_sim_corr, df_sim_corr_ref)
    print('For time interval', time_interval, 'min, DRAC threshold', drac_threshold, ', spearmans correlation coefficient: %.3f' % coef)
    # interpret the significance
    alpha = 0.05
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)
    return(coef)

if __name__ == "__main__":
    program_st = time.time()
    print("*******************  Start Program  *******************")
    print("Start time %s" % (time.strftime('%X', time.localtime(program_st))))

    s_t = input("Please select the type of the SSM (1 - Hard Braking, 2 - DRAC, 3 - TTCD):")
    time_window = int(input("Please input the optimal time interval:"))
    opt_threshold = float(input("Please input the optimal threshold:"))
    ssm_file_100 = input("Please input the name of the 100MPR SSM file(*.csv):")
    ssm_file_75 = input("Please input the name of the 75MPR SSM file(*.csv):")
    ssm_file_50 = input("Please input the name of the 50MPR SSM file(*.csv):")
    ssm_file_20 = input("Please input the name of the 20MPR SSM file(*.csv):")
    ssm_file_5 = input("Please input the name of the 5MPR SSM file(*.csv):")
    Start_Point = float("{:.1f}".format(input("Please input the start time of the sub-interval(int): ")))
    End_Point = float("{:.1f}".format(input("Please input the end time of the sub-interval(int): ")))

    Diff_100_75 = main(ssm_file_75, ssm_file_100, time_window, opt_threshold, s_t, Start_Point, End_Point)
    Diff_100_50 = main(ssm_file_50, ssm_file_100, time_window, opt_threshold, s_t, Start_Point, End_Point)
    Diff_100_20 = main(ssm_file_20, ssm_file_100, time_window, opt_threshold, s_t, Start_Point, End_Point)
    Diff_100_5 = main(ssm_file_5, ssm_file_100, time_window, opt_threshold, s_t, Start_Point, End_Point)

    Diff_result = [Diff_100_75, Diff_100_50, Diff_100_20, Diff_100_5]
    print(Diff_result)
    
    x_label = ["100% & 75%", "100% & 50%", "100% & 20%", "100% & 5%"]
    x_pos = [i for i, _ in enumerate(x_label)]

    mpl.rcParams['font.size'] = 18.0
    mpl.rcParams['axes.titlesize'] = 18.0
    csfont = {'fontname': 'Times New Roman'}
    plt.plot(x_pos, Diff_result, 'o-')
    for x,y in zip(x_pos,Diff_result):
        label = "{:.2f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center', va='bottom', size = 16,**csfont)
    plt.xlabel("Market Penetration Rate Pair",**csfont)
    plt.ylabel("Correlation Coefficient",**csfont)
    plt.title("Correlation coefficient between\n 100% MPR and each MPR level",**csfont)
    plt.grid()
    plt.xticks(x_pos, x_label,**csfont)
    plt.yticks(**csfont)
    plt.ylim(0, 1)
    figure = plt.gcf()
    figure.set_size_inches(7, 6)
    # Adjust the output name if there are multiple results will be generated.
    plt.savefig('CorrCoef_Typ%s.png'%(s_t),bbox_inches='tight',dpi=100)
    plt.show()

    ed_time = time.time()
    print("End time %s (%f)" % (time.strftime('%X', time.localtime(ed_time)), (ed_time - program_st)))
    print("*******************  End Program  *******************")