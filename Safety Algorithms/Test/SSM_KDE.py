# C2SMART Lab, NYU
# NCHRP 03-137
# @file    SSM_DRAC_KDE.py
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

# 2 mpss = 6.56168 fpss
# 3 mpss = 9.84252 fpss
# 4 mpss = 13.1234 fpss

def minmax(files, col_x, col_y):
    """
    The function of identifying the maximum geo-boundaries of the data.

    Keyword arguments:
    >>> files: The list of SSM data files. (List of strings)
    >>> col_x: The column name of the x-coordinates. (str)
    >>> col_y: The column name of the y-coordinates. (str)

    RETURN: List of the coordinates boundaries of involved data: [Minimum x, Maximum x, Minimum y, Maximum y].

    """
    x_min_list = []
    x_max_list = []
    y_min_list = []
    y_max_list = []
    for file in files:
        geo = pd.read_csv(dataset, usecols=[col_x, col_y])
        geo = geo.dropna()
        x_min_list.append(df_kde[col_x].min())
        x_max_list.append(df_kde[col_x].max())
        y_min_list.append(df_kde[col_y].min())
        y_max_list.append(df_kde[col_y].max())
    return ([min(x_min_list), max(x_max_list), min(y_min_list),max(y_max_list)])



def main(dataset, bins, threshold, ssm_type, bound):
    """
    The main processing function of estimating the correlation coefficient.
    Returns the correlation coefficient of number of Surrogated Safety Measurements (SSM) and the number of real crashes.

    Keyword arguments:
    >>> dataset: The generated SSM from the BSM data.
    >>> bins: The spatial data will be seperated into bins*bins matrix.
    >>> DRAC_threshold: The threshold of identifying the unsafe movements.
    >>> ssm_type: The type of the SSM output.
    >>> bound: The list of coordinates boundaries [Minimum x, Maximum x, Minimum y, Maximum y]

    RETURN: Spearmans correlation coefficient of the SSM and the crash data

    """
    # Read the whole file using useful columns.
    if ssm_type == '1':
        df = pd.read_csv(dataset, usecols=['transtime', 'X', 'Y', 'Avg_Acceleration'])
        df = df.sort_values(by=['transtime'])
        # Select rows with acceleration lower than threshold
        df = df[df.Avg_Acceleration < (-1 * threshold * 3.28)]
        ### Kernel density ranking ###
        df_kde = df.drop(columns = ['transtime', 'Avg_Acceleration'])
        df_kde = df_kde.dropna()
    elif ssm_type == '2':
        df = pd.read_csv(dataset, usecols=['Time', 'x_B', 'y_B', 'DRAC'])
        df = df.sort_values(by=['Time'])
        # Select rows with acceleration lower than threshold
        df = df[df.DRAC > (threshold * 3.28)]
        ### Kernel density ranking ###
        df_kde = df.drop(columns=['Time', 'DRAC'])
        df_kde = df_kde.dropna()
    elif ssm_type == '3':
        df = pd.read_csv(dataset, usecols=['Time', 'x_B', 'y_B'])
        df = df.sort_values(by=['Time'])
        ### Kernel density ranking ###
        df_kde = df.drop(columns=['Time'])
        df_kde = df_kde.dropna()

    nbins = bins
    k = kde.gaussian_kde(df_kde.values.T, bw_method='scott')

    min_x = bound[0]
    max_x = bound[1]
    min_y = bound[2]
    max_y = bound[3]

    grid_x, grid_y = np.mgrid[min_x:max_x:nbins*1j, min_y:max_y:nbins*1j]
    zi = k(np.vstack([grid_x.flatten(), grid_y.flatten()]))
    df_order_100MPR = pd.DataFrame (zi,columns=['Origin'])
    df_order_100MPR['Order_Origin'] = df_order_100MPR.index
    df_order_100MPR = df_order_100MPR.sort_values(by=['Origin'])
    df_order_100MPR.insert(loc=0,column='Order_Rank',value=df_order_100MPR.groupby('Origin').ngroup()+1)
    df_order_100MPR = df_order_100MPR.sort_values(by=['Order_Origin'])
    df_order_100MPR.index = pd.RangeIndex(start=0, stop=len(df_order_100MPR), step=1)
    return df_order_100MPR

if __name__ == "__main__":
    program_st = time.time()
    print("*******************  Start Program  *******************")
    print("Start time %s" % (time.strftime('%X', time.localtime(program_st))))

    
    bin_size = int(input("Please input the number of bins:"))
    s_t = input("Please select the type of the SSM (1 - Hard Braking, 2 - DRAC, 3 - TTCD):")
    optimal_threshold = float(input("Please input the optimal threshold:"))
    ssm_file_100 = input("Please input the name of the 100MPR SSM file(*.csv):")
    ssm_file_75 = input("Please input the name of the 75MPR SSM file(*.csv):")
    ssm_file_50 = input("Please input the name of the 50MPR SSM file(*.csv):")
    ssm_file_20 = input("Please input the name of the 20MPR SSM file(*.csv):")
    ssm_file_5 = input("Please input the name of the 5MPR SSM file(*.csv):")

    SSM_files = [ssm_file_100,ssm_file_75,ssm_file_50,ssm_file_20,ssm_file_5]
    bounds = minmax(SSM_files, x_column, y_column)

    kdf_100MPR = main(ssm_file_100, bin_size, optimal_threshold, s_t, bounds)
    kdf_75MPR = main(ssm_file_75, bin_size, optimal_threshold, s_t, bounds)
    kdf_50MPR = main(ssm_file_50, bin_size, optimal_threshold, s_t, bounds)
    kdf_20MPR = main(ssm_file_20, bin_size, optimal_threshold, s_t, bounds)
    kdf_5MPR = main(ssm_file_5, bin_size, optimal_threshold, s_t, bounds)

    Diff_100_75 = (kdf_100MPR['Order_Rank'] - kdf_75MPR['Order_Rank']).apply(lambda x: abs(x)).mean()
    Diff_100_50 = (kdf_100MPR['Order_Rank'] - kdf_50MPR['Order_Rank']).apply(lambda x: abs(x)).mean()
    Diff_100_20 = (kdf_100MPR['Order_Rank'] - kdf_20MPR['Order_Rank']).apply(lambda x: abs(x)).mean()
    Diff_100_5 = (kdf_100MPR['Order_Rank'] - kdf_5MPR['Order_Rank']).apply(lambda x: abs(x)).mean()

    Diff_result = [Diff_100_75, Diff_100_50, Diff_100_20, Diff_100_5]

    print(Diff_result)

    #plt.style.use('ggplot')
    mpl.rcParams['font.size'] = 18.0
    mpl.rcParams['axes.titlesize'] = 18.0
    csfont = {'fontname': 'Times New Roman'}
    x_label = ["100% & 75%", "100% & 50%", "100% & 20%", "100% & 5%"]
    x_pos = [i for i, _ in enumerate(x_label)]

    plt.plot(x_pos, Diff_result, 'o-')
    for x,y in zip(x_pos,Diff_result):
        label = "{:.2f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center', va='bottom', size = 12, **csfont)
    plt.xlabel("Market Penetration Rate Pair", **csfont)
    plt.ylabel("Mean Ranking Difference", **csfont)
    plt.title("Corresponding Ranking Differences between MPR", **csfont)
    plt.grid()
    plt.ylim(20,140)
    plt.xticks(x_pos, x_label, **csfont)
    plt.yticks(**csfont)
    figure = plt.gcf()
    figure.set_size_inches(7, 6)
    plt.savefig('RankDiff_Typ%s_%sx%s_Th%s.png' %
                (s_t, str(bin_size), str(bin_size), str(int(optimal_threshold*10))),bbox_inches='tight',dpi=100)
    plt.show()

    ed_time = time.time()
    print("End time %s (%f)" % (time.strftime('%X', time.localtime(ed_time)), (ed_time - program_st)))
    print("*******************  End Program  *******************")