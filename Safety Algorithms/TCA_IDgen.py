# C2SMART Lab, NYU
# NCHRP 03-137
# @file    TCD_IDgen.py
# @author  Fan Zuo
# @author  Di Yang
# @date    2020-10-18

import pandas as pd
import numpy as np
import math
import time

def distCal(x,y,data):
    """
    Returns the distance between selected vehicle pair.

    Keyword arguments:
    >>> x: The BSM temporary ID of selected vehicle one.
    >>> y: The BSM temporary ID of selected vehicle two.
    >>> data: The original BSM trajectory data.
    RETURN: The euclidean distance between two selected vehicles(float).
    """

    # Multiple Road-side-unit(RSU) may tracking the same vehicle at the same time, so there will be some duplicated BSM info in the data.
    # Drop the duplicated data by combination of ID and time.
    x_lead = data[(data['BSM_tmp_ID'] == x) & (data['transtime'] == ID_merge_copy.at[x,'transtime_x'])].drop_duplicates(subset='transtime', keep="last").X.item()
    y_lead = data[(data['BSM_tmp_ID'] == x) & (data['transtime'] == ID_merge_copy.at[x,'transtime_x'])].drop_duplicates(subset='transtime', keep="last").Y.item()
    x_fol = data[(data['BSM_tmp_ID'] == y) & (data['transtime'] == ID_merge_copy.at[y,'transtime_y'])].drop_duplicates(subset='transtime', keep="last").X.item()
    y_fol = data[(data['BSM_tmp_ID'] == y) & (data['transtime'] == ID_merge_copy.at[y,'transtime_y'])].drop_duplicates(subset='transtime', keep="last").Y.item()
    return math.sqrt((x_lead-x_fol)**2+(y_lead-y_fol)**2)

program_st = time.time()
print("*******************  Start Program  *******************")
print("Start time %s" % (time.strftime('%X', time.localtime(program_st))))

# Loading and cleaning data. 
input_file = input("Please input the name of the trajectory file(*.csv):")
df = pd.read_csv(input_file)
df = df.drop_duplicates()
df = df.drop_duplicates(subset=['X', 'Y', 'Speed', 'Heading', 'transtime'], keep="first")

ID_max=df.groupby(['BSM_tmp_ID'],sort=False)['transtime'].max()
ID_min=df.groupby(['BSM_tmp_ID'],sort=False)['transtime'].min()
ID_merge = pd.merge(ID_max, ID_min, on = 'BSM_tmp_ID')
ID_merge['Duration'] = round(ID_merge['transtime_x'], 8) - round(ID_merge['transtime_y'], 8)
ID_merge_copy = ID_merge
ID_select = ID_merge[round(ID_merge.Duration, 8) >= round(299.9, 1)]
ID_merge = ID_merge.reset_index()
ID_select = ID_select.reset_index()

# Extract the potential ID-pairs (the same vehicle assigned multiple temporary ID) by the order of appearance time and distance.
Pairs = []
for i in range(len(ID_merge)):
    for j in range(len(ID_merge)):
        if (round(ID_merge.at[i, 'Duration'], 2) >= 299.9) and (round(ID_merge.at[i, 'transtime_x']+0.1, 2) == round(ID_merge.at[j, 'transtime_y'], 2)):
            Pairs.append([ID_merge.at[i, 'BSM_tmp_ID'], ID_merge.at[j,'BSM_tmp_ID']])
Paired = pd.DataFrame(Pairs, columns=["Leader", "Follower"])
Paired['dist'] = ''
for k in range(len(Paired)):
    Paired.at[k,'dist'] = distCal(Paired.at[k,'Leader'],Paired.at[k,'Follower'],df)
print (Paired.head())

###################################################################################################################
# Identify the ID-chains by the order of the ID-pairs, each ID-chain will match one specific vehicle

Paired = Paired.sort_values(by="dist").drop_duplicates(subset=["Leader"], keep="first")
print (Paired)
print ("PairLength:", len(Paired))
Paired = Paired[Paired.dist < 5.0] # Distance in 0.1 second greater than 5 meters will be banned 
print ("PairLength:", len(Paired))
Paired['assign'] = np.NaN
Paired.index=pd.RangeIndex(start=0, stop=len(Paired), step=1)

list_list = []
while Paired['assign'].isnull().sum() != 0:
    temp_list = []
    for i in range(len(Paired)):
        if np.isnan(Paired.at[i,'assign']):
            if len(temp_list) == 0:
                temp_list.append(Paired.at[i,'Leader'])
                temp_list.append(Paired.at[i,'Follower'])
                Paired.at[i,'assign'] = 1
            elif temp_list[-1] == Paired.at[i,'Leader']:
                temp_list.append(Paired.at[i,'Follower'])
                Paired.at[i,'assign'] = 1
            elif temp_list[0] == Paired.at[i,'Follower']:
                temp_list.append(Paired.at[i,'Leader'])
                Paired.at[i,'assign'] = 1
            else:
                pass
        else:
            pass
    list_list.append(temp_list)
print (len(list_list))

list_copy = list_list
comb_list=[]

if len(list_list)>0:
    for j in list_list:
        for k in list_list:
            if j != k:
                if j[0] == k[-1]:
                    j = k + list(set(j) - set(k))
                    list_list = list_list.pop(list_list.index(k))
                    print("0-1",len(list_list))
                elif j[-1] == k[0]:
                    j = j + list(set(k) - set(j))
                    list_list = list_list.pop(list_list.index(k))
                    print("-10",len(list_list))
        comb_list.append(j)

# Assigning new IDs.
for count in range(len(comb_list)):
    for elem in comb_list[count]:
        df['BSM_tmp_ID'] = df['BSM_tmp_ID'].replace(elem, "GID"+str(count))
df['Vehicle_ID'] = df['BSM_tmp_ID']
# Saving the updated data to a new file.
df = df.sort_values(by="Vehicle_ID")
df.to_csv("ReGen_"+input_file[:-4]+".csv")

ed_time = time.time()
print("End time %s (%f)" % (time.strftime('%X', time.localtime(ed_time)), (ed_time - program_st)))
print("*******************  End Program  *******************")