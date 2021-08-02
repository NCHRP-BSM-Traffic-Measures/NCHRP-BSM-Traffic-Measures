# C2SMART Lab, NYU
# NCHRP 03-137
# @file    TTCD_Calculation_Offline.py
# @author  Fan Zuo
# @author  Di Yang
# @date    2020-10-18

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import math
import time
import multiprocessing as mp
from itertools import repeat
from scipy import spatial
import sys

def frange(start, stop=None, step=None):
    """Returns the range by float numbers."""

    if stop == None:
        stop = start + 0.0
        start = 0.0

    if step == None:
        step = 1.0

    while True:
        if step > 0 and start >= stop:
            break
        elif step < 0 and start <= stop:
            break
        yield ("%g" % start) # return float number
        start = start + step


def dist(x1, y1, x2, y2):
    """
    Returns the euclidean distance.

    Keyword arguments:
    >>> x1: float value for X for first point (ft.)
    >>> y1: float value for Y for first point
    >>> x2: float value for X for 2nd point
    >>> y2: float value for Y for 2nd point
    RETURN: The euclidean distance(float, meter).
    """

    return float("{:.6f}".format(math.sqrt((x2-x1) ** 2 + (y2 - y1) ** 2)))

def get_heading(x1, y1, x2, y2):
    """
    Returns the Heading based on two points

    Keyword arguments:
    >>> x1: Float value for X for first point
    >>> y1: Float value for Y for first point
    >>> x2: Float value for X for 2nd point
    >>> y2: Float value for Y for 2nd point
    RETURN: The new heading value(float).
    """

    heading = 0
    dx = x2 - x1
    dy = y2 - y1

    if dx != 0:
        heading = float("{:.6f}".format((90 - math.degrees(math.atan2(dy, dx)) + 360) % 360))

    elif dy > 0:
        heading = 0

    elif dy < 0:
        heading = 180

    return heading

def ttc_location(data_check, distance, start_time):
    """
    Returns TTCmax Location (Please see the document for the detailed definition).

    Keyword arguments:
    >>> data_check: The working data frame selected from the main data frame.
    >>> distance: The projecting distance based on the current speed (ft.).
    >>> start_time: The time stamp of the processing step.
    RETURN: TTCmax point X, TTCmax point Y, the nearest time stamp before the TTCmax location projected, 
            heading of the vehicle at the TTCmax point.
    """

    # Start with jump 0.1 sec
    dist1 = distance
    Start_X = data_check.at[0, 'X']
    Start_Y = data_check.at[0, 'Y']
    TTC_X = np.NaN
    TTC_Y = np.NaN
    Heading = np.NaN

    for i in range(len(data_check)-1):
        Check_X = data_check.at[i + 1, 'X']
        Check_Y = data_check.at[i + 1, 'Y']
        dist2 = dist(Start_X, Start_Y, Check_X, Check_Y)
        if dist2 <= dist1:
            dist1 = dist1 - dist2
            Start_X = Check_X
            Start_Y = Check_Y
            start_time = float("{:.1f}".format(start_time + 0.1))
            pass
        else:
            Heading = get_heading(Start_X, Start_Y, Check_X, Check_Y)
            rad = math.pi / 2 - math.radians(Heading)
            TTC_X = Start_X + dist1 * math.cos(rad)
            TTC_Y = Start_Y + dist1 * math.sin(rad)
            start_time = float("{:.1f}".format(start_time + 0.1))
            break
    return [TTC_X, TTC_Y, float("{:.1f}".format(start_time - 0.1)), Heading]

def next_location(data_check, point, X_now, Y_now, speed_now, accel, heading):
    """Returns next time step's location of the target vehicle.

    Keyword arguments:
    >>> data_check: The working data frame selected from the main dataframe.
    >>> point: The index of the target vehicle's trajectory in the working data frame.
    >>> X_now: Current X of the target vehicle (ft.).
    >>> Y_now: Current Y of the target vehicle (ft.).
    >>> speed_now: Current speed of the target vehicle (ft/s).
    >>> accel: Acceleration of the target vehicel (fpss).
    >>> heading: Current heading angle of the target vehicle.
    RETURN: Next X, Next Y, Next heading, Next speed, the index of the reference trajectory for next time stamp.

    """

    if len(data_check) <= 1:
        return (X_now, Y_now, heading, speed_now, point)
    else:
        Start_X = X_now
        Start_Y = Y_now
        Check_X = data_check.at[point, 'X']
        Check_Y = data_check.at[point, 'Y']
        count = 0
        dist1 = float("{:.6f}".format(speed_now * 0.1  + 0.5 * accel * 0.01))
        speed_next = float("{:.6f}".format(speed_now + 0.5 * accel * 0.1))
        dist2 = dist(Start_X, Start_Y, Check_X, Check_Y)
        if accel < 0:
            time0 = float("{:.6f}".format(-2 * speed_now  / accel))
            dist0 = float("{:.6f}".format(speed_now * time0  + 0.5 * accel * time0 * time0))
            if (time0 < 0.1) and (dist0 <= dist2):
                Heading = heading
                rad = math.pi / 2 - math.radians(Heading)
                Des_X = Start_X + dist0 * math.cos(rad)
                Des_Y = Start_Y + dist0 * math.sin(rad)
                return (Des_X, Des_Y, Heading, 0, point)
            elif (time0 < 0.1) and (dist0 > dist2):
                while (dist0 > dist2):
                    count += 1
                    if point + count < len(data_check):
                        dist0 = dist0 - dist2
                        Start_X = Check_X
                        Start_Y = Check_Y
                        Check_X = data_check.at[point + count, 'X']
                        Check_Y = data_check.at[point + count, 'Y']
                        dist2 = dist(Start_X, Start_Y, Check_X, Check_Y)
                    else:
                        break
                Heading = get_heading(Start_X, Start_Y, Check_X, Check_Y)
                rad = math.pi / 2 - math.radians(Heading)
                Des_X = Start_X + dist0 * math.cos(rad)
                Des_Y = Start_Y + dist0 * math.sin(rad)
                return (Des_X, Des_Y, Heading, 0, point + count)
            else:
                while (dist1 > dist2):
                    count += 1
                    if point + count < len(data_check):
                        dist1 = dist1 - dist2
                        Start_X = Check_X
                        Start_Y = Check_Y
                        Check_X = data_check.at[point + count, 'X']
                        Check_Y = data_check.at[point + count, 'Y']
                        dist2 = dist(Start_X, Start_Y, Check_X, Check_Y)
                        speed_next = float("{:.6f}".format(speed_next + 0.5 * accel * 0.1))
                    else:
                        break
                Heading = get_heading(Start_X, Start_Y, Check_X, Check_Y)
                rad = math.pi / 2 - math.radians(Heading)
                Des_X = Start_X + dist1 * math.cos(rad)
                Des_Y = Start_Y + dist1 * math.sin(rad)
                return (Des_X, Des_Y, Heading, speed_next, point + count)
        elif (accel == 0) & (speed_now == 0):
            return (Start_X, Start_Y, heading, speed_now, point)
        else:
            while (dist1 > dist2):
                count += 1
                if point + count < len(data_check):
                    dist1 = dist1 - dist2
                    Start_X = Check_X
                    Start_Y = Check_Y
                    Check_X = data_check.at[point + count, 'X']
                    Check_Y = data_check.at[point + count, 'Y']
                    dist2 = dist(Start_X, Start_Y, Check_X, Check_Y)
                    speed_next = float("{:.6f}".format(speed_next + 0.5 * accel * 0.1))
                else:
                    break
            Heading = get_heading(Start_X, Start_Y, Check_X, Check_Y)
            rad = math.pi / 2 - math.radians(Heading)
            Des_X = Start_X + dist1 * math.cos(rad)
            Des_Y = Start_Y + dist1 * math.sin(rad)
            return (Des_X, Des_Y, Heading, speed_next, point + count)

def next_location_online(data_check, point, X_now, Y_now, speed_now, accel, heading):
    """Returns next time step's location of the target vehicle without projections potential trajectory. 
       This is the online version, can be used for single step length updating process.
       Replace all the function [next_location] for the online version.

    Keyword arguments:
    >>> data_check: The working data frame selected from the main dataframe.
    >>> point: The index of the target vehicle's trajectory in the working data frame.
    >>> X_now: Current X of the target vehicle (ft.).
    >>> Y_now: Current Y of the target vehicle (ft.).
    >>> speed_now: Current speed of the target vehicle (ft/s).
    >>> accel: Acceleration of the target vehicel (fpss).
    >>> heading: Current heading angle of the target vehicle.
    RETURN: Next X, Next Y, Next heading, Next speed, the index of the reference trajectory for next time stamp.

    """
    Start_X = X_now
    Start_Y = Y_now
    dist1 = float("{:.6f}".format(speed_now * 0.1 + 0.5 * accel * 0.01))
    speed_next = float("{:.6f}".format(speed_now + 0.5 * accel * 0.1))

    if (accel == 0) & (speed_now == 0):
        return (Start_X, Start_Y, heading, speed_now, point + 1)
    elif accel < 0:
        time0 = float("{:.6f}".format(-2 * speed_now / accel))
        dist0 = float("{:.6f}".format(speed_now * time0 + 0.5 * accel * time0 * time0))
        if time0 < 0.1:
            Heading = heading
            rad = math.pi / 2 - math.radians(Heading)
            Des_X = Start_X + dist0 * math.cos(rad)
            Des_Y = Start_Y + dist0 * math.sin(rad)
            return (Des_X, Des_Y, Heading, 0, point + 1)
        else:
            Heading = heading
            rad = math.pi / 2 - math.radians(Heading)
            Des_X = Start_X + dist1 * math.cos(rad)
            Des_Y = Start_Y + dist1 * math.sin(rad)
            return (Des_X, Des_Y, Heading, speed_next, point + 1)
    else:
        Heading = heading
        rad = math.pi / 2 - math.radians(Heading)
        Des_X = Start_X + dist1 * math.cos(rad)
        Des_Y = Start_Y + dist1 * math.sin(rad)
        return (Des_X, Des_Y, Heading, speed_next, point + 1)

def overlap(shape1, shape2):
    """
    Checking overlap of two shapes.
    
    Keyword arguments:
    >>> shape1: list of for corners of vehicle1, sort by TL_x, TL_y, TR_x, TR_y, BL_x, BL_y, BR_x, BR_y
    >>> shape2: list of for corners of vehicle2, sort by TL_x, TL_y, TR_x, TR_y, BL_x, BL_y, BR_x, BR_y
    RETURN: True or False
    """

    p1 = Polygon([(shape1[0], shape1[1]), (shape1[2], shape1[3]), (shape1[4], shape1[5]), (shape1[6], shape1[7])])
    p2 = Polygon([(shape2[0], shape2[1]), (shape2[2], shape2[3]), (shape2[4], shape2[5]), (shape2[6], shape2[7])])
    return p1.intersects(p2)


def rectangular(x, y, length, width, angle, style):
    """Returns the coordinates of the four points of a vehicle (rectangular) given the center of the front bumper.
    
    Keyword arguments:
    >>> x: X of the reference point (ft.).
    >>> y: Y of the reference point (ft.).
    >>> length: Length of the vehicle (ft.).
    >>> width: Width of the vehicle (ft.).
    >>> angle: Heading of the vehicle.
    >>> style: Using front bumper or centroid as reference point (1:front bumper; 2: centroid)
    RETURN: Top-Left-x, Top-Left-y, Top-Right-x, Top-Right-y, Bottom-Left-x, Bottom-Left-y, Bottom-Right-x, Bottom-Right-y
    """

    if style == 1:
        # Radian of heading
        rad = math.pi / 2 - math.radians(angle)
        # Radian of 90 degree
        rad90 = math.atan2(1, 0)
        # Radian of length and half width
        t_rad = math.atan2(length, width/2)

        TL_x = float("{:.4f}".format(x + width/2 * math.cos(rad+rad90)))
        TL_y = float("{:.4f}".format(y + width/2 * math.sin(rad+rad90)))

        TR_x = float("{:.4f}".format(x + width/2 * math.cos(rad-rad90)))
        TR_y = float("{:.4f}".format(y + width/2 * math.sin(rad-rad90)))

        BR_x = float("{:.4f}".format(x + math.sqrt((width/2)**2+length**2) * math.cos(rad - rad90 - t_rad)))
        BR_y = float("{:.4f}".format(y + math.sqrt((width/2)**2+length**2) * math.sin(rad - rad90 - t_rad)))

        BL_x = float("{:.4f}".format(x + math.sqrt((width/2)**2+length**2) * math.cos(rad + rad90 + t_rad)))
        BL_y = float("{:.4f}".format(y + math.sqrt((width/2)**2+length**2) * math.sin(rad + rad90 + t_rad)))

        return [TL_x, TL_y, TR_x, TR_y, BR_x, BR_y, BL_x, BL_y]
    
    elif style == 2:
        # Radian of heading
        rad = math.pi / 2 - math.radians(angle)
        # Radian of 90 degree
        rad90 = math.atan2(1, 0)
        # Radian of length and half width
        t_rad_1 = math.atan2(length / 2, width / 2)
        t_rad_2 = math.atan2(width / 2, length / 2)

        TL_x = float("{:.4f}".format(x + math.sqrt((width/2)**2+(length/2)**2) * math.cos(rad + t_rad_2)))
        TL_y = float("{:.4f}".format(y + math.sqrt((width/2)**2+(length/2)**2) * math.sin(rad + t_rad_2)))

        TR_x = float("{:.4f}".format(x + math.sqrt((width/2)**2+(length/2)**2) * math.cos(rad - t_rad_2)))
        TR_y = float("{:.4f}".format(y + math.sqrt((width/2)**2+(length/2)**2) * math.sin(rad - t_rad_2)))

        BR_x = float("{:.4f}".format(x + math.sqrt((width/2)**2+(length/2)**2) * math.cos(rad - rad90 - t_rad_1)))
        BR_y = float("{:.4f}".format(y + math.sqrt((width/2)**2+(length/2)**2) * math.sin(rad - rad90 - t_rad_1)))

        BL_x = float("{:.4f}".format(x + math.sqrt((width/2)**2+(length/2)**2) * math.cos(rad + rad90 + t_rad_1)))
        BL_y = float("{:.4f}".format(y + math.sqrt((width/2)**2+(length/2)**2) * math.sin(rad + rad90 + t_rad_1)))

        return [TL_x, TL_y, TR_x, TR_y, BR_x, BR_y, BL_x, BL_y]

def main(Start_time, dataset, TTCD_thr, MC_Num, coor_style):
    """The main processing function.

    Keyword arguments:
    >>> Start_time: The processing time step.
    >>> dataset: The loaded trajectory data generated by TCA.
    >>> TTCD_thr: The TTCD threshold, float.
    >>> MC_Num: The number of Monte Carlo runs. 
    >>> coor_style: The reference point style of generating the vehicle's shape(1: front bumper; 2:centroid)
    RETURN: Time of detected conflict, Locations of involved vehicles, the TTCD scores for multiple conflict type
    """

    df = dataset
    Start_time = float(Start_time)
    Start_time = float("{:.1f}".format(Start_time))
    # Extract all vehicles and related data in this time step
    # Storing in working data frame df1, and working on df1
    df1 = df[df.transtime == Start_time]
    df1.index = pd.RangeIndex(start=0, stop=len(df1), step=1)
    print("Processing Time Step:", Start_time, "for ", len(df1.Vehicle_ID.unique()), " vehicles.")

    # Pass steps have only one vehicle
    if len(df1.Vehicle_ID.unique()) <= 1:
        print("Lonely car...")
    # Main processing
    else:
        # Calculate TTCmax location
        for i in range(len(df1)):
            if df1.at[i, 'Speed'] == 0.0:
                df_veh = df[(df.Vehicle_ID == df1.at[i, 'Vehicle_ID']) & (df.transtime < Start_time) & (df.Speed != 0.0)]
                if len(df_veh) != 0:
                    df_veh = df_veh.tail(1)
                    df_veh.index = pd.RangeIndex(start=0, stop=len(df_veh), step=1)
                    df1.at[i, 'Cal_heading'] = get_heading(df_veh.at[0, 'X'], df_veh.at[0, 'Y'], df1.at[i, 'X'], df1.at[i, 'Y'])
                else:
                    df_veh = df[(df.Vehicle_ID == df1.at[i, 'Vehicle_ID']) & (df.transtime > Start_time) & (df.Speed != 0.0)]
                    if len(df_veh) != 0:
                        df_veh.index = pd.RangeIndex(start=0, stop=len(df_veh), step=1)
                        df1.at[i, 'Cal_heading'] = get_heading(df1.at[i, 'X'], df1.at[i, 'Y'], df_veh.at[0, 'X'], df_veh.at[0, 'Y'])
                    else:
                        df1.at[i, 'Cal_heading'] = df1.at[i, 'Heading']
            else:
                # Find out all data related to current checking veh in 10 seconds data
                df_veh = df[(df.Vehicle_ID == df1.at[i, 'Vehicle_ID']) & (df.transtime > Start_time) & (df.Speed != 0.0)]
                if len(df_veh) != 0:
                    df_veh.index = pd.RangeIndex(start=0, stop=len(df_veh), step=1)
                    df1.at[i, 'Cal_heading'] = get_heading(df1.at[i, 'X'], df1.at[i, 'Y'], df_veh.at[0, 'X'],
                                                       df_veh.at[0, 'Y'])
                else:
                    df1.at[i, 'Cal_heading'] = df1.at[i, 'Heading']
        df1 = df1.dropna()
        df1.index = pd.RangeIndex(start=0, stop=len(df1), step=1)
        # Generate shape for each vehicle
        df1['Shape'] = ''
        df1['Shape'] = df1['Shape'].astype('object')
        pd.set_option('display.max_columns', None)
        for j in range(len(df1)):
            df1.at[j, 'Shape'] = rectangular(df1.at[j, 'X'], df1.at[j, 'Y'],
                                             float("{:.6f}".format(df1.at[j, 'length'] * 3.2804)),
                                             float("{:.6f}".format(df1.at[j, 'width'] * 3.2804)),
                                             df1.at[j, 'Cal_heading'], coor_style)
        # Overlap Checking, and fill the conflict table
        # Top-down iterating every vehicle in df1 and checking with each other.
        progress_count = 0
        factory_1 = math.factorial(len(df1))
        factory_2 = math.factorial(len(df1)-2)
        for p in range(len(df1)):
            for q in range(p+1, len(df1)):
                # Generate two random acceleration pools for Monte Carlo simulation, feet per second per second(fpss).
                # mu, sigma = 0, 2
                runs = MC_Num
                population = np.arange(start = -9.9, stop = 9.9, step = 0.2)
                # Pre-generated acceleration pool
                weights = [0.0000309,0.00003,0.0000382,0.0000388,0.0000488,0.0000435,0.0000465,0.0000524,0.0000597,0.0000582,0.0000615,
                           0.0000703,0.0000812,0.0000844,0.0000841,0.0000891,0.000104,0.000124,0.000113,0.000132,0.000152,0.000165,0.000173,
                           0.000201,0.000216,0.000219,0.00025,0.000269,0.000317,0.000371,0.000412,0.000466,0.000542,0.000621,0.000739,
                           0.000857,0.001072658,0.001384131,0.001861489,0.002492672,0.003418858,0.004820048,0.006600948,0.009066855,
                           0.012256299,0.016931934,0.02403024,0.03459064,0.046560171,0.287730619,0.287823466,0.052042873,0.045202217,
                           0.039347452,0.03254062,0.025447313,0.018770482,0.013356604,0.008993325,0.005907412,0.00384945,0.002362377,
                           0.001474427,0.00091,0.000574,0.000401,0.000276,0.000184,0.000113,0.0000726,0.0000503,0.0000326,0.0000274,
                           0.0000126,0.0000103,0.00000559,0.00000441,0.00000412,0.00000324,0.00000412,0.00000324,0.00000147,0.00000235,
                           0.00000206,0.00000118,0.00000118,0.000000882,0.00000118,0.000000882,0,0.000000882,0.000000588,0,0.000000882,
                           0.000000882,0,0,0.000000588,0.000000294]
                
                # Normal distribution approach
                # pandora_box_a = np.random.normal(mu, sigma, runs)
                # pandora_box_b = np.random.normal(mu, sigma, runs)

                # Nonparamatric approach
                pandora_box_a = np.random.choice(population, runs, p=weights)
                pandora_box_b = np.random.choice(population, runs, p=weights)
                #################################################################################################
                # This is the TTCD calculating part.
                if overlap(df1.at[p, 'Shape'], df1.at[q, 'Shape']):
                    print ("Vehicle ", df1.at[p, 'Vehicle_ID'], " and Vehicle ", df1.at[q, 'Vehicle_ID'], "crushed at time ", Start_time)
                    break

                # Find out when will the overlapped pair separate.
                else:
                    df_check1 = df[(df.Vehicle_ID == df1.at[p, 'Vehicle_ID']) & (df.transtime >= Start_time) & (df.transtime <= Start_time + 100)]
                    df_check1.index = pd.RangeIndex(start=0, stop=len(df_check1), step=1)
                    df_check2 = df[(df.Vehicle_ID == df1.at[q, 'Vehicle_ID']) & (df.transtime >= Start_time) & (df.transtime <= Start_time + 100)]
                    df_check2.index = pd.RangeIndex(start=0, stop=len(df_check2), step=1)
                                       
                    TTCD_count_RE = 0
                    TTCD_count_CR = 0
                    TTCD_count_LC = 0

                    for box_order in range (runs):
                        acc1 = pandora_box_a[box_order]
                        acc2 = pandora_box_b[box_order]

                        if (df_check1.at[0, 'Speed']) == 0 and (acc1 == 0.0):
                            dist1v = 0
                        else:
                            if acc1 < 0:
                                time1v = float("{:.6f}".format(-2 * df_check1.at[0, 'Speed'] * 1.4667  / acc1))
                                dist1v = float("{:.6f}".format(df_check1.at[0, 'Speed'] * 1.4667 * time1v  + 0.5 * acc1 * time1v * time1v))
                            else:
                                dist1v = float("{:.6f}".format(df_check1.at[0, 'Speed'] * 1.4667 * TTCD_thr  + 0.5 * acc1 * TTCD_thr * TTCD_thr))

                        if (df_check2.at[0, 'Speed']) == 0 and (acc2 == 0.0):
                            dist2v = 0
                        else:
                            if acc2 < 0:
                                time2v = float("{:.6f}".format(-2 * df_check2.at[0, 'Speed'] * 1.4667  / acc2))
                                dist2v = float("{:.6f}".format(df_check2.at[0, 'Speed'] * 1.4667 * time2v  + 0.5 * acc2 * time2v * time2v))
                            else:
                                dist2v = float("{:.6f}".format(df_check2.at[0, 'Speed'] * 1.4667 * TTCD_thr  + 0.5 * acc2 * TTCD_thr * TTCD_thr))

                        Loc_check1 = ttc_location(df_check1, dist1v, Start_time)
                        Loc_check2 = ttc_location(df_check2, dist2v, Start_time)

                        if dist(Loc_check1[0],Loc_check1[1],Loc_check2[0],Loc_check2[1])<50:
                            # Obtain the TTC location
                            Loc1 = next_location(df_check1, 1, df_check1.at[0, 'X'], df_check1.at[0, 'Y'], float("{:.6f}".format(df_check1.at[0, 'Speed'] * 1.4667)), acc1, df1.at[p, 'Cal_heading'])
                            Loc2 = next_location(df_check2, 1, df_check2.at[0, 'X'], df_check2.at[0, 'Y'], float("{:.6f}".format(df_check2.at[0, 'Speed'] * 1.4667)), acc2, df1.at[q, 'Cal_heading'])
                            for time_step in range(1, int(TTCD_thr*10)+1):
                                check_point = float("{:.1f}".format(time_step*0.1))
                                # Generate shape for each vehicle
                                Shape1 = rectangular(Loc1[0], Loc1[1], df1.at[p, 'length'] * 3.2804,
                                                     df1.at[p, 'width'] * 3.2804, Loc1[2], coor_style)
                                Shape2 = rectangular(Loc2[0], Loc2[1], df1.at[q, 'length'] * 3.2804,
                                                     df1.at[q, 'width'] * 3.2804, Loc2[2], coor_style)
                                Speed_1 = Loc1[3]
                                Speed_2 = Loc2[3]
                                X_new_1 = Loc1[0]
                                Y_new_1 = Loc1[1]
                                X_new_2 = Loc2[0]
                                Y_new_2 = Loc2[1]
                                point_1 = Loc1[4]
                                point_2 = Loc2[4]
                                heading_1 = Loc1[2]
                                heading_2 = Loc2[2]
                                if overlap(Shape1, Shape2):
                                    relative_angle = abs(float("{:.6f}".format(heading_1))
                                                         - float("{:.6f}".format(heading_2)))
                                    if relative_angle < 30:
                                        TTCD_count_RE += 1
                                    elif relative_angle > 85:
                                        TTCD_count_CR += 1
                                    else:
                                        TTCD_count_LC += 1
                                    break
                                else:
                                    if int(point_1) == len(df_check1):
                                        Loc1 = Loc1
                                    else:
                                        Loc1 = next_location(df_check1, point_1, X_new_1, Y_new_1, Speed_1, acc1, heading_1)
                                    if int(point_2) == len(df_check2):
                                        Loc2 = Loc2
                                    else:
                                        Loc2 = next_location(df_check2, point_2, X_new_2, Y_new_2, Speed_2, acc2, heading_2)
                        else:
                            pass

                    CRD_RE = TTCD_count_RE / runs
                    CRD_CR = TTCD_count_CR / runs
                    CRD_LC = TTCD_count_LC / runs
                    if (CRD_RE > 0) or (CRD_CR > 0) or (CRD_LC > 0):
                        return({'Time': Start_time, 'Involve_A': df1.at[p, 'Vehicle_ID'], 'x_A': df1.at[p, 'X'], 'y_A': df1.at[p, 'Y'],
                                                    'Involve_B': df1.at[q, 'Vehicle_ID'], 'x_B': df1.at[q, 'X'], 'y_B': df1.at[q, 'Y'],
                                                    'CRD_RE': CRD_RE, 'CRD_CR': CRD_CR, 'CRD_LC': CRD_LC })
                    else:
                        pass

                progress_count += 1
                # Tracking the progress
                # print("Progress: ", float("{:.2f}".format(progress_count/(math.factorial(len(df1))/(2*(math.factorial(len(df1)-2))))*100)),"%. Finished vehicle pair: ", df1.at[p, 'Vehicle_ID'], df1.at[q, 'Vehicle_ID'])
                # print("Progress: ", float("{:.2f}".format(progress_count / (factory_1 / (2 * (factory_2))) * 100)), "%")

if __name__ == "__main__":
    # Collecting the required info from user.
    traj_file = input("Please input the name of the trajectory file(*.csv):")
    ttcd_threshold = float("{:.1f}".format(input("Please input the TTCD threshold(1 digit float):")))
    mc_number = int(input("Please input the number of Monte Carlo simulation(int):"))
    reference_style = int(input("Please select the reference point style (Front bumper: 1; Centroid: 2)"))
    if reference_style != 1 or reference_style != 2:
        print("Error: You typed wrong option. The program will be terminated.")
        sys.exit()

    # Loading data from the trajectory file
    # !!!!! IMPORTANT !!!!!
    # Please make sure the units in the data are following:
    # Coordinates: feet(ft.), Speed: miles per hour(mph), length/width: meters(m), acceleration: feet per second per second(fpss)
    program_st = time.time()
    print("Start Loading %s" % (time.strftime('%X', time.localtime(program_st))))
    loaded_data = pd.read_csv(traj_file, usecols=['Vehicle_ID', 'transtime', 'X', 'Y', 'Speed',
                                                                       'Heading', 'length', 'width'])
    loaded_data = loaded_data.sort_values(by=['transtime', 'Vehicle_ID'])
    print("Before drop duplicates, data size is:", len(loaded_data))
    loaded_data = loaded_data.drop_duplicates(subset=['X', 'Y', 'Speed', 'Heading', 'transtime'], keep="first")
    print("After drop duplicates, data size is:", len(loaded_data))
    loaded_data.index = pd.RangeIndex(start=0, stop=len(loaded_data), step=1)
    ld_time = time.time()
    print("Finish Loading %s (%f)" % (time.strftime('%X', time.localtime(ld_time)), (ld_time - program_st)))

    program_st = time.time()
    print("*******************  Start Program  *******************")
    print("Start time %s" % (time.strftime('%X', time.localtime(program_st))))
    
    # If the trajectory file is too large, you can run it seperately by deciding the sub-interval.
    sub_interval = input("Do you want to use part of the trajectory file? (Yes: 1; No: 2)")
    if sub_interval == "1":
        Start_Point = float("{:.1f}".format(input("Please input the start time of the sub-interval(1 digit float): ")))
        End_Point = float("{:.1f}".format(input("Please input the end time of the sub-interval(1 digit float): ")))
        Time_Period = frange(Start_Point, End_Point, 0.1)
    elif sub_interval == "2":
        # Full time period
        Start_Point = float("{:.1f}".format(loaded_data.transtime.min()))
        End_Point = float("{:.1f}".format(loaded_data.transtime.max()))
        Time_Period = frange(Start_Point, End_Point, 0.1)
    else:
        print("Error: You typed wrong option. The program will be terminated.")
        sys.exit()

    # Generate an empty table for the final results
    Conflict = pd.DataFrame(columns=['Time', 'Involve_A', 'x_A', 'y_A', 'Involve_B', 'x_B', 'y_B','CRD_RE', 'CRD_CR', 'CRD_LC'])
    # Checking the number of the processors
    print ("Number of processors:", mp.cpu_count())
    # Parallel processing the main function
    processor_use = int(input("Please input the number of processors you want to use: (1-"+mp.cpu_count()+")"))
    if processor_use > mp.cpu_count() or processor_use <= 0:
        print("Error: You typed wrong number of the processors. The program will be terminated.")
        sys.exit()
    pool = mp.Pool(processor_use)

    results = pool.starmap(main, zip(Time_Period, repeat(loaded_data), repeat(ttcd_threshold), repeat(mc_number), repeat(coor_style)))
    clean_results = list(filter(None, results))
    print ("Total Conflict found: ", len(clean_results))
    if len(clean_results) != 0:
        Conflict = Conflict.append(clean_results, ignore_index=True)
        # Merge the continuous conflicts
        ConflictD = ConflictD.sort_values(by=['Involve_A', 'Involve_B', 'Time'])
        ConflictD.index = pd.RangeIndex(start=0, stop=len(ConflictD), step=1)

        ConflictD['Event_Combine'] = ''
        ConflictD.at[0, 'Event_Combine'] = 1
        event_num = 1

        for iter in range(1, len(ConflictD)):
            if (ConflictD.loc[iter, 'Involve_A'] == ConflictD.loc[iter-1, 'Involve_A']) & 
               (ConflictD.loc[iter, 'Involve_B'] == ConflictD.loc[iter-1, 'Involve_B']) & 
               (float("{:.1f}".format(ConflictD.loc[iter, 'transtime'] - ConflictD.loc[iter-1, 'transtime'])) == 0.1):
                ConflictD.at[iter, 'Event_Combine'] = event_num
            else:
                event_num += 1
                ConflictD.at[iter, 'Event_Combine'] = event_num
        # Generate the output file
        Conflict.to_csv("TTCD_Offline_"+str(ttcd_threshold)+"_"+str(mc_number)+"MC_"+"Time_"+str(Start_Point)+str(End_Point)+traj_file[:-4]+".csv")
    else:
        print("No Conflict detected!")

    ed_time = time.time()
    print("End time %s (%f)" % (time.strftime('%X', time.localtime(ed_time)), (ed_time - program_st)))
    print("*******************  End Program  *******************")
