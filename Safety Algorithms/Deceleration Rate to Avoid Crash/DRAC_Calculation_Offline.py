# C2SMART Lab, NYU
# NCHRP 03-137
# @file    DRAC_Calculation_Offline.py
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
    >>> y1: float value for Y for first point (ft.)
    >>> x2: float value for X for 2nd point (ft.)
    >>> y2: float value for Y for 2nd point (ft.)
    RETURN: The euclidean distance(float, ft.).
    """

    return float("{:.6f}".format(math.sqrt((x2-x1) ** 2 + (y2 - y1) ** 2)))



def get_heading(x1, y1, x2, y2):
    """
    Returns the Heading based on two points

    Keyword arguments:
    >>> x1: Float value for X for first point (ft.)
    >>> y1: Float value for Y for first point (ft.)
    >>> x2: Float value for X for 2nd point (ft.)
    >>> y2: Float value for Y for 2nd point (ft.)
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

def ttc_location_online(data_check, distance, start_time):
    """
    Returns TTCmax Location (Please see the document for the detailed definition) without projections potential trajectory.
    This is the online version, can be used for single step length updating process.
    Replace all the function [ttc_location] for the online version.

    Keyword arguments:
    >>> data_check: The working data frame selected from the main data frame.
    >>> distance: The projecting distance based on the current speed (ft.).
    >>> start_time: The time stamp of the processing step.
    RETURN: TTCmax point X, TTCmax point Y, the nearest time stamp before the TTCmax location projected, 
            heading of the vehicle at the TTCmax point.
    """

    dist1 = distance

    Start_X = data_check.at[0, 'X']
    Start_Y = data_check.at[0, 'Y']
    Check_X = data_check.at[1, 'X']
    Check_Y = data_check.at[1, 'Y']

    Heading = get_heading(Start_X, Start_Y, Check_X, Check_Y)
    rad = math.pi / 2 - math.radians(Heading)
    TTC_X = Start_X + dist1 * math.cos(rad)
    TTC_Y = Start_Y + dist1 * math.sin(rad)
    start_time = float("{:.1f}".format(start_time + 0.1))

    return [TTC_X, TTC_Y, float("{:.1f}".format(start_time - 0.1)), Heading]

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

def main(Start_time, dataset, coor_style):
    """The main processing function.

    Keyword arguments:
    >>> Start_time: The processing time step.
    >>> dataset: The loaded trajectory data generated by TCA.
    >>> coor_style: The reference point style of generating the vehicle's shape(1: front bumper; 2:centroid)
    RETURN: Time of detected conflict, Locations of involved vehicles, the DRAC value
    """

    df = dataset
    Start_time = float(Start_time)
    Start_time = float("{:.1f}".format(Start_time))
    # Extract all vehicles and related data in this time step
    # Storing in working data frame df1, and working on df2
    df2 = df[df.transtime == Start_time]
    df2 = df2.dropna(subset=['Speed'])
    df2 = df2.sort_values(by=['transtime'])
    df2.index = pd.RangeIndex(start=0, stop=len(df2), step=1)
    print("Processing Time Step:", Start_time, "Processing: ", len(df2), "Vehicle.")

    # Pass steps have only one vehicle
    if len(df2.Vehicle_ID.unique()) <= 1:
        print("Lonely car...")
        pass

    # Main processing
    else:
        for i in range(len(df2)):
            df_veh = df[(df.Vehicle_ID == df2.at[i, 'Vehicle_ID']) & (df.transtime < Start_time) & (df.Speed != 0.0)]
            if len(df_veh) != 0:
                df_veh = df_veh.tail(1)
                df_veh.index = pd.RangeIndex(start=0, stop=len(df_veh), step=1)
                df2.at[i, 'TTC_heading'] = get_heading(df_veh.at[0, 'X'], df_veh.at[0, 'Y'], df2.at[i, 'X'], df2.at[i, 'Y'])
            else:
                df_veh = df[(df.Vehicle_ID == df2.at[i, 'Vehicle_ID']) & (df.transtime > Start_time) & (df.Speed != 0.0)]
                df_veh.index = pd.RangeIndex(start=0, stop=len(df_veh), step=1)
                df2.at[i, 'TTC_heading'] = get_heading(df2.at[i, 'X'], df2.at[i, 'Y'], df_veh.at[0, 'X'], df_veh.at[0, 'Y'])

        #######################################################################################################
        # This is the DRAC calculation part

        # Calculate the DRAC threshold
        for p in range(len(df2)):
            for q in range(p+1, len(df2)):
                DRAC_TTC = float("{:.6f}".format(abs(df2.at[p, 'Speed'] - df2.at[q, 'Speed']) * 1.46667 / (2 * 8.2021)))

                Dist0D1 = df2.at[p, 'Speed'] * DRAC_TTC * 1.4667
                Dist0D2 = df2.at[q, 'Speed'] * DRAC_TTC * 1.4667

                df_check0D1 = df[(df.Vehicle_ID == df2.at[i, 'Vehicle_ID']) & (df.transtime >= Start_time) & (df.transtime <= Start_time + 100)]
                df_check0D1.index = pd.RangeIndex(start=0, stop=len(df_check0D1), step=1)
                df_check0D1 = df_check0D1.dropna(subset=['Speed'])
                df_check0D2 = df[(df.Vehicle_ID == df2.at[i, 'Vehicle_ID']) & (df.transtime >= Start_time) & (df.transtime <= Start_time + 100)]
                df_check0D2.index = pd.RangeIndex(start=0, stop=len(df_check0D2), step=1)
                df_check0D2 = df_check0D2.dropna(subset=['Speed'])

                # Obtain the projected DRAC-TTC location
                Loc0D1 = ttc_location(df_check0D1, Dist0D1, Start_time)
                Loc0D2 = ttc_location(df_check0D2, Dist0D2, Start_time)

                # Select the vehicle pairs within a distance threshold (ft.) to improve the computing efficiency
                if dist(Loc0D1[0],Loc0D1[1],Loc0D2[0],Loc0D2[1]) < 50:
                    # DRAC threshold is 8.2021 ft/s^2 (2.5 m/s^2)
                    DRAC_TTC = float("{:.6f}".format(abs(df2.at[p, 'Speed'] - df2.at[q, 'Speed']) * 1.46667 / (2 * 8.2021)))
                    Dist1D = df2.at[p, 'Speed'] * DRAC_TTC * 1.4667
                    Dist2D = df2.at[q, 'Speed'] * DRAC_TTC * 1.4667

                    df_check1D = df[(df.Vehicle_ID == df2.at[p, 'Vehicle_ID']) & (df.transtime >= Start_time) & (df.transtime <= Start_time + 100)]
                    df_check1D.index = pd.RangeIndex(start=0, stop=len(df_check1D), step=1)
                    df_check2D = df[(df.Vehicle_ID == df2.at[q, 'Vehicle_ID']) & (df.transtime >= Start_time) & (df.transtime <= Start_time + 100)]
                    df_check2D.index = pd.RangeIndex(start=0, stop=len(df_check2D), step=1)
                    df_check1D = df_check1D.dropna(subset=['Speed'])
                    df_check2D = df_check2D.dropna(subset=['Speed'])

                    # Obtain the TTC location
                    Loc1D = ttc_location(df_check1D, Dist1D, Start_time)
                    Loc2D = ttc_location(df_check2D, Dist2D, Start_time)
                    Loc1D1 = Loc1D
                    Loc2D1 = Loc2D

                    # Generate shape for each vehicle
                    if (df2.at[p, 'Speed'] == 0.0) & (df2.at[q, 'Speed'] != 0.0) & ((np.NaN in Loc2D) is False):
                        Shape1D = rectangular(df2.at[p, 'X'], df2.at[p, 'Y'],
                                              df2.at[p, 'length'] * 3.2804,
                                              df2.at[p, 'width'] * 3.2804,
                                              df2.at[p, 'TTC_heading'], coor_style)
                        Shape2D = rectangular(Loc2D[0], Loc2D[1], df2.at[q, 'length'] * 3.2804,
                                                                  df2.at[q, 'width'] * 3.2804, Loc2D[3], coor_style)
                    elif (df2.at[p, 'Speed'] != 0.0) & (df2.at[q, 'Speed'] == 0.0) & ((np.NaN in Loc1D) is False):
                        Shape1D = rectangular(Loc1D[0], Loc1D[1], df2.at[p, 'length'] * 3.2804,
                                                                  df2.at[p, 'width'] * 3.2804, Loc1D[3], coor_style)
                        Shape2D = rectangular(df2.at[q, 'X'], df2.at[q, 'Y'],
                                              df2.at[q, 'length'] * 3.2804,
                                              df2.at[q, 'width'] * 3.2804,
                                              df2.at[q, 'TTC_heading'], coor_style)
                    elif (df2.at[p, 'Speed'] == 0.0) & (df2.at[q, 'Speed'] == 0.0):
                        Shape1D = rectangular(df2.at[p, 'X'], df2.at[p, 'Y'],
                                              df2.at[p, 'length'] * 3.2804,
                                              df2.at[p, 'width'] * 3.2804,
                                              df2.at[p, 'TTC_heading'], coor_style)
                        Shape2D = rectangular(df2.at[q, 'X'], df2.at[q, 'Y'],
                                              df2.at[q, 'length'] * 3.2804,
                                              df2.at[q, 'width'] * 3.2804,
                                              df2.at[q, 'TTC_heading'], coor_style)
                    elif (df2.at[p, 'Speed'] != 0.0) & (df2.at[q, 'Speed'] != 0.0) & ((np.NaN in Loc1D) is False) & ((np.NaN in Loc2D) is False):
                        Shape1D = rectangular(Loc1D[0], Loc1D[1], df2.at[p, 'length'] * 3.2804,
                                                                  df2.at[p, 'width'] * 3.2804, Loc1D[3], coor_style)
                        Shape2D = rectangular(Loc2D[0], Loc2D[1], df2.at[q, 'length'] * 3.2804,
                                                                  df2.at[q, 'width'] * 3.2804, Loc2D[3], coor_style)
                    else:
                        break

                    if overlap(Shape1D, Shape2D):
                        while DRAC_TTC > 0.0:
                            DRAC_TTC = float("{:.6f}".format(DRAC_TTC - 0.1))
                            Back_dist1D = df2.at[p, 'Speed'] * DRAC_TTC * 1.46667
                            Back_dist2D = df2.at[q, 'Speed'] * DRAC_TTC * 1.46667

                            # Obtain the TTC location
                            Loc1D = ttc_location(df_check1D, Back_dist1D, Start_time)
                            Loc2D = ttc_location(df_check2D, Back_dist2D, Start_time)

                            # Generate shape for each vehicle
                            if (df2.at[p, 'Speed'] == 0.0) & (df2.at[q, 'Speed'] != 0.0):
                                Shape1D = rectangular(df2.at[p, 'X'], df2.at[p, 'Y'],
                                                      df2.at[p, 'length'] * 3.2804,
                                                      df2.at[p, 'width'] * 3.2804,
                                                      df2.at[p, 'TTC_heading'], coor_style)
                                Shape2D = rectangular(Loc2D[0], Loc2D[1], df2.at[q, 'length'] * 3.2804,
                                                                          df2.at[q, 'width'] * 3.2804, Loc2D[3], coor_style)
                                Angle1D = df2.at[p, 'TTC_heading']
                                Angle2D = Loc2D[3]
                            elif (df2.at[p, 'Speed'] != 0.0) & (df2.at[q, 'Speed'] == 0.0):
                                Shape1D = rectangular(Loc1D[0], Loc1D[1], df2.at[p, 'length'] * 3.2804,
                                                                          df2.at[p, 'width'] * 3.2804, Loc1D[3], coor_style)
                                Shape2D = rectangular(df2.at[q, 'X'], df2.at[q, 'Y'],
                                                      df2.at[q, 'length'] * 3.2804,
                                                      df2.at[q, 'width'] * 3.2804,
                                                      df2.at[q, 'TTC_heading'], coor_style)
                                Angle1D = Loc1D[3]
                                Angle2D = df2.at[q, 'TTC_heading']
                            elif (df2.at[p, 'Speed'] == 0.0) & (df2.at[q, 'Speed'] == 0.0):
                                Shape1D = rectangular(df2.at[p, 'X'], df2.at[p, 'Y'],
                                                      df2.at[p, 'length'] * 3.2804,
                                                      df2.at[p, 'width'] * 3.2804,
                                                      df2.at[p, 'TTC_heading'], coor_style)
                                Shape2D = rectangular(df2.at[q, 'X'], df2.at[q, 'Y'],
                                                      df2.at[q, 'length'] * 3.2804,
                                                      df2.at[q, 'width'] * 3.2804,
                                                      df2.at[q, 'TTC_heading'], coor_style)
                                Angle1D = df2.at[p, 'TTC_heading']
                                Angle2D = df2.at[q, 'TTC_heading']
                            else:
                                Shape1D = rectangular(Loc1D[0], Loc1D[1], df2.at[p, 'length'] * 3.2804,
                                                                          df2.at[p, 'width'] * 3.2804, Loc1D[3], coor_style)
                                Shape2D = rectangular(Loc2D[0], Loc2D[1], df2.at[q, 'length'] * 3.2804,
                                                                          df2.at[q, 'width'] * 3.2804, Loc2D[3], coor_style)
                                Angle1D = Loc1D[3]
                                Angle2D = Loc2D[3]
                            if overlap(Shape1D, Shape2D):
                                if (df2.at[p, 'Speed'] == 0.0) & (df2.at[q, 'Speed'] == 0.0):
                                    print("Conflict(DRAC) at:", Start_time, ", involved ", df2.at[p, 'Vehicle_ID'], "and",
                                          df2.at[q, 'Vehicle_ID'], ", DRAC:", float('inf'))
                                    return({'Time': Start_time, 'Involve_A': df2.at[p, 'Vehicle_ID'], 'x_A': df2.at[p, 'X'], 'y_A': df2.at[p, 'Y'],
                                                                'Involve_B': df2.at[q, 'Vehicle_ID'], 'x_B': df2.at[q, 'X'], 'y_B': df2.at[q, 'Y'],
                                                                'DRAC': float('inf'),
                                                                'Relative_Angle': abs(float("{:.1f}".format(df2.at[p, 'TTC_heading'])) - float("{:.1f}".format(df2.at[q, 'TTC_heading'])))})
                                    break
                                else:
                                    if DRAC_TTC <= 0.0:
                                        print("Conflict(DRAC) at:", Start_time, ", involved ", df2.at[p, 'Vehicle_ID'],
                                              "and", df2.at[q, 'Vehicle_ID'], ", DRAC:", float('inf'))
                                        return(
                                            {'Time': Start_time, 'Involve_A': df2.at[p, 'Vehicle_ID'], 'x_A': df2.at[p, 'X'], 'y_A': df2.at[p, 'Y'],
                                                                 'Involve_B': df2.at[q, 'Vehicle_ID'], 'x_B': df2.at[q, 'X'], 'y_B': df2.at[q, 'Y'],
                                                                 'DRAC': float('inf'), 
                                                                 'Relative_Angle': abs(float("{:.1f}".format(Loc1D[3])) - float("{:.1f}".format(Loc2D[3])))})
                                    pass
                            else:
                                # print (Shape1D, Shape2D, Loc1D[3],Loc2D[3])
                                print("Conflict(DRAC) at:", Start_time, ", involved ", df2.at[p, 'Vehicle_ID'], "and",
                                      df2.at[q, 'Vehicle_ID'], ", DRAC:", float("{:.1f}".format(abs(df2.at[p, 'Speed'] - df2.at[q, 'Speed']) * 1.4667 / (2 * (DRAC_TTC+0.1)))))
                                return({'Time': Start_time, 'Involve_A': df2.at[p, 'Vehicle_ID'], 'x_A': Loc1D1[0], 'y_A': Loc1D1[1],
                                                            'Involve_B': df2.at[q, 'Vehicle_ID'], 'x_B': Loc2D1[0], 'y_B': Loc2D1[1],
                                                            'DRAC': float("{:.1f}".format(abs(df2.at[p, 'Speed'] - df2.at[q, 'Speed']) * 1.4667 / (2 * (DRAC_TTC+0.1)))),
                                                            'Relative_Angle': abs(float("{:.1f}".format(Loc1D[3])) - float("{:.1f}".format(Loc2D[3])))})
                                # print (Conflict)
                                break
                            Loc1D1 = Loc1D
                            Loc2D1 = Loc2D
                    else:
                        pass
                else:
                    pass

if __name__ == "__main__":
    # Collecting the required info from user.
    traj_file = input("Please input the name of the trajectory file(*.csv):")
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
    ConflictD =  pd.DataFrame(columns=['Time', 'Involve_A', 'x_A', 'y_A', 'Involve_B', 'x_B','y_B','DRAC', 'Relative_Angle'])
    # Checking the number of the processors
    print ("Number of processors:", mp.cpu_count())
    # Parallel processing the main function
    processor_use = int(input("Please input the number of processors you want to use: (1-"+mp.cpu_count()+")"))
    if processor_use > mp.cpu_count() or processor_use <= 0:
        print("Error: You typed wrong number of the processors. The program will be terminated.")
        sys.exit()
    pool = mp.Pool(processor_use)

    results = pool.starmap(main, zip(Time_Period, repeat(loaded_data), repeat(coor_style)))
    clean_results = list(filter(None, results))
    print ("Total Conflict found: ", len(clean_results))
    if len(clean_results) != 0:
        ConflictD = ConflictD.append(clean_results, ignore_index=True)
        # Select the conflict angle lower than 30 degrees for the rear-end conflict
        index_names = ConflictD[(ConflictD['Relative_Angle'] > 30)].index 
        ConflictD.drop(index_names, inplace = True)
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
        Conflict.to_csv("DRAC_Offline_"+"Time_"+str(Start_Point)+str(End_Point)+traj_file[:-4]+".csv")
    else:
        print ("No conflict founded.")
    
    ed_time = time.time()
    print("End time %s (%f)" % (time.strftime('%X', time.localtime(ed_time)), (ed_time - program_st)))
    print("*******************  End Program  *******************")


