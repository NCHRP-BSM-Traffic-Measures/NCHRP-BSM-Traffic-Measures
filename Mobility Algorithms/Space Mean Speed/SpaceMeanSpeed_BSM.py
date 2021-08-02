import argparse
import numpy as np
import os
import math
import time
from bsm_stream import BSM, BSMStream

"""Estimate Space Mean Speed for given stretches of roadway defined as Superlinks using Basic Safety Messages.
"""


TIME_WINDOW = 5 #ft
DISTANCE_WINDOW = 20 #ft
EVALUATION_INTERVAL = 30 #Measure Travel Time every 30 seconds
MOVEMENT_INTERVAL = 5 #Move hypothetical vehicles every 5 seconds
MAX_TIME = 240 #secs

class Vehicle():
    """Hypothetical vehicle class stores information as a hypothetical vehicle traverses a route to estimate the travel time."""

    def __init__(self, superlinks, superlink, link_endpoints, tp):
        """ Initialize a new hypothetical vehicle on the given superlink to track travel time
            
            Arguments:
                - superlinks: The dictionary of superlinks, where each superlink is a list of Measures Estimation links
                - superlink: The index in superlinks of the specific superlink a hypothetical vehicle is being created for
                - link_endpoints: A dictionary of X,Y points representing the origin and destination of Measures Estimation links
                - tp: The current evaluation time period
        """
        self.SuperlinkNum = superlink
        self.Superlink = superlinks[superlink]
        self.CurrentLink = superlinks[superlink][0]
        self.CurrentX = link_endpoints[self.CurrentLink]['origin_x']
        self.CurrentY = link_endpoints[self.CurrentLink]['origin_y']
        self.DestinationX = link_endpoints[self.CurrentLink]['destination_x']
        self.DestinationY = link_endpoints[self.CurrentLink]['destination_y']
        x_diff = self.DestinationX - self.CurrentX
        y_diff = self.DestinationY - self.CurrentY
        self.CurrentHeading = math.atan2(y_diff,x_diff)
        self.DistanceToTravel = (x_diff**2 + y_diff**2)**.5
        self.CurrentTime = tp
        self.StartTime = tp
        self.EndTime = None 

def read_full_superlinks(filename):
    """Read the full superlinks input file and create a dictionary with superlink number as the key and a list of Measures Estimation links that make it up as the value"""
    superlinks = {}

    with open(filename) as in_f:
        for line in in_f:
            row = line.split(',')
            superlinks[int(row[0])] = [x.strip() for x in row[1:]]
    return superlinks

def read_link_endpoints(filename):
    """Read the link endpoints input file and create a dictionary with the Measures Estimation link number as the key and a dictionary of the origin and desination X,Y points as the value"""
    link_endpoints = {}
    with open(filename) as in_f:
        isheader = True
        for line in in_f:
            if isheader:
                isheader = False
                continue
            row = line.strip().split(',')
            link_number = row[5]
            origin_x = float(row[1])
            origin_y = float(row[2])
            destination_x = float(row[3])
            destination_y = float(row[4])
            if link_number not in link_endpoints:
                link_endpoints[link_number] = {'origin_x': origin_x,
                            'origin_y': origin_y,
                            'destination_x': destination_x,
                            'destination_y': destination_y
                }
            else:
                link_endpoints[link_number]['destination_x'] = destination_x
                link_endpoints[link_number]['destination_y'] = destination_y
    return link_endpoints

def calculate_superlink_lengths(superlinks, link_endpoints):
    """Calculate the length of each superlink and store it in a dictionary"""
    superlink_lengths = {}
    for superlink in superlinks:
        length = 0
        for link in superlink:
            link_data = link_endpoints[link]
            link_length = ((link_data['desination_x'] - link_data['origin_x'])**2 + (link_data['destination_y'] - link_data['origin_y'])**2)**0.5
            length += link_length
        superlink_lengths[superlink] = length

def initialize_vehicles(superlinks, link_endpoints, tp, hypothetical_vehicles):
    """Initialize a new group of hypothetical vehicles on each superlink for the given time period"""
    for superlink in superlinks:
        hypothetical_vehicles.append(Vehicle(superlinks, superlink, link_endpoints, tp))
        
def run_space_mean_speed(bsm_stream, superlinks, link_endpoints, superlink_lengths, spacemeanspeed_output_name, timing_output_name):
    """ Read BSM stream, initialize and move hypothetical vehicles along superlinks to estimate travel time at current time. Divide travel time by superlink length to get space mean speed and output it to file.
        
        Arguments:
            - bsm_stream: A BSMStream object that yields a list of Basic Safety Messages that were generated for a given time tp, in chronological order
            - superlinks: The dictionary of superlinks
            - link_endpoints: The dictionary of link endpoints
            - superlink_lengths: The dictionary of superlink lengths
            - spacemeanspeed_output_name: The filename to write space mean speed data to
            - timing_output_name: The filename to write performance execution timing data to
    """
    bsms_list = []
    hypothetical_vehicles = []
    timing = []

    with open(spacemeanspeed_output_name, 'w') as out_f:
        out_f.write('Superlink,CurrentTime,space_mean_speed\n')
        for tp, bsms in bsm_stream.read():
            loop_timer = time.time()
            bsms_list += bsms
            
            if tp % EVALUATION_INTERVAL == 0:
                initialize_vehicles(superlinks,link_endpoints,tp,hypothetical_vehicles)

            if tp % MOVEMENT_INTERVAL == 0:
                bsms_array = np.array(bsms_list)
                bsms_array = bsms_array[(bsms_array[:,BSM.SuperlinkIndex] > 0)]
                bsms_array = bsms_array[(bsms_array[:,BSM.TimeIndex] > tp - MAX_TIME)]
                bsms_list = bsms_array.tolist()
                completedvehicles = []
                for vehicle in hypothetical_vehicles:
                    bsms_inrange = get_bsms(vehicle,bsms_array,TIME_WINDOW,DISTANCE_WINDOW)
                    if bsms_inrange.shape[0] == 0:
                        if tp - vehicle.CurrentTime > MAX_TIME:
                            vehicle.EndTime = 'NA'
                            out_f.write("{},{},{}\n".format(vehicle.SuperlinkNum,vehicle.StartTime, vehicle.EndTime))
                            completedvehicles.append(vehicle)
                        else:
                            continue
                    else:
                        bsms_inrange = np.hstack((bsms_inrange,(((((bsms_inrange[:,BSM.XIndex:BSM.XIndex+1] - vehicle.CurrentX)**2 + 
                            (bsms_inrange[:,BSM.YIndex:BSM.YIndex + 1] - vehicle.CurrentY)**2)**.5)/((bsms_inrange[:,BSM.SpeedIndex:BSM.SpeedIndex + 1])**2 + 1) + 
                            (bsms_inrange[:,BSM.TimeIndex:BSM.TimeIndex + 1] - vehicle.CurrentTime)**2)**.5)))
                        bsms_top = bsms_inrange[np.argsort(bsms_inrange[:,-1], axis=0)][0:8,:]
                        bsms_top[:,-1] = 1/bsms_top[:,-1]
                        new_speed = np.sum(bsms_top[:,BSM.SpeedIndex] * bsms_top[:,-1])/np.sum(bsms_top[:,-1])
                        distance_traveled = new_speed * MOVEMENT_INTERVAL
                        vehicle.DistanceToTravel -= distance_traveled
                        if vehicle.DistanceToTravel <= 0:
                            if reached_destination(vehicle, distance_traveled, MOVEMENT_INTERVAL,link_endpoints):
                                out_f.write("{},{},{}\n".format(vehicle.SuperlinkNum, vehicle.StartTime, 
                                                                superlink_lengths[vehicle.SuperlinkNum] / (vehicle.EndTime - vehicle.StartTime)))
                                completedvehicles.append(vehicle)
                        else:
                            update_vehicle(vehicle, distance_traveled, MOVEMENT_INTERVAL)
                for vehicle in completedvehicles:
                    hypothetical_vehicles.remove(vehicle)
                timing.append((tp, time.time() - loop_timer))
    
    with open(timing_output_name, 'w') as out_f:
        out_f.write("simulation_time, loop_time")
        for time_result in timing:
            out_f.write("{},{}\n".format(time_result[0],time_result[1]))

def get_bsms(vehicle, bsms_array, time_max, distance_max):
    """ Return the BSMs that are on the current superlink within a set time and distance window of the hypothetical vehicle's location.
        
        Arguments:
            - vehicle: The hypothetical vehicle that is being updated and needs the closest BSMs for
            - bsms_array: A numpy array of recent BSM messages received
            - time_max: The current time search window size in seconds
            - distance_max: The current distance window size in feet 

        Returns:
            A numpy array of bsms on the hypothetical vehicle's superlink within time_max and distance_max of the hypothetical vehicle's time and distance. time_max and distance_max are expanded until at least one BSM is found.
    """

    bsms_onsuperlink = bsms_array[(bsms_array[:,BSM.SuperlinkIndex] == vehicle.SuperlinkNum)]
    mask = (np.abs(bsms_onsuperlink[:,BSM.TimeIndex] - vehicle.CurrentTime) <= time_max) & (((bsms_onsuperlink[:,BSM.XIndex] - vehicle.CurrentX)**2 + (bsms_onsuperlink[:,BSM.YIndex] - vehicle.CurrentY)**2)**.5 <= distance_max)
    while len(np.nonzero(mask)[0]) == 0 and time_max <= MAX_TIME:
        time_max += time_max
        distance_max += distance_max
        mask = (np.abs(bsms_onsuperlink[:,BSM.TimeIndex] - vehicle.CurrentTime) <= time_max) & (((bsms_onsuperlink[:,BSM.XIndex] - vehicle.CurrentX)**2 + (bsms_onsuperlink[:,BSM.YIndex] - vehicle.CurrentY)**2)**.5 <= distance_max)
    bsms_inrange = bsms_onsuperlink[mask]
    return bsms_inrange

def update_vehicle(vehicle, distance_traveled, time):
    """Update the hypothetical vehicle's X,Y position and time based on the estimated distance traveled at the averaged speed of the surrounding BSMs."""
    vehicle.CurrentX += distance_traveled * math.cos(vehicle.CurrentHeading)
    vehicle.CurrentY += distance_traveled * math.sin(vehicle.CurrentHeading)
    vehicle.CurrentTime += time

def move_to_next_link(vehicle,link_endpoints):
    """Determine the next link in the superlink that the hypothetical vehicle needs to traverse, if there are no more links, mark the end time."""
    try:
        vehicle.CurrentLink = vehicle.Superlink[vehicle.Superlink.index(vehicle.CurrentLink) + 1]
    except IndexError:
        vehicle.EndTime = vehicle.CurrentTime
        return True
    vehicle.DestinationX = link_endpoints[vehicle.CurrentLink]['destination_x']
    vehicle.DestinationY = link_endpoints[vehicle.CurrentLink]['destination_y']
    x_diff = vehicle.DestinationX - vehicle.CurrentX
    y_diff = vehicle.DestinationY - vehicle.CurrentY
    vehicle.CurrentHeading = math.atan2(y_diff,x_diff)
    vehicle.DistanceToTravel = (x_diff**2 + y_diff**2)**.5
    return False

def reached_destination(vehicle, distance_traveled, time, link_endpoints):
    """Move hypothtical vehicle to next link in superlink, by removing distance and time traveled to the end of the current link and applying the remainder to the next link."""
    while vehicle.DistanceToTravel <= 0:
        distance_to_end = distance_traveled + vehicle.DistanceToTravel
        partial_time = MOVEMENT_INTERVAL * (distance_to_end/distance_traveled)
        update_vehicle(vehicle, distance_to_end, partial_time)
        if move_to_next_link(vehicle, link_endpoints):
            return True
        distance_traveled -= distance_to_end
        time -= partial_time
        update_vehicle(vehicle, distance_traveled, time)
        vehicle.DistanceToTravel -= distance_traveled
    return False

def main():
    """Parse command line arguments, create BSM stream, superlinks dictionary, link endpoints dictionary and superlink lengths dictionary then start space mean speed evaluation."""
    parser = argparse.ArgumentParser(description='Measures Estimation program for reading in BSMs and producing Space Mean Speed values')
    parser.add_argument('bsm_filename') # CSV file of Basic Safety Messages
    parser.add_argument('fullsuperlinks_filename') # CSV file of full superlinks
    parser.add_argument('links_filename') # CSV file of links in network
    parser.add_argument('link_endpoints_filename') # CSV file of link end points
    parser.add_argument('timing_output_filename')
    parser.add_argument('--out', help = 'Output csv file (include .csv)')  
    args = parser.parse_args()

    dir_path = os.path.dirname( os.path.realpath( __file__ ) )

    bsm_stream = BSMStream(filename = args.bsm_filename, links_filename = args.links_filename, add_link = 'superlink', time_bucket = 30)

    superlinks = read_full_superlinks(args.fullsuperlinks_filename)

    link_endpoints = read_link_endpoints(args.link_endpoints_filename)

    if args.out:
        out_file = dir_path + '/' + args.out

    else:
        out_file = dir_path + '/spacemeanspeed_bsm.csv'

    superlink_lengths = calculate_superlink_lengths(superlinks, link_endpoints)

    run_space_mean_speed(bsm_stream, superlinks, link_endpoints, superlink_lengths, out_file, args.timing_output_filename)

if __name__ == "__main__":
    main()