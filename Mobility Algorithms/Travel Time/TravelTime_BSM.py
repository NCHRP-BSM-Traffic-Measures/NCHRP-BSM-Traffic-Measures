import argparse
import numpy as np
import os
import math
import time
from bsm_stream import BSM, BSMStream

"""Estimate Travel Time for given stretches of roadway defined as Routes using Basic Safety Messages."""

TIME_WINDOW = 20 #ft
DISTANCE_WINDOW = 100 #ft
EVALUATION_INTERVAL = 30 #Measure Travel Time every 30 seconds
MOVEMENT_INTERVAL = 5 #Move hypothetical vehicles every 5 seconds
MAX_TIME = 240 #seconds

class Vehicle():
"""Hypothetical vehicle class stores information as a hypothetical vehicle traverses a route to estimate the travel time."""
    def __init__(self, routes, route, link_endpoints, tp):
        """ Initialize a new hypothetical vehicle on the given route to track travel time
            
            Arguments:
                - routes: The dictionary of routes, where each route is defined as a list of Measures Estimation links
                - route: The index in routes of the specific route a hypothetical vehicle is being created for
                - link_endpoints: A dictionary of X,Y points representing the origin and destination of Measures Estimation links
                - tp: The current evaluation time period
        """
        self.RouteNum = route
        self.Route = routes[route]
        self.CurrentLink = routes[route][0]
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

def read_full_routes(filename):
    """Read the full routes input file and create a dictionary with route number as the key and a list of Measures Estimation links that make it up as the value"""
    routes = {}

    with open(filename) as in_f:
        for line in in_f:
            row = line.split(',')
            routes[int(row[0])] = [x.strip() for x in row[1:]]
    return routes

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


def initialize_vehicles(routes, link_endpoints, tp, hypothetical_vehicles):
    """Initialize a new group of hypothetical vehicles on each route for the given time period"""
    for route in routes:
        hypothetical_vehicles.append(Vehicle(routes, route, link_endpoints, tp))
        
def run_travel_time(bsm_stream, routes, link_endpoints, traveltime_output_name, timing_output_name):
    """ Read BSM stream, initialize and move hypothetical vehicles along routes to estimate travel time at current time and output it to file.
        
        Arguments:
            - bsm_stream: A BSMStream object that yields a list of Basic Safety Messages that were generated for a given time tp, in chronological order
            - routes: The dictionary of routes
            - link_endpoints: The dictionary of link endpoints
            - traveltime_output_name: The filename to write space mean speed data to
            - timing_output_name: The filename to write performance execution timing data to
    """
    bsms_list = []
    hypothetical_vehicles = []
    timing = []

    with open(traveltime_output_name, 'w') as out_f:
        out_f.write('route, simulation_time, average_travel_time\n')
        for tp, bsms in bsm_stream.read():
            loop_timer = time.time()
            bsms_list += bsms

            if tp % EVALUATION_INTERVAL == 0:
                initialize_vehicles(routes,link_endpoints,tp,hypothetical_vehicles)

            if tp % MOVEMENT_INTERVAL == 0:
                bsms_array = np.array(bsms_list)
                bsms_array = bsms_array[(bsms_array[:,BSM.RouteIndex] > 0)]
                bsms_array = bsms_array[(bsms_array[:,BSM.TimeIndex] > tp - MAX_TIME)]
                bsms_list = bsms_array.tolist()
                completedvehicles = []
                for vehicle in hypothetical_vehicles:
                    bsms_inrange = get_bsms(vehicle,bsms_array,TIME_WINDOW, DISTANCE_WINDOW)
                    if bsms_inrange.shape[0] == 0:
                        if tp - vehicle.CurrentTime > MAX_TIME:
                            vehicle.EndTime = 'NA'
                            out_f.write("{},{},{}\n".format(vehicle.RouteNum,vehicle.StartTime, vehicle.EndTime))
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
                                out_f.write("{},{},{}\n".format(vehicle.RouteNum,vehicle.StartTime, 
                                    vehicle.EndTime - vehicle.StartTime))
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
    """ Return the BSMs that are on the current route within a set time and distance window of the hypothetical vehicle's location.
        
        Arguments:
            - vehicle: The hypothetical vehicle that is being updated and needs the closest BSMs for
            - bsms_array: A numpy array of recent BSM messages received
            - time_max: The current time search window size in seconds
            - distance_max: The current distance window size in feet 

        Returns:
            A numpy array of bsms on the hypothetical vehicle's route within time_max and distance_max of the hypothetical vehicle's time and distance. time_max and distance_max are expanded until at least one BSM is found.
    """
    bsms_onroute = bsms_array[(bsms_array[:,BSM.RouteIndex] == vehicle.RouteNum)]
    mask = (np.abs(bsms_onroute[:,BSM.TimeIndex] - vehicle.CurrentTime) <= time_max) & (((bsms_onroute[:,BSM.XIndex] - vehicle.CurrentX)**2 + (bsms_onroute[:,BSM.YIndex] - vehicle.CurrentY)**2)**.5 <= distance_max)
    while len(np.nonzero(mask)[0]) == 0 and time_max <= MAX_TIME:
        time_max += time_max
        distance_max += distance_max
        mask = (np.abs(bsms_onroute[:,BSM.TimeIndex] - vehicle.CurrentTime) <= time_max) & (((bsms_onroute[:,BSM.XIndex] - vehicle.CurrentX)**2 + (bsms_onroute[:,BSM.YIndex] - vehicle.CurrentY)**2)**.5 <= distance_max)
    bsms_inrange = bsms_onroute[mask]
    return bsms_inrange

def update_vehicle(vehicle, distance_traveled, time):
    """Update the hypothetical vehicle's X,Y position and time based on the estimated distance traveled at the averaged speed of the surrounding BSMs."""
    vehicle.CurrentX += distance_traveled * math.cos(vehicle.CurrentHeading)
    vehicle.CurrentY += distance_traveled * math.sin(vehicle.CurrentHeading)
    vehicle.CurrentTime += time

def move_to_next_link(vehicle,link_endpoints):
    """Determine the next link in the route that the hypothetical vehicle needs to traverse, if there are no more links, mark the end time."""
    try:
        vehicle.CurrentLink = vehicle.Route[vehicle.Route.index(vehicle.CurrentLink) + 1]
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
    """Move hypothtical vehicle to next link in route, by removing distance and time traveled to the end of the current link and applying the remainder to the next link."""
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
    """Parse command line arguments, create BSM stream, routes dictionary, and link endpoints dictionary then start travel time evaluation."""
    parser = argparse.ArgumentParser(description='ME program for reading in BSMs and producing Travel Time values')
    parser.add_argument('bsm_filename') # CSV file of Basic Safety Messages
    parser.add_argument('fullroutes_filename') # CSV file of full routes
    parser.add_argument('links_filename') # CSV file of links in network
    parser.add_argument('link_endpoints_filename') # CSV file of link end points
    parser.add_argument('timing_output_filename')
    parser.add_argument('--out', help = 'Output csv file (include .csv)')  
    args = parser.parse_args()

    dir_path = os.path.dirname( os.path.realpath( __file__ ) )

    bsm_stream = BSMStream(filename = args.bsm_filename, links_filename = args.links_filename, add_link = 'route', time_bucket = 30)

    routes = read_full_routes(args.fullroutes_filename)

    link_endpoints = read_link_endpoints(args.link_endpoints_filename)

    if args.out:
        out_file = dir_path + '/' + args.out

    else:
        out_file = dir_path + '/traveltime_bsm.csv'

    run_travel_time(bsm_stream, routes, link_endpoints, out_file, args.timing_output_filename)

if __name__ == "__main__":
    main()