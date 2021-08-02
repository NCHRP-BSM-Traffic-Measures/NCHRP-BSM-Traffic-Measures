import numpy as np

"""Read Basic Safety Messages from a TCA output file and turn it into a real time stream of BSMs by time step."""

class BSM:
    """Helper class defines the index for BSM data that is returned by BSMStream"""
    TimeIndex = 0
    XIndex = 1
    YIndex = 2
    SpeedIndex = 3 
    LinkIndex = 8
    AccelerationIndex = 4
    HardBrakingIndex = 5
    BrakePressureIndex = 6
    TimeBucketIndex = 7
    RouteIndex = 8
    SuperlinkIndex = 8

class BSMStream:
    """Main class reads multiple versions of TCA BSM output files, pulls relevant data for Measures Estimation, groups by time period and calls Link Indentifier before yielding BSM list to Measures Code."""
    def __init__(self, filename, links_filename, add_link, time_bucket):
        """Initialize a BSMStream reading the given filename and assigning BSMs to links from the given links_filename.
            
            Arguments:
                - filename: the name of the BSM file to read
                - links_filename: the name of the links file to read
                - add_link: string of what geographic assignment column from the links file to use "link", "route" or "superlink"
                - time_bucket: the time in seconds that BSMs should be grouped by. Common usage: 30 seconds for travel time and space mean speed, 5 seconds for incident detection
        """
        self.filename = filename
        self.TimeIndex = self.XIndex = self.YIndex = self.SpeedIndex = self.VISSIMIndex = = self.BrakePressureIndex = None
        self.link_identifier = LinkIdentifier(links_filename, add_link)
        self.time_bucket = time_bucket

    def read(self):
        """Read through the given TCA BSM output file and yield a list of BSMs that were generated at the same timestep."""
        with open(self.filename) as in_f:
            old_tp = None
            tp_list = []
            is_header = True
            for line in in_f:
                data = line.strip().split(',')
                if is_header:
                    try:
                        self.TimeIndex = data.index('localtime')
                        self.XIndex = data.index('x')
                        self.YIndex = data.index('y')
                        self.SpeedIndex = data.index('spd')
                        self.AccelerationIndex = data.index('instant_accel')
                        self.HardBrakingIndex = data.index('hardBraking')
                        self.BrakePressureIndex = data.index('brakePressure')
                    except ValueError:
                        self.TimeIndex = data.index('transtime')
                        self.XIndex = data.index('X')
                        self.YIndex = data.index('Y')
                        self.SpeedIndex = data.index('Speed')
                        self.AccelerationIndex = data.index('Instant_Acceleration')
                        self.HardBrakingIndex = data.index('hardBraking')
                        self.BrakePressureIndex = data.index('brakePressure')
                    is_header = False
                    continue
                tp = float(data[self.TimeIndex])
                if tp != old_tp:
                    if old_tp != None:
                        tp_list = self.link_identifier.findLink(np.array(tp_list))
                        yield old_tp, tp_list
                    old_tp = tp
                    tp_list = []
                x = float(data[self.XIndex])
                y = float(data[self.YIndex])
                #Speed is multiplied by 1.46667 to convert TCA MPH speed to ft/second
                tp_list.append([tp,x,y,float(data[self.SpeedIndex]) * 1.46667,float(data[self.AccelerationIndex]),float(data[self.HardBrakingIndex]),float(data[self.BrakePressureIndex]),int(tp/self.time_bucket)])
            #yield last tp
            tp_list = self.link_identifier.findLink(np.array(tp_list))
            yield tp, tp_list

class LinkIdentifier:
    """Helper class that receives a numpy array of BSMs and a numpy array of geographic links and assigns the BSMs to a link based on their X,Y location."""
    def __init__(self, filename, add_link):
        """Initialize LinkIndetifier with the links defined in the given filename. Pre-define constants AB_dot_AB and AD_dot_AD that are used for BSM localization.
            
            Arguments:
                - filename: the name of the links input file
                - add_link: string of what geographic assignment column from the links file to use "link", "route" or "superlink"
        """
        self.links = np.genfromtxt(filename, skip_header=1, delimiter=',',dtype=np.float64)
        self.AB_dot_AB = self.links[:,3:4]**2 + self.links[:,4:5]**2
        self.AD_dot_AD = self.links[:,5:6]**2 + self.links[:,6:7]**2 
        self.add_link = add_link

    def findLink(self, tp_list):
        """Adds a link, route or superlink assignment to the end of the passed BSM array tp_list using the formula detailed in https://math.stackexchange.com/a/190373."""
        point_vectors_x = tp_list[:,1] - self.links[:,1:2]
        point_vectors_y = tp_list[:,2] - self.links[:,2:3]
        AM_dot_AB = point_vectors_x * self.links[:,3:4] + point_vectors_y * self.links[:,4:5]
        AM_dot_AD = point_vectors_x * self.links[:,5:6] + point_vectors_y * self.links[:,6:7]
        truth_eval = (AM_dot_AB > 0) & (AM_dot_AB < self.AB_dot_AB) & (AM_dot_AD > 0) & (AM_dot_AD < self.AD_dot_AD)
        link_assignments = self.links[np.argmax(np.transpose(truth_eval),axis=1)]
        if add_link == 'link':
            tp_list = np.hstack((tp_list,link_assignments[:,0:1]))
        elif add_route == 'route':
            tp_list = np.hstack((tp_list,link_assignments[:,9:10]))
        elif add_superlink == 'superlink':
            tp_list = np.hstack((tp_list,link_assignments[:,8:9]))
        return tp_list.tolist()