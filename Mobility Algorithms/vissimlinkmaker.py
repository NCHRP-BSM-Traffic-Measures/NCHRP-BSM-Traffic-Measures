"""Create a Measures Estimation Links File using modified data from a VISSIM .inp file, currently set for the I-405 data
"""
def findNextKeyword(data, i):
	"""Find the next OVER or TO keyword in VISSIM file to know where to read next X,Y coordinate
	"""
	while data[i] != "OVER" and data[i] != "TO":
		i += 1
	return i

def readData(data,i,link_count,start_x,start_y,link,road_width,superlink,route):
	"""Read through a VISSIM link and find the keywords to define smaller Measures Estimation Links and ouput them to file
	"""
	keyword = data[i]
	i += 1
	end_x = float(data[i]) * 3.28084
	i += 1 
	end_y = float(data[i]) * 3.28084
	xdif = end_x - start_x
	ydif = end_y - start_y
	magnitude = (xdif**2 + ydif**2)**0.5
	v_x	= -1 * ydif * (road_width/(2 * magnitude))
	v_y = xdif * (road_width/(2 * magnitude))
	x1 = start_x + v_x
	y1 = start_y + v_y
	x2 = start_x - v_x
	y2 = start_y - v_y
	x3 = end_x + v_x
	y3 = end_y + v_y
	v1x = x3 - x1
	v1y = y3 - y1
	v2x = x2 - x1
	v2y = y2 - y1
	out_f.write("{},{},{},{},{},{},{},{},{},{}\n".format(link_count,x1,y1,v1x,v1y,v2x,v2y,link,superlink,route))
	link_count += 1
	if keyword == "OVER":
		i = findNextKeyword(data,i)
		start_x = end_x
		start_y = end_y
		link_count = readData(data,i,link_count,start_x,start_y,link,road_width,superlink,route)
	elif keyword == "TO":
		link_count += 1
	return link_count

lookup_table = {}
with open("i405linkwidths.csv") as in_f:
	is_header = True
	for line in in_f:
		if is_header:
			is_header = False
			continue
		data = line.split(",")
		lookup_table[data[0]] = float(data[1])

with open('i405_superlinks.csv') as in_f:
	superlink_ref = {}
	for line in in_f:
		row = line.strip().split(',')
		superlink = row[0]
		links_list = [i for i in row[1:]] 
		for link in links_list:
			superlink_ref[link] = superlink

with open('i405_fullroutes.csv') as in_f:
	route_ref = {}
	for line in in_f:
		row = line.strip().split(',')
		route = row[0]
		links_list = [i for i in row[1:]] 
		for link in links_list:
			route_ref[link] = route

with open("i405links.csv","w") as out_f:
	out_f.write("Link,X1,Y1,Vector1_x,Vector1_y,Vector2_x,Vector2_y,VissimLink,Superlink,Route\n")
	with open("i405links.txt") as in_f:
		link_count = 1
		for line in in_f:
			data = line.split(" ")
			road_width = 0
			link = data[1]
			i = 3
			while data[i] != "FROM":
				road_width += float(data[i]) * 3.28084
				i += 1
			i += 1
			start_x = float(data[i]) * 3.28084
			i += 1
			start_y = float(data[i]) * 3.28084
			i = findNextKeyword(data,i)
			link_count = readData(data,i,link_count,start_x,start_y,link,road_width,superlink_ref.get(link,'NA'),route_ref.get(link,'NA'))
	with open("i405connectors.txt") as in_f:
		for line in in_f:
			data = line.strip().split(" ")
			link_lookup = data[-1]
			road_width = lookup_table[link_lookup]  * 3.28084
			link = data[1]
			start_x = float(data[3]) * 3.28084
			start_y = float(data[4]) * 3.28084
			i = findNextKeyword(data,4)
			link_count = readData(data,i,link_count,start_x,start_y,link,road_width,superlink_ref.get(link,'NA'),route_ref.get(link,'NA'))