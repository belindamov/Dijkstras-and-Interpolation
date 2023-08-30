# imports
import numpy as np
import matplotlib.pyplot as plt


def interpolate_linear(ti, yi, tj, default=None):
    """
    Performs linear interpolation of sampled data to a desired set of measurement points

    Parameters
    ---------
    ti : numpy.ndarray
        measurement points of the sampled data
    yi : numpy.ndarray
        measurement values of the sampled data
    tj : numpy.ndarray
        measurement points of the desired linearly interpolated data.
    default : None or other
        optional input; this value should be set for the measurement value when the corresponding measurement point is
        outside the sampled data

    Returns
    -------
    yj : numpy.ndarray
        measurement values for the linearly interpolated data.
    """

    # initialise yj array
    yj = np.zeros(len(tj))

    # nested for loop that sets values for elements of yj
    for i in range(0, len(tj)):
        # if the tj value is outside the sampled data, set the current index of yj to the default input
        if tj[i] < ti[0] or tj[i] > ti[-1]:
            yj[i] = default
        # if the tj value is inside the sampled data, check what two ti values it's between and interpolate for yj
        else:
            for j in range(0, len(ti) - 1):
                if ti[j] <= tj[i] <= ti[j + 1]:
                    yj[i] = yi[j] + (((yi[j + 1] - yi[j]) / (ti[j + 1] - ti[j])) * (tj[i] - ti[j]))

    return yj


def integrate_composite_trapezoid(tj, yj):
    """
    Integrates using the Newton-Cotes composite trapezoid rule

    Parameters
    ----------
    tj : numpy.ndarray
        measurement points of the integrand
    yj : numpy.ndarray
        measurement values of the integrand

    Returns
    -------
    integral : float
        numerical approximation of the integral

    Notes
    -----
    first and last points in tj array correspond to integral limits
    """

    # initialise n and integral
    n = len(tj)
    integral = 0

    for i in range(0, n - 1):
        # find h by taking the difference between the integral limits
        h = tj[i + 1] - tj[i]
        # find the area (integral) under the limits
        area = (h / 2) * (yj[i] + yj[i + 1])
        # add the area value onto integral
        integral += area

    return float(integral)


def spath_initialise(network, source_name):
    """
    Sets the initial distance and predecessor node for each node, and returns an unvisited set containing the names
    of all nodes

    Parameters
    ----------
    network : Network
        object that belongs to the Network class
    source_name : str
        name of source node

    Returns
    -------
    unvisited : set
        set containing the names of all nodes in the network

    Notes
    -----
    source_name and unvisited are not/do not contain Node objects
    """

    # if the source node doesn't exist in the network, return an empty set
    node_names = []
    for node in network.nodes:
        node_names.append(node.name)
    if source_name not in node_names:
        return {}

    unvisited = {source_name}

    # get the node with the same name as source_name and assign its distance to 0 and predecessor to None
    source_node = network.get_node(source_name)
    source_node.value = [0.0, None]

    node_list = network.nodes

    for i in range(0, len(node_list)):
        # if the current iteration in node_list is the same as the source_node, move onto the next iteration
        if node_list[i] == source_node:
            continue
        # add the name of the other nodes in the unvisited set
        unvisited.add(node_list[i].name)
        # set the distances of the other nodes to infinity and the predecessors to None
        node_list[i].value = [np.inf, None]

    return unvisited


def spath_iteration(network, unvisited):
    """
    Performs one iteration of the shortest path algorithm

    Parameters
    ----------
    network : Network
        object that belongs to the Network class
    unvisited : set
        set containing the names of all currently unsolved nodes in the network

    Returns
    -------
    solved_node : str or None
        name of the node that is solved on this iteration and is removed from the unvisited set; if no node can be
        solved then None
    """

    # if unvisited set is empty, return None
    if len(unvisited) == 0:
        return None

    # set solved_node to None and the shortest distance to infinity
    solved_node = None
    shortest_distance = np.inf

    for node_name in unvisited:
        # get the node with the same name as the string in the current iteration of the unvisited set
        node = network.get_node(node_name)
        # get the current shortest known distance to the node
        distance = node.value[0]
        # if the current shortest known distance to that node is shorter than the shortest distance, set that distance
        # to the shortest distance and set solved_node to the current node
        if distance < shortest_distance:
            shortest_distance = distance
            solved_node = node

    if solved_node is None:
        return None

    # remove the solved node from the unvisited set and get the arcs that start from the solved node
    unvisited.remove(solved_node.name)
    arcs_out = solved_node.arcs_out

    for i in range(0, len(arcs_out)):
        # get a node that one of the arcs go to (if there's more than one)
        node_to = arcs_out[i].to_node
        # set current_weight as the current shortest known distance to the node
        current_weight = node_to.value[0]
        # set arc_weight as the sum of the weight of the arc and the shortest known distance to the solved node
        arc_weight = arcs_out[i].weight + solved_node.value[0]
        # if arc_weight is less than current_weight, set the distance for node_to as arc_weight and its predecessor
        # to the solved node
        if arc_weight < current_weight:
            node_to.value = [arc_weight, solved_node.name]

    return solved_node.name


def spath_extract_path(network, destination_name):
    """
    Uses the chain of predecessors nodes to generate a list of node names for the shortest path from source to
    destination node.

    Parameters
    ----------
    network : Network
        object in the Network class
    destination_name : str
        name of the destination node

    Returns
    -------
    path : list
        list of node names for the shortest path, starting with the source node name and ending with the
        destination node name

    Notes
    -----
    destination_name is not a Node object
    """
    path = [destination_name]

    # get the node with the same name as the destination node and its predecessor
    destination_node = network.get_node(destination_name)
    node_before = network.get_node(destination_node.value[1])

    # while loop that continuously gets predecessors until the source node is reached
    while node_before is not None:
        # add the predecessor's node name onto the path list and get that predecessor's predecessor
        path.append(node_before.name)
        node_before = network.get_node(node_before.value[1])

    # return the path reversed
    return path[::-1]


def spath_algorithm(network, source_name, destination_name):
    """
    Performs Dijkstra’s shortest-path algorithm.

    Parameters
    ----------
    network : Network
        object in the Network class
    source_name : str
        name of the source node
    destination_name : str
        name of the destination node

    Returns
    -------
    distance : float or None
        the distance of the shortest path if a solution was found - if no solution found then None
    path : list or None
        list of node names for the shortest path starting with the source node name and ending with the destination node
        name; if no solution found then None
    """

    # get all the unvisited nodes
    unvisited = spath_initialise(network, source_name)

    i = 0

    # if the destination name is not in the unvisited set, return None for distance and None for path
    if destination_name not in unvisited:
        return None, None

    # if the source is the destination, return distance as 0 and path as None
    if source_name == destination_name:
        return 0, None

    # while loop that goes through, at most, all the nodes in the unvisited set
    while i != len(network.nodes):
        solved = spath_iteration(network, unvisited)
        # if the node that got removed from the unvisited set is the same as the destination node, stop the iterations
        if solved == destination_name:
            break
        # increase iteration by 1
        i += 1

    # get the path and initialise distance as 0
    path = spath_extract_path(network, destination_name)
    distance = 0

    for i in range(0, len(path) - 1):
        # get the node with the same name as the node in the current iteration of path
        node = network.get_node(path[i])
        # get the arcs that start from the node
        arcs_out = node.arcs_out

        for j in range(0, len(arcs_out)):
            # get a node that one of the arcs point to (if there is more than 1)
            node_to = arcs_out[j].to_node
            # if the node that the arc points to is the same as the next one in the path list, add that arc's weight
            # onto distance
            if node_to.name == path[i + 1]:
                distance += arcs_out[j].weight

    # if the distance has not changed, return None and None
    if distance == 0:
        return None, None

    return distance, path


class Node(object):
    """
    Object representing network node.

    Attributes:
    -----------
    name : str, int
        unique identifier for the node.
    value : float, int, bool, str, list, etc...
        information associated with the node.
    arcs_in : list
        Arc objects that end at this node.
    arcs_out : list
        Arc objects that begin at this node.
    """

    def __init__(self, name=None, value=None, arcs_in=None, arcs_out=None):

        self.name = name
        self.value = value
        if arcs_in is None:
            self.arcs_in = []
        if arcs_out is None:
            self.arcs_out = []

    def __repr__(self):
        return f"node:{self.name}"


class Arc(object):
    """
    Object representing network arc.

    Attributes:
    -----------
    weight : int, float
        information associated with the arc.
    to_node : Node
        Node object (defined above) at which arc ends.
    from_node : Node
        Node object at which arc begins.
    """

    def __init__(self, weight=None, from_node=None, to_node=None):
        self.weight = weight
        self.from_node = from_node
        self.to_node = to_node

    def __repr__(self):
        return f"arc:({self.from_node.name})--{self.weight}-->({self.to_node.name})"


class Network(object):
    """
    Basic Implementation of a network of nodes and arcs.

    Attributes
    ----------
    nodes : list
        A list of all Node (defined above) objects in the network.
    arcs : list
        A list of all Arc (defined above) objects in the network.
    """

    def __init__(self, nodes=None, arcs=None):
        if nodes is None:
            self.nodes = []
        if arcs is None:
            self.arcs = []

    def __repr__(self):
        node_names = '\n'.join(node.__repr__() for node in self.nodes)
        arc_info = '\n'.join(arc.__repr__() for arc in self.arcs)
        return f'{node_names}\n{arc_info}'

    def get_node(self, name):
        """
        Return network node with name.

        Parameters:
        -----------
        name : str
            Name of node to return.

        Returns:
        --------
        node : Node, or None
            Node object (as defined above) with corresponding name, or None if not found.
        """
        # loop through list of nodes until node found
        for node in self.nodes:
            if node.name == name:
                return node

        # if node not found, return None
        return None

    def add_node(self, name, value=None):
        """
        Adds a node to the Network.

        Parameters
        ----------
        name : str
            Name of the node to be added.
        value : float, int, str, etc...
            Optional value to set for node.
        """
        # create node and add it to the network
        new_node = Node(name, value)
        self.nodes.append(new_node)

    def add_arc(self, node_from, node_to, weight):
        """
        Adds an arc between two nodes with a desired weight to the Network.

        Parameters
        ----------
        node_from : Node
            Node from which the arc departs.
        node_to : Node
            Node to which the arc arrives.
        weight : float
            Desired arc weight.
        """
        # create the arc and add it to the network
        new_arc = Arc(weight, node_from, node_to)
        self.arcs.append(new_arc)

        # update the connected nodes to include arc information
        node_from.arcs_out.append(new_arc)
        node_to.arcs_in.append(new_arc)

    def read_network(self, filename):
        """
        Reads a file to construct a network of nodes and arcs.

        Parameters
        ----------
        filename : str
            The name of the file (inclusive of extension) from which to read the network data.
        """
        with open(filename, 'r') as file:

            # get first line in file
            line = file.readline()

            # check for end of file, terminate if found
            while line != '':
                items = line.strip().split(',')

                # create source node if it doesn't already exist
                if self.get_node(items[0]) is None:
                    self.add_node(items[0])

                # get starting node for this line
                source_node = self.get_node(items[0])

                for item in items:

                    # initial item ignored as it has no arc
                    if item == source_node.name:
                        continue

                    # separate out to destination node name and arc weight
                    data = item.split(';')
                    destination_node = data[0]
                    arc_weight = data[1]

                    # Create destination node if not already in network, then obtain the node itself
                    if self.get_node(destination_node) is None:
                        self.add_node(destination_node)
                    destination_node = self.get_node(destination_node)

                    # Add arc from source to destination node, with associated weight
                    self.add_arc(source_node, destination_node, float(arc_weight))

                # get next line in file
                line = file.readline()
