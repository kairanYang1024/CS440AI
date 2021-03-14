# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)
import itertools
from collections import deque
import heapq


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # assumption: BFS & A* are zero-heuristic thus consistent, first processed is thus optimal
    frt = deque()  # type of indices tuples for exploring states
    explored_set = set([])  # empty set recording indices being explored (pushed to frt, to avoid duplicates)
    prev = {}  # the dict recording the cell and its prev cell (first one adjacent to it that is processed by the frt)
    start = maze.start  # a tuple consisting [i,j] coord for start of the maze
    goal = maze.waypoints[0]  # single waypoint case for BFS, not true for part3-4, is a tuple ~ for goal of the maze

    frt.append(start)
    explored_set.add(start)
    prev[start] = None  # start cell does not have any prevs
    while frt:  # overload data-structure so that frt == True when len(frt) > 0, similar to if(pointer) in C++
        curr = frt.popleft()  # append() and popleft() makes FIFO, appendleft() and pop() makes LIFO
        if curr == goal:  # if goal is being explored we just exit the searching algo
            break
        neighbors = maze.neighbors(*curr)  # unpack tuple curr = (i, j)
        for neighbor in neighbors:
            if neighbor not in explored_set:  # only record new cells
                frt.append(neighbor)
                explored_set.add(neighbor)
                prev[neighbor] = curr

    trace = goal
    path = []
    while trace is not None:  # trace the path using the prev dict, reverse so it begins at the start cell
        path.append(trace)
        trace = prev[trace]
    path.reverse()
    return path


def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    frt = []
    heapq.heapify(frt)  # elems take in format of (f(n), n) where n is a tuple representing maze cells
    # using dictionary for explored set is redundant in consistent but necessary for admissible heurs
    # use nodes to prev_label for entries of dict, states of diff costs are considered different states
    prev = {}  # this time mapping n to n_prev, updated when a 'better' f(n) appeared
    cost_dict = {}  # mapping n to g(n), f(n) = g(n)+h(n), checked when (f(n),n) needed to be updated to the best path
    start = maze.start
    goal = maze.waypoints[0]

    cost_dict[start] = 0
    heapq.heappush(frt, (manhattan_heur(maze, start, goal), start))  # state is a tuple
    prev[start] = None
    while frt:
        curr = heapq.heappop(frt)
        curr_cell = curr[1]
        if curr_cell == goal:
            break
        neighbors = maze.neighbors(*curr_cell)
        for neighbor in neighbors:
            # every node when being popped from the frt should have defined cost
            tentative_cost = cost_dict[curr_cell] + 1
            # handling cases of neighbor being a new state (not stored in prev) or need update to a new path (prev)
            if neighbor not in prev.keys() or tentative_cost < cost_dict[neighbor]:
                cost_dict[neighbor] = tentative_cost  # update for a better path cost OR defined as being visited
                neighbor_total_cost = cost_dict[neighbor] + manhattan_heur(maze, neighbor, goal)
                heapq.heappush(frt, (neighbor_total_cost, neighbor))  # each state is sorted in its f=g+h value
                prev[neighbor] = curr_cell

    trace = goal
    path = []
    while trace is not None:
        path.append(trace)
        trace = prev[trace]
    path.reverse()
    return path


def manhattan_heur(maze, curr, goal):  # for part 2 astar_single only, part 3 and 4 requires other heuristics
    # curr[0] = i is the y_coord and curr[1] = j is the x_coord,
    # reminder for converting between array and cartesian systems
    return abs(goal[0] - curr[0]) + abs(goal[1] - curr[1])


"""
Part 3 and 4 Only
Node class for multiple waypoints search algo, since simple data structure cannot sustain the complexity anymore
Consisting the cell coordinate, cost, cost+heuristic (f_value) and the waypoints so far reached, for pqueue comparison
"""


class Node:
    def __init__(self, i, j, input_cost=2 ** 30 - 1, waypoints_reached=[], input_heur=2 ** 30 - 1):
        self.y = i
        self.x = j
        self.cost = input_cost  # the g_score for this Node
        self.wpt_reached = waypoints_reached  # a set recording all the unique waypoints reached by this state
        self.total_cost = input_cost + input_heur  # the computed heuristic for cell (i,j) given waypoints reached

    def __repr__(self):
        return f"({self.y}, {self.x}), g={self.cost}, f={self.total_cost}, waypoints reached:{self.wpt_reached}"

    def __lt__(self, other):
        if self.total_cost == other.get_total_cost():  # if 2 nodes having same f-val, then the more goals reached win
            return len(self.wpt_reached) > len(other.get_waypoints_reached())
        return self.total_cost < other.get_total_cost()

    def __eq__(self, other):  # 2 states are equal if same cell and same waypoints visited
        return self.get_cell() == other.get_cell() and self.wpt_reached == other.get_waypoints_reached()

    def __hash__(self):  # make node object immutable and implementable using dict, also used in dict keys 'in' operator
        return hash((self.y, self.x, tuple(self.wpt_reached)))

    def get_cell(self):
        return self.y, self.x

    def get_cost(self):
        return self.cost

    def get_waypoints_reached(self):
        return self.wpt_reached

    def get_total_cost(self):
        return self.total_cost


def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    frt = []
    heapq.heapify(frt)
    prev_dict = {}  # Node to Node for distinguishing same cell different heur/goals
    cost_dict = {}  # closed set for Node mapping to optimal f_val, 2 Node are same if only same cell and goals visited
    # since heuristic is no longer unique for each cell, compare the f_value instead

    start = maze.start
    total_waypoints = maze.waypoints
    start_heur = permute_heur(maze, start, set([]))
    start_node = Node(start[0], start[1], 0, set([]), start_heur)
    heapq.heappush(frt, start_node)
    prev_dict[start_node] = None
    cost_dict[start_node] = start_heur
    final_goal = None  # place_holder

    while frt:
        curr_node = heapq.heappop(frt)
        curr_cell = curr_node.get_cell()
        curr_waypoints_reached = curr_node.get_waypoints_reached()
        if curr_waypoints_reached == set(total_waypoints):
            final_goal = curr_node
            break

        neighbors = maze.neighbors(*curr_cell)
        for neighbor_cell in neighbors:
            # every node when being popped from the frt should have defined cost
            tentative_cost = curr_node.get_cost() + 1
            neighbor_waypoint_reached = curr_waypoints_reached.copy()  # each node inherits the waypoint reached from its prev
            if neighbor_cell in total_waypoints:  # if we are visiting a waypoint
                neighbor_waypoint_reached.add(neighbor_cell)  # checks duplicated waypoint also since set is used
            tentative_heur = permute_heur(maze, neighbor_cell, neighbor_waypoint_reached)
            tentative_f = tentative_cost + tentative_heur
            neighbor_node = Node(neighbor_cell[0], neighbor_cell[1], tentative_cost,
                                 neighbor_waypoint_reached,
                                 tentative_heur)  # construct the node for open set sorting and comparison

            # handling cases of neighbor being a new state (not stored in prev) or need update to a new path (prev)
            if neighbor_node not in prev_dict.keys() and \
                    neighbor_node.get_total_cost() < cost_dict.setdefault(neighbor_node, 2 ** 30 - 1 + tentative_cost):
                # since each state is now unique in goal_visited, add parameter of a better cost than recorded to
                # filter out garbage states of large f in frt
                cost_dict[neighbor_node] = tentative_f
                heapq.heappush(frt, neighbor_node)
                prev_dict[neighbor_node] = curr_node

    trace = final_goal
    path = []
    while trace is not None:
        path.append(trace.get_cell())
        trace = prev_dict[trace]
    path.reverse()
    return path


def permute_heur(maze, curr_cell, reached_waypoints):
    """
    Permuting the waypoint orders so we can enumerate the smallest heuristic sum as the heuristic for the state

    @param maze: The maze to execute the search on

    @param curr_cell: the tuple of the cell that needed to compute heuristic

    @param reached_waypoints: the set of all waypoints reached by the current node (state)

    @return the h_value for the state being the minimum of all enumerations
    """
    total_waypoints = set(maze.waypoints)
    permute_set = total_waypoints.difference(reached_waypoints)  # reached_waypoints should be subset of all waypts

    # this algorithm is O(n!) to size of (remaining) waypoints, only applicable in part 3
    min_heur = 2 ** 30 - 1  # default large number
    # order matters in comparing different visiting sequences
    for waypoints_perm in itertools.permutations(permute_set, len(permute_set)):
        perm_heur = 0  # total heuristic for part-3 is sum of heuristics of segments between waypoints
        start = curr_cell
        # waypoint_perm is a tuple (permutation sequence) of cells
        for waypoint in waypoints_perm:
            perm_heur += manhattan_heur(maze, start, waypoint)
            start = waypoint  # move to the next segment
        min_heur = min(min_heur, perm_heur)

    return min_heur


def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    frt = []
    heapq.heapify(frt)
    prev_dict = {}  # Node to Node for distinguishing same cell different heur/goals
    cost_dict = {}  # closed set for Node mapping to optimal f_val, 2 Node are same if only same cell and goals visited

    # use DP technique to store precomputed MST if same subset of unvisited goals present many times
    # mapping subset of maze waypoints (unvisited) to its minimum cost of edges (measured in Man-dist)
    MST_dict = {}

    start = maze.start
    total_waypoints = maze.waypoints
    start_heur = mst_heur(maze, start, set([]), MST_dict)
    start_node = Node(start[0], start[1], 0, set([]), start_heur)
    heapq.heappush(frt, start_node)
    prev_dict[start_node] = None
    cost_dict[start_node] = start_heur
    final_goal = None  # place_holder

    while frt:
        curr_node = heapq.heappop(frt)
        # print("large counter", counter)
        curr_cell = curr_node.get_cell()
        curr_waypoints_reached = curr_node.get_waypoints_reached()
        if curr_waypoints_reached == set(total_waypoints):
            final_goal = curr_node
            break

        neighbors = maze.neighbors(*curr_cell)
        for neighbor_cell in neighbors:
            # every node when being popped from the frt should have defined cost
            tentative_cost = curr_node.get_cost() + 1
            neighbor_waypoint_reached = curr_waypoints_reached.copy()  # each node inherits the waypoint reached from its prev
            if neighbor_cell in total_waypoints:  # if we are visiting a waypoint
                neighbor_waypoint_reached.add(neighbor_cell)  # checks duplicated waypoint also since set is used
            tentative_heur = mst_heur(maze, neighbor_cell, neighbor_waypoint_reached, MST_dict)
            tentative_f = tentative_cost + tentative_heur
            neighbor_node = Node(neighbor_cell[0], neighbor_cell[1], tentative_cost,
                                 neighbor_waypoint_reached,
                                 tentative_heur)  # construct the node for open set sorting and comparison

            # handling cases of neighbor being a new state (not stored in prev) or need update to a new path (prev)
            if neighbor_node not in prev_dict.keys() and \
                    neighbor_node.get_total_cost() < cost_dict.setdefault(neighbor_node, 2 ** 30 - 1 + tentative_cost):
                # since each state is now unique in goal_visited, add parameter of a better cost than recorded to
                # filter out garbage states of large f in frt
                cost_dict[neighbor_node] = tentative_f
                heapq.heappush(frt, neighbor_node)
                prev_dict[neighbor_node] = curr_node

    trace = final_goal
    path = []
    while trace is not None:
        path.append(trace.get_cell())
        trace = prev_dict[trace]
    path.reverse()
    return path


def mst_heur(maze, curr_cell, reached_waypoints, MST_dict):
    """
    Compute the heuristic based on MST minimum cost for approximating multiple waypoints visiting cost.

    @param maze: The maze to execute the search on

    @param curr_cell: the tuple of the cell that needed to compute heuristic

    @param reached_waypoints: the set of all waypoints reached by the current node (state)

    @param MST_dict: the dict storing pattern of reached_waypoints's MST cost, if found no need to compute MST again
    only used for in-place modification (pass by ref)

    @return the h_value for the state being the min edge weight (sum in manhattan) + dist(curr, closest waypoint),
    where we say "closest" to keep this heuristic admissible h(n) < cost(n)
    """
    unreached_waypoints = set(maze.waypoints).difference(reached_waypoints)
    # if we visited every waypoint in the maze then h=0 since we got to the goal
    closest_distance = 0 if len(unreached_waypoints) == 0 else manhattan_heur(maze, curr_cell, list(unreached_waypoints)[0])
    for v in unreached_waypoints:
        closest_distance = min(manhattan_heur(maze, curr_cell, v), closest_distance)

    # setdefault assigns key with default value if key not found in dictionary
    # but it still computes it no matter found or not if default is a function instead of const
    # dont use it when the default value need to compute by function (differs each time)
    if tuple(reached_waypoints) in MST_dict.keys():
        mst_val = MST_dict[tuple(reached_waypoints)]
    else:
        mst_val = mst_cost(maze, unreached_waypoints)
        MST_dict[tuple(reached_waypoints)] = mst_val
    return closest_distance + mst_val


def mst_cost(maze, unreached_waypoints):
    """
    Build a MST for connecting all unvisited waypoints and compute the total edge cost using Prim's algorithm

    @param maze: The maze to execute the search on

    @param unreached_waypoints: the set of all waypoints not yet reached by the current node (state), each be a vertex
    of the set of building the MST

    @return the minimum cost of connecting every vertex in the set of unreached_waypoint
    """
    # trivial case
    if len(unreached_waypoints) <= 1:
        return 0

    # building the edges between each waypts: O(n^2) = O(E)
    vertex_list = list(unreached_waypoints)
    edge_dict = {}  # mapping edge as tuple of 2 vertices to its cost
    for u in vertex_list:
        for v in vertex_list:
            if u != v and (u, v) not in edge_dict.keys():
                # undirected
                edge_dict[(u, v)] = manhattan_heur(maze, u, v)
                edge_dict[(v, u)] = manhattan_heur(maze, v, u)

    # starting at a random waypoint and subsequently expand the edges until all waypoints are connected, O(E + logV)
    # like a mini UCS search
    start = vertex_list[0]
    edge_frt = []
    heapq.heapify(edge_frt)
    heapq.heappush(edge_frt, (0, start))
    cost_dict = {start: 0}  # mapping vertices to the current cost to connect to it, no need to build the tree itself
    nodes_connected = {start}
    iter = 0
    while edge_frt:
        iter += 1
        curr = heapq.heappop(edge_frt)
        curr_cost, curr_v = curr
        nodes_connected.add(curr_v)
        cost_dict[curr_v] = min(cost_dict.setdefault(curr_v, 2**30-1), curr_cost)
        for adj_v in vertex_list:
            if adj_v in nodes_connected:
                continue
            heapq.heappush(edge_frt, (edge_dict[(adj_v, curr_v)], adj_v))
    return sum(cost_dict.values())


def fast(maze):  # opt-out
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
