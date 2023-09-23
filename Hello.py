from unittest import result
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import heapq
import time
from collections import deque
from typing import List, Tuple, Dict, Callable
from copy import deepcopy
from math import sqrt

# define MOVES, COSTS, DIRS, GIFT constants
MOVES = [(0,-1), (1,0), (0,1), (-1,0)]
COSTS = { '🌾': 1, '🌲': 3, '⛰': 5, '🐊': 7}
DIRS = ['⏫','⏩','⏬','⏪']
GIFT = '🎁'

full_world = [
['🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌋', '🌋', '🌋', '🌋', '🌋', '🌋', '🌋', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '🌋', '⛰', '⛰', '⛰', '🌋', '🌋', '⛰', '⛰'],
['🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🐊', '🐊', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '⛰', '🌾'],
['🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '🌲', '🌲', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '⛰', '🌾'],
['🌾', '⛰', '⛰', '⛰', '🌋', '🌋', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '⛰', '🌾', '🌾'],
['🌾', '⛰', '⛰', '🌋', '🌋', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🌾', '⛰', '⛰', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾'],
['🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '🌾', '🐊', '🐊', '🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '⛰', '⛰', '🌋', '🌋', '🌋', '🌋', '🌾', '🌾', '🌾', '🐊', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '⛰', '⛰', '🌋', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '🌋', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🌾', '🌾', '⛰', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾'],
['🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '⛰', '🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '⛰', '🌋', '⛰', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '🌲', '🌲', '⛰', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌋', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '⛰', '⛰', '⛰', '⛰', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '⛰', '🌋', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌋', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '🌋', '⛰', '⛰', '🌾', '🐊', '🌾', '⛰', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '🌋', '🌾', '🌾', '🌋', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌋', '🌋', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌋', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌋', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '⛰', '⛰', '⛰', '⛰', '🌋', '🌋', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '🌋', '🌋', '🌋', '🌲', '🌲', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '🌋', '🌋', '🌋', '🌋', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '⛰', '⛰', '🌾', '🌾', '⛰', '⛰', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '⛰', '⛰', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊'],
['⛰', '🌋', '⛰', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌋', '🌋', '🌋', '⛰', '⛰', '🌋', '🌋', '🌾', '🌋', '🌋', '⛰', '⛰', '🐊', '🐊', '🐊', '🐊'],
['⛰', '🌋', '🌋', '🌋', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '🌋', '🌋', '🌋', '🌋', '⛰', '⛰', '⛰', '⛰', '🌋', '🌋', '🌋', '🐊', '🐊', '🐊', '🐊'],
['⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾', '🌾', '⛰', '⛰', '⛰', '🌾', '🌾', '🌾']
]

small_world = [
    ['🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
    ['🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
    ['🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲'],
    ['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
    ['🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾'],
    ['🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾'],
    ['🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾']
]

# heuristic takes a start location and a goal location, computes the estimated distance between the two. 
def heuristic(start: Tuple[int,int], goal: Tuple[int,int])->float:
    total = abs(goal[0] - start[0]) +  abs(goal[1] - start[1])
    return total

# place_on_frontier takes a frontier and a state, put the state on the frontier.
def place_on_frontier(frontier : List[int], state : Tuple) -> List:
    heapq.heappush(frontier, state)
    return frontier

# take_off_frontier takes a frontier, take the state on the top off the frontier.
def take_off_frontier(frontier : List) -> Tuple:
    current = heapq.heappop(frontier)
    return current

# state_is_in_explored takes a dict and a state, check if the location of the state is already in the dict.
def state_is_in_explored(state: Tuple, explored : Dict) -> bool:
    return state[1] in explored

# is_goal takes a current state and a goal location, check whether the goal location is the same as the current location. 
def is_goal(current : Tuple, goal: Tuple) -> bool:
    return current[1] == goal

# is_valid_state takes a current state and a world, check whether the state is valid to visit in the world. 
def is_valid_state(state: Tuple, world : List) -> bool:
    loc_x,loc_y = state
    return loc_x < len(world) and loc_x >= 0 and loc_y < len(world[0]) and loc_y >= 0 and world[loc_x][loc_y] != '🌋'

# construct_path_in_offsets takes a current state, uses the path metadata within the state to compute the new path in offsets.
def construct_path_in_offsets(current_state : Tuple) -> List:
    path = current_state[2]
    full_path = []
    for i in range(len(path)):
        if i >= 1:
            offset = (path[i][1] - path[i-1][1], path[i][0] - path[i-1][0])
            full_path.append(offset)
    return full_path

# successors takes a current state, a moves list, a world, a goal, an explored list and heuristic, computes the all valid successor states generated 
# by current state . 
def successors(current_state : Tuple, moves : List, world : List, costs : List, goal : Tuple, explored : Dict, heuristic : callable) -> List:
    successors = []
    current_cost, current, path = current_state
    new_cost = explored[current] + costs[world[current[0]][current[1]]]
    
    for move in moves:
        new_x = current[0] + move[0]
        new_y = current[1] + move[1]
        neighbor = (new_x, new_y)
        if is_valid_state(neighbor, world):
            new_cost = explored[current] + costs[world[current[0]][current[1]]]
            total_cost = new_cost + heuristic(neighbor, goal)
            neighbor_path = deepcopy(path)
            neighbor_path.append(neighbor)
            successors.append((total_cost, neighbor, neighbor_path))
    return successors

# a_star_search takes a world map, a start location, a goal location, a costs function, a moves list, an esimated function, by using A star algorithm, 
# optimize by the cost function total_cost = uniform_cost + estimated cost, to find out the best path with the lowest total cost from the start location
# to the goal location, prints out the desired optimal and complete path, returns the total cost.
def a_star_search( world: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int], moves: List[Tuple[int, int]], heuristic: Callable) -> List[Tuple[int, int]]:
    explored = {start:0} 
    frontier = [] 
    initial_state = (0, start, [start])
    frontier = place_on_frontier(frontier, initial_state)
    
    while len(frontier) > 0:
        current_state = take_off_frontier(frontier)
        if is_goal(current_state, goal):
            return construct_path_in_offsets(current_state)
        
        uniform_cost = explored[current_state[1]] + costs[world[current_state[1][0]][current_state[1][1]]]
        children = successors(current_state, moves, world, costs, goal, explored, heuristic)
        for child in children:
            if not state_is_in_explored(child, explored) or uniform_cost < explored[child[1]]:
                place_on_frontier(frontier, child)
                explored[child[1]] = uniform_cost 
    return [] 

# pretty_print_direction takes a world map, a single move, a location and prints the direction at the location.
def pretty_print_direction(world: List[List[str]], move: Tuple, loc: Tuple) -> None:
    cur_x, cur_y = loc
    if move == MOVES[0]:
        world[cur_x][cur_y] = DIRS[0]
        pass
    elif move == MOVES[1]:
        world[cur_x][cur_y] = DIRS[1]
        pass
    elif move == MOVES[2]:
        world[cur_x][cur_y] = DIRS[2]
        pass
    elif move == MOVES[3]:
        world[cur_x][cur_y] = DIRS[3]
        pass
    return None

# pretty_print_path takes a world map, a path list, a start location, a goal location and costs function, 
# prints out the path from the start location to the goal location, and returns the total cost of all the steps in the path. 
def pretty_print_path( world: List[List[str]], path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int]) -> int:
    curX = start[0]
    curY = start[1]
    total = 0
    new_world = deepcopy(world)
    for move in path:
        if world[curX][curY] in costs:
            total += costs[world[curX][curY]]
        pretty_print_direction(new_world, move, (curX,curY))
        curX += move[1]
        curY += move[0]
    new_world[goal[0]][goal[1]] = GIFT
    for line in new_world:
        print("".join(line))
    return total # replace with the real value!

# Streamlit app
st.title("A* search Visualization on Character Matrix with Animation")

# User interface
st.title("Costs Table")
st.write(COSTS)


current_world = small_world
world = st.sidebar.selectbox("select world map:", ["small world", "full world"])
st.sidebar.header("Choose a starting cell")

# The default is small world. Change to full world if user selects.
if world == "full world":
    current_world = full_world

# Allow users select start and end location.
start_row = st.sidebar.selectbox("start location row:", range(len(current_world)))
start_col = st.sidebar.selectbox("start location col:", range(len(current_world[0])))

st.sidebar.header("Choose an ending cell")
end_row = st.sidebar.selectbox("end location row:", range(len(current_world)))
end_col = st.sidebar.selectbox("end location col:", range(len(current_world[0])))

# Check users input valid or not, if valid we simulate A* search animation in the current map.
if current_world[start_row][start_col] == '🌋' or current_world[end_row][end_col] == '🌋':
    st.sidebar.warning("Please select a valid row and col. 🌋 is not possible to pass!")
# Start Running A* search if user press the button.
elif st.sidebar.button("Run A* search"):
    current_start = (start_row, start_col)
    current_goal = (end_row, end_col)
    current_path = a_star_search(current_world, current_start, current_goal, COSTS, MOVES, heuristic)
    current_path_cost = pretty_print_path(current_world, current_path, current_start, current_goal, COSTS)
    new_world = deepcopy(current_world)
    new_world[current_goal[0]][current_goal[1]] = GIFT
    curX, curY = current_start
    st.header("New World")
    result_container = st.empty()
    for move in current_path:
        pretty_print_direction(new_world, move, (curX,curY))
        curX += move[1]
        curY += move[0]
        result_container.dataframe(new_world)
        time.sleep(1)
    st.write("Total path costs: " + str(current_path_cost))

# Display the original character matrix.
st.header("Original World")
st.dataframe(current_world)


# Instructions.
st.sidebar.markdown("### Instructions:")
st.sidebar.markdown("1. Choose a starting cell and an end cell (row and column) from the sidebar.")
st.sidebar.markdown("2. Click the 'Run A* Search' button to visualize A* search traversal on the character matrix with animation.")
