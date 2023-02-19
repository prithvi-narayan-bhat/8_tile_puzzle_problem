from queue import PriorityQueue
import argparse
from auxillary import *

'''
    Function to implement the Breadth First Search algorithm
'''
def bfs(start_state, end_state, log_file):

    start_node = Node(start_state)                      # The starting state of the board
    end_node = Node(end_state)                          # The goal state for the board
    visited = set()                                     # Set of all nodes that have been visited
    visited.add(tuple(map(tuple, start_node.state)))    # Keep appending to it
    fringe = [start_node]                               # Fringe that includes all nodes, visited and unvisited
    max_fringe_size = 1                                 # default value
    nodes_popped = 0                                    # default value

    # Execute until fringe is empty
    while len(fringe) > 0:
        max_fringe_size = max(max_fringe_size, len(fringe))
        node = fringe.pop(0)                            # Pop from fringe
        nodes_popped += 1                               # Increment the popped nodes count

        if node.state == end_node.state:                # Check if the goal has been reached
            depth, cost, moves = reconstruct_path(node) # Reconstruct path if a goal state has been reached
            if log_file is True:                        # Log only if requested from command line
                LOG(visited, fringe, None, "BFS")
            # return search statistics to calling function
            return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

        children = get_children(node)                                                   # Get all child nodes for give node
        nodes_generated, nodes_expanded = explore_children(children, visited, fringe)   # Explore all child nodes

    return None

'''
    Function to implement the Breadth First Search algorithm
'''
def dfs(start_state, end_state, log_file=None):

    start_node = Node(start_state)                      # The starting state of the board
    end_node = Node(end_state)                          # The goal state for the board
    visited = set()                                     # Set of all nodes that have been visited
    visited.add(tuple(map(tuple, start_node.state)))    # Keep appending to it
    fringe = [start_node]                               # Fringe that includes all nodes, visited and unvisited
    max_fringe_size = 1                                 # Initialise variables with default values
    nodes_popped = 0                                    # Initialise variables with default values
    nodes_expanded = 0                                  # Initialise variables with default values
    nodes_generated = 1                                 # Initialise variables with default values

    while len(fringe) > 0:                              # Execute until fringe is empty
        max_fringe_size = max(max_fringe_size, len(fringe))
        node = fringe.pop()
        nodes_popped += 1                               # Increment the popped nodes count

        if node.state == end_node.state:                # Check if current state matches the goal state of the board
            depth, cost, moves = reconstruct_path(node) # Reconstruct path if a goal state has been reached
            if log_file is True:                        # Log only if requested from command line
                LOG(visited, fringe, None, "DFS")

            # return search statistics to calling function
            return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

        children = get_children(node)                                                   # Get all child nodes for give node
        nodes_generated, nodes_expanded = explore_children(children, visited, fringe)   # Explore all child nodes

    return None

'''
    Function to perform Iteratively Deepening Search
'''
def ids(start_state, end_state, log_file=None):

    start_node = Node(start_state)
    end_node = Node(end_state)
    depth_limit = 0
    max_fringe_size = 1

    while True:
        stack = [[start_node]]
        visited = set()
        visited.add(tuple(map(tuple, start_node.state)))
        nodes_popped = 0
        nodes_expanded = 0
        nodes_generated = 1

        while stack:
            current_level = stack[-1]

            if not current_level:
                stack.pop()
                continue

            node = current_level.pop()
            nodes_popped += 1                               # Increment the popped nodes count

            if node.state == end_node.state:
                depth = 0
                cost = 0
                moves = []

                while node.parent is not None:
                    depth += 1
                    cost += node.cost
                    moves.append(node.move)
                    node = node.parent
                moves.reverse()

                if log_file is True:                        # Log only if requested from command line
                    LOG(visited, None, stack, "IDS")

                # return search statistics to calling function
                return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

            if node.depth < depth_limit:

                children = get_children(node)

                for child in children:
                    if tuple(map(tuple, child.state)) not in visited:
                        visited.add(tuple(map(tuple, child.state)))
                        stack.append([child])
                        nodes_generated += 1
                nodes_expanded += 1

        depth_limit += 1



def ucs(start_state, end_state, log_file=None):
    start_node = Node(start_state)
    end_node = Node(end_state)
    visited = set()
    visited.add(tuple(map(tuple, start_node.state)))
    fringe = PriorityQueue()
    fringe.put((0, start_node))
    max_fringe_size = 1
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 1
    while not fringe.empty():
        max_fringe_size = max(max_fringe_size, fringe.qsize())
        node = fringe.get()[1]
        nodes_popped += 1                               # Increment the popped nodes count
        if node.state == end_node.state:
            depth = 0
            cost = 0
            moves = []
            while node.parent is not None:
                depth += 1
                cost += node.cost
                moves.append(node.move)
                node = node.parent
            moves.reverse()

            if log_file is not None:

                with open(log_file, 'w') as f:
                    f.write('Closed nodes:\n')

                    for n in visited:
                        f.write(str(n)+'\n')

                    f.write('Fringes:\n')

                    for n in list(fringe.queue):
                        f.write(str(n[1].state)+'\n')

            if log_file is True:                        # Log only if requested from command line
                LOG(visited, fringe, None, "DFS")

            return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

        children = get_children(node)
        for child in children:
            if tuple(map(tuple, child.state)) not in visited:
                visited.add(tuple(map(tuple, child.state)))
                fringe.put((child.cost, child))
                nodes_generated += 1
        nodes_expanded += 1
    return None


# def manhattan_distance(state, goal_state):
#     distance = 0
#     for i in range(3):
#         for j in range(3):
#             if state[i][j] != 0:
#                 row, col = divmod(goal_state.index(state[i][j]), 3)
#                 distance += abs(row - i) + abs(col - j)
#     return distance

def manhattan_distance(state, goal_state):
    """
    Calculates the Manhattan distance heuristic for the 8-puzzle problem.

    Parameters:
        state (list of lists): The current state of the board.
        goal_state (list of lists): The goal state of the board.

    Returns:
        The Manhattan distance heuristic as an integer.
    """
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                continue
            row, col = divmod(goal_state.index(state[i][j]), 3)
            distance += abs(i - row) + abs(j - col)
    return distance


def gfs(start_state, end_state, heuristic, log_file=None):
    start_node = Node(start_state)
    end_node = Node(end_state)
    visited = set()
    visited.add(tuple(map(tuple, start_node.state)))
    fringe = PriorityQueue()
    fringe.put((heuristic(start_node.state, end_state), start_node))
    max_fringe_size = 1
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 1
    while not fringe.empty():
        max_fringe_size = max(max_fringe_size, fringe.qsize())
        node = fringe.get()[1]
        nodes_popped += 1                               # Increment the popped nodes count
        if node.state == end_node.state:
            depth = 0
            cost = 0
            moves = []
            while node.parent is not None:
                depth += 1
                cost += node.cost
                moves.append(node.move)
                node = node.parent
            moves.reverse()

            if log_file is not None:
                with open(log_file, 'w') as f:
                    f.write('Closed nodes:\n')
                    for n in visited:
                        f.write(str(n)+'\n')

                    f.write('Fringes:\n')
                    for n in list(fringe.queue):
                        f.write(str(n[1].state)+'\n')

            # return search statistics to calling function
            return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

        children = get_children(node)
        for child in children:
            if tuple(map(tuple, child.state)) not in visited:
                visited.add(tuple(map(tuple, child.state)))
                fringe.put((heuristic(child.state, end_state), child))
                nodes_generated += 1
        nodes_expanded += 1
    return None


def astar(start_state, end_state, heuristic_function, log_file=None):
    start_node = Node(start_state)
    end_node = Node(end_state)
    visited = set()
    visited.add(tuple(map(tuple, start_node.state)))
    fringe = PriorityQueue()
    fringe.put((0, start_node))
    max_fringe_size = 1
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 1
    while not fringe.empty():
        max_fringe_size = max(max_fringe_size, fringe.qsize())
        node = fringe.get()[1]
        nodes_popped += 1                               # Increment the popped nodes count
        if node.state == end_node.state:
            depth = 0
            cost = 0
            moves = []
            while node.parent is not None:
                depth += 1
                cost += node.cost
                moves.append(node.move)
                node = node.parent
            moves.reverse()

            if log_file is not None:

                with open(log_file, 'w') as f:
                    f.write('Closed nodes:\n')

                    for n in visited:
                        f.write(str(n)+'\n')

                    f.write('Fringes:\n')

                    for n in list(fringe.queue):
                        f.write(str(n[1].state)+'\n')

            # return search statistics to calling function
            return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

        children = get_children(node)
        for child in children:
            if tuple(map(tuple, child.state)) not in visited:
                visited.add(tuple(map(tuple, child.state)))
                cost = child.cost + heuristic_function(child.state, end_state)
                fringe.put((cost, child))
                nodes_generated += 1
        nodes_expanded += 1
    return None


def dls(start_state, end_state, max_depth, log_file=None):
    start_node = Node(start_state)
    end_node = Node(end_state)
    stack = [[start_node]]
    visited = set()
    visited.add(tuple(map(tuple, start_node.state)))
    max_fringe_size = 1
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 1
    while stack:
        print(f'Nodes Popped: {nodes_popped}')
        print(f'Nodes Expanded: {nodes_expanded}')
        print(f'Nodes Generated: {nodes_generated}')
        print(f'Max Fringe Size: {max_fringe_size}')
        max_fringe_size = max(max_fringe_size, len(stack))
        path = stack[-1]
        node = path[-1]
        nodes_popped += 1                               # Increment the popped nodes count
        if node.state == end_node.state:
            depth = len(path) - 1
            cost = 0
            moves = []
            for i in range(1, len(path)):
                cost += path[i].cost
                moves.append(path[i].move)

            if log_file is not None:
                with open(log_file, 'w') as f:
                    f.write('Closed nodes:\n')
                    for n in visited:
                        f.write(str(n)+'\n')

            # return search statistics to calling function
            return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

        if len(path) <= max_depth:
            children = get_children(node)
            nodes_expanded += 1
            for child in children:
                if tuple(map(tuple, child.state)) not in visited:
                    visited.add(tuple(map(tuple, child.state)))
                    stack.append(path + [child])
                    nodes_generated += 1
        else:
            stack.pop()

    return None


# Practically the main function that invokes the right algorithm from from cli parameters
parser = argparse.ArgumentParser()
parser.add_argument('start', help='start state file')
parser.add_argument('end', help='end state file')
parser.add_argument('algorithm', help='Algorithm to use | '
                    'BFS = Breadth First Search | '
                    'DFS = Depth First Search | '
                    'DLS = Depth Limited Search | '
                    'IDS = Iteratively Deepening Search | '
                    'UCS = Uniform Cost Search | '
                    'GFS = Greedy first Search | '
                    'ASS = A* Search')

parser.add_argument('log', help='Log to file | l=log nl=no log')
args = parser.parse_args()

algorithm = args.algorithm

if (args.log) == 'l':
    log = True
else:
    log = False

start_state = load_state(args.start)
end_state = load_state(args.end)

if (algorithm == 'BFS'):
    result = bfs(start_state, end_state, log)
elif (algorithm == 'DFS'):
    result = dfs(start_state, end_state, log)
elif (algorithm == 'DLS'):
    result = dls(start_state, end_state, log)
elif (algorithm == 'IDS'):
    result = ids(start_state, end_state, log)
elif (algorithm == 'UCS'):
    heuristic_manhattan = manhattan_distance(start_state, end_state)
    result = ucs(start_state, end_state, log)
elif (algorithm == 'GFS'):
    heuristic_manhattan = manhattan_distance(start_state, end_state)
    result = gfs(start_state, end_state, log)
elif (algorithm == 'ASS'):
    heuristic_manhattan = manhattan_distance(start_state, end_state)
    result = astar(start_state, end_state, heuristic_manhattan, log)

if result is None:
    print('No solution found!')
else:
        depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size = result
        print(f'Nodes Popped: {nodes_popped}')
        print(f'Nodes Expanded: {nodes_expanded}')
        print(f'Nodes Generated: {nodes_generated}')
        print(f'Max Fringe Size: {max_fringe_size}')
        print(f'Solution Found at depth {depth} with cost of {cost}.')
        print('Steps:')
        for move in moves:
            print(f'\tMove {move}')
