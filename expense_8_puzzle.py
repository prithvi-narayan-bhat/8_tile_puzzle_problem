from queue import PriorityQueue
from argparse import ArgumentParser as parser
from auxillary import *

def breadthFirstSearch(start_state, end_state, log_file=False):
    """
    Performs Breadth First Search to find a path from start_state to end_state.

    Args:
        start_state (list): The initial state.
        end_state (list): The goal state.
        log_file (bool): If True, log the search process to a file.

    Returns:
        A tuple of search statistics and the path from start_state to end_state, or None if no path is found.
    """

    start_node = Node(start_state)                                  # The starting state of the board
    end_node = Node(end_state)                                      # The goal state for the board
    visited = set()                                                 # Set of all nodes that have been visited
    visited.add(tuple(map(tuple, start_node.state)))                # Keep appending to it
    fringe = [start_node]                                           # Fringe that includes all nodes, visited and unvisited
    max_fringe_size = 1                                             # default value
    nodes_popped = 0                                                # default value
    nodes_expanded = 0                                              # Initialise variables with default values
    nodes_generated = 1                                             # Initialise variables with default values

    while len(fringe) > 0:                                          # Execute until fringe is empty
        max_fringe_size = max(max_fringe_size, len(fringe))
        node = fringe.pop(0)                                        # Pop node from fringe to explore
        nodes_popped += 1                                           # Increment the popped nodes count

        if node.state == end_node.state:                            # Check if the goal has been reached
            depth, cost, moves = reconstruct_path(node, "BFS")      # Reconstruct path if a goal state has been reached
            if log_file is True:                                    # Log only if requested from command line
                logProgress(visited, fringe, "BFS")

            # return search statistics to calling function
            return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

        nodes_generated, nodes_expanded = exploreChildren(node, visited, fringe, None, "BFS") # Explore all child nodes

    return None


def depthFirstSearch(start_state, end_state, log_file=False):
    """
    Performs Depth First Search to find a path from start_state to end_state.

    Args:
        start_state (list): The initial state.
        end_state (list): The goal state.
        log_file (bool): If True, log the search process to a file.

    Returns:
        A tuple of search statistics and the path from start_state to end_state, or None if no path is found.
    """

    start_node = Node(start_state)                                  # The starting state of the board
    end_node = Node(end_state)                                      # The goal state for the board
    visited = set()                                                 # Set of all nodes that have been visited
    visited.add(tuple(map(tuple, start_node.state)))                # Keep appending to it
    fringe = [start_node]                                           # Fringe that includes all nodes, visited and unvisited
    max_fringe_size = 1                                             # Initialise variables with default values
    nodes_popped = 0                                                # Initialise variables with default values
    nodes_expanded = 0                                              # Initialise variables with default values
    nodes_generated = 1                                             # Initialise variables with default values

    while len(fringe) > 0:                                          # Execute until fringe is empty
        max_fringe_size = max(max_fringe_size, len(fringe))
        node = fringe.pop()                                         # Pop node from fringe to explore
        nodes_popped += 1                                           # Increment the popped nodes count

        if node.state == end_node.state:                            # Check if current state matches the goal state of the board
            depth, cost, moves = reconstruct_path(node, "DFS")      # Reconstruct path if a goal state has been reached
            if log_file is True:                                    # Log only if requested from command line
                logProgress(visited, fringe, "DFS")

            # return search statistics to calling function
            return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

        nodes_generated, nodes_expanded = exploreChildren(node, visited, fringe, None, "DFS")   # Explore all child nodes

    return None


def iterativelyDeepeningSearch(start_state, end_state, log_file=False):
    """
    Performs Iterative Deepening Depth-First Search to find a path from start_state to end_state.

    Args:
        start_state (list): The initial state.
        end_state (list): The goal state.
        log_file (bool): If True, log the search process to a file.

    Returns:
        A tuple of search statistics and the path from start_state to end_state, or None if no path is found.
    """

    start_node = Node(start_state)                                  # The starting state of the board
    end_node = Node(end_state)                                      # The goal state for the board
    depth_limit = 0                                                 # Initialise with default values
    max_fringe_size = 1                                             # Initialise with default values

    while True:
        fringe = [[start_node]]                                     # Fringe that includes all nodes, visited and unvisited
        visited = set()                                             # Set of all nodes that have been visited
        visited.add(tuple(map(tuple, start_node.state)))            # Keep appending to it
        nodes_popped = 0                                            # Initialise variables with default values
        nodes_expanded = 0                                          # Initialise variables with default values
        nodes_generated = 1                                         # Initialise variables with default values

        while fringe:                                               # Execute until fringe is empty
            current_level = fringe[-1]

            if not current_level:
                fringe.pop()                                        # Pop node from fringe to explore
                continue

            node = current_level.pop()                              # Pop node from current depth level
            nodes_popped += 1                                       # Increment count of popped nodes

            if node.state == end_node.state:                        # Check if current state matches the goal state of the board
                depth, cost, moves = reconstruct_path(node, "IDS")  # Reconstruct path if a goal state has been reached

                if log_file:                                        # Log only if requested from command line
                    logProgress(visited, fringe, "IDS")

                # return search statistics to calling function
                return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

            if node.depth < depth_limit:
                nodes_generated, nodes_expanded = exploreChildren(node, visited, fringe, None, "IDS")

                if len(fringe[-1]) > max_fringe_size:               # Update max_fringe_size if needed
                    max_fringe_size = len(fringe[-1])

        depth_limit += 1                                            # Increment depth limit
        if depth_limit > len(start_state) * len(start_state[0]):
            return None


def uniformCostSearch(start_state, end_state, log_file=False):
    """
    Performs Uniform Cost Search to find a path from start_state to end_state.

    Args:
        start_state (list): The initial state.
        end_state (list): The goal state.
        log_file (bool): If True, log the search process to a file.

    Returns:
        A tuple of search statistics and the path from start_state to end_state, or None if no path is found.
    """
    start_node = Node(start_state)                                  # The starting state of the board
    end_node = Node(end_state)                                      # The goal state for the board
    visited = set()                                                 # Set of all nodes that have been visited
    visited.add(tuple(map(tuple, start_node.state)))                # Keep appending to it
    fringe = PriorityQueue()                                        # Fringe that includes all nodes, visited and unvisited

    cost = 0                                                        # Calculate cost

    fringe.put((cost, start_node))                                  # Append to fringe in order of cost

    max_fringe_size = 1                                             # Initialise variables with default values
    nodes_popped = 0                                                # Initialise variables with default values
    nodes_expanded =  0                                             # Initialise variables with default values
    nodes_generated = 1                                             # Initialise variables with default values

    while not fringe.empty():                                       # Execute until fringe is empty
        max_fringe_size = max(max_fringe_size, fringe.qsize())
        node = fringe.get()[1]                                      # Pop node from fringe
        nodes_popped += 1                                           # Increment the popped nodes count

        if node.state == end_node.state:                            # Reconstruct path if a goal state has been reached
            depth, cost, moves = reconstruct_path(node, "DFS")

            if log_file is True:                                    # Log only if requested from command line
                logProgress(visited, fringe, "UCS")

            return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

        nodes_generated, nodes_expanded = exploreChildren(node, visited, fringe, None, "UCS")     # Explore all child nodes
    return None


def greedyFirstSearch(start_state, end_state, log_file=False):
    """
    Performs Greedy First Search to find a path from start_state to end_state.

    Args:
        start_state (list): The initial state.
        end_state (list): The goal state.
        log_file (bool): If True, log the search process to a file.

    Returns:
        A tuple of search statistics and the path from start_state to end_state, or None if no path is found.
    """
    start_node = Node(start_state)                                      # The starting state of the board
    end_node = Node(end_state)                                          # The goal state for the board
    visited = set()                                                     # Set of all nodes that have been visited
    visited.add(tuple(map(tuple, start_node.state)))                    # Keep appending to it
    fringe = PriorityQueue()                                            # Fringe that includes all nodes, visited and unvisited

    heuristic = manhattan_distance(start_state, end_state)              # Calculate the manhattan distance

    fringe.put((heuristic, start_node))                                 # Append to fringe in order of heuristic value

    max_fringe_size = 1                                                 # Initialise variables with default values
    nodes_popped = 0                                                    # Initialise variables with default values
    nodes_expanded = 0                                                  # Initialise variables with default values
    nodes_generated = 1                                                 # Initialise variables with default values

    while not fringe.empty():                                           # Execute until fringe is empty
        max_fringe_size = max(max_fringe_size, fringe.qsize())
        node = fringe.get()[1]                                          # Pop node from fringe to explore
        nodes_popped += 1                                               # Increment the popped nodes count

        if node.state == end_node.state:                                # Reconstruct path if a goal state has been reached
            depth, cost, moves = reconstruct_path(node, "GFS")

            if log_file is True:                                        # Log only if requested from command line
                logProgress(visited, fringe, "GFS")

            # return search statistics to calling function
            return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

        nodes_generated, nodes_expanded = exploreChildren(node, visited, fringe, end_state, "GFS")      # Explore all child nodes
    return None


def aStarSearch(start_state, end_state, log_file=None):
    """
    Performs Uniform Cost Search to find a path from start_state to end_state.

    Args:
        start_state (list): The initial state.
        end_state (list): The goal state.
        log_file (bool): If True, log the search process to a file.

    Returns:
        A tuple of search statistics and the path from start_state to end_state, or None if no path is found.
    """
    start_node = Node(start_state)                                      # The starting state of the board
    end_node = Node(end_state)                                          # The goal state for the board
    visited = set()                                                     # Set of all nodes that have been visited
    visited.add(tuple(map(tuple, start_node.state)))                    # Keep appending to it
    fringe = PriorityQueue()                                            # Fringe that includes all nodes, visited and unvisited

    heuristic = manhattan_distance(start_state, end_state)              # Calculate the manhattan distance
    cost = heuristic + 0                                                # Calculate cost

    fringe.put((cost, start_node))                                      # Insert into fringe in order of cost

    max_fringe_size = 1                                                 # Initialise variables with default values
    nodes_popped = 0                                                    # Initialise variables with default values
    nodes_expanded = 0                                                  # Initialise variables with default values
    nodes_generated = 1                                                 # Initialise variables with default values

    while not fringe.empty():
        max_fringe_size = max(max_fringe_size, fringe.qsize())
        node = fringe.get()[1]
        nodes_popped += 1                                               # Increment the popped nodes count

        if node.state == end_node.state:                                # Reconstruct path if a goal state has been reached
            depth, cost, moves = reconstruct_path(node, "ASS")

            if log_file is True:                                        # Log only if requested from command line
                logProgress(visited, fringe, "ASS")

            # return search statistics to calling function
            return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

        nodes_generated, nodes_expanded = exploreChildren(node, visited, fringe, end_state, "ASS")      # Explore all child nodes

    return None


def depthLimitedSearch(start_state, end_state,depth_limit, log_file=False):
    """
    Performs Depth Limited Search to find a path from start_state to end_state.

    Args:
        start_state (list): The initial state.
        end_state (list): The goal state.
        log_file (bool): If True, log the search process to a file.

    Returns:
        A tuple of search statistics and the path from start_state to end_state, or None if no path is found.
    """
    start_node = Node(start_state)                                  # The starting state of the board
    end_node = Node(end_state)                                      # The goal state for the board
    max_fringe_size = 1                                             # Initialise with default values

    while True:
        fringe = [[start_node]]                                     # Fringe that includes all nodes, visited and unvisited
        visited = set()                                             # Set of all nodes that have been visited
        visited.add(tuple(map(tuple, start_node.state)))            # Keep appending to it
        nodes_popped = 0                                            # Initialise variables with default values
        nodes_expanded = 0                                          # Initialise variables with default values
        nodes_generated = 1                                         # Initialise variables with default values

        while fringe:                                               # Execute until fringe is empty
            current_level = fringe[-1]

            if not current_level:
                fringe.pop()                                        # Pop node from fringe to explore
                continue

            node = current_level.pop()                              # Pop node from current depth level
            nodes_popped += 1                                       # Increment count of popped nodes

            if node.state == end_node.state:                        # Check if current state matches the goal state of the board
                depth, cost, moves = reconstruct_path(node, "DLS")  # Reconstruct path if a goal state has been reached
                if (depth < depth_limit):

                    if log_file:                                    # Log only if requested from command line
                        logProgress(visited, fringe, "DLS")

                    # return search statistics to calling function
                    return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size
                else:
                    print("Reached maximum permitted depth")
                    return None

            if node.depth < depth_limit:
                nodes_generated, nodes_expanded = exploreChildren(node, visited, fringe, None, "DLS")

                if len(fringe[-1]) > max_fringe_size:               # Update max_fringe_size if needed
                    max_fringe_size = len(fringe[-1])

        if depth > depth_limit:
            return None


# Practically the main function that invokes the right algorithm from from cli parameters
cli = parser()
cli.add_argument('start', help='start state file')
cli.add_argument('end', help='end state file')
cli.add_argument('algorithm', help='Algorithm to use | '
                    'BFS = Breadth First Search | '
                    'DFS = Depth First Search | '
                    'DLS = Depth Limited Search | '
                    'IDS = Iteratively Deepening Search | '
                    'UCS = Uniform Cost Search | '
                    'GFS = Greedy first Search | '
                    'ASS = A* Search')

cli.add_argument('log', help='Log to file | l=log nl=no log')
cli_args = cli.parse_args()

if (cli_args.log) == 'l':
    log = True
else:
    log = False

start_state = readBoard(cli_args.start)
end_state = readBoard(cli_args.end)

if (cli_args.algorithm == 'BFS'):
    result = breadthFirstSearch(start_state, end_state, log)

elif (cli_args.algorithm == 'DFS'):
    result = depthFirstSearch(start_state, end_state, log)

elif (cli_args.algorithm == 'DLS'):
    depth = int(input("Enter depth limit: "))
    result = depthLimitedSearch(start_state, end_state, depth, log)

elif (cli_args.algorithm == 'IDS'):
    result = iterativelyDeepeningSearch(start_state, end_state, log)

elif (cli_args.algorithm == 'UCS'):
    result = uniformCostSearch(start_state, end_state, log)

elif (cli_args.algorithm == 'GFS'):
    result = greedyFirstSearch(start_state, end_state, log)

elif (cli_args.algorithm == 'ASS'):
    result = aStarSearch(start_state, end_state, log)

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
