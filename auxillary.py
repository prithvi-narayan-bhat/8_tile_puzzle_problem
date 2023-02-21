# File contains all auxillary functions required to run each of the search algorithms

from queue import PriorityQueue

'''
    A class to represent a node in the search graph
'''
class Node:
    def __init__(self, state, parent=None, move=None, depth=0, cost=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.depth = depth
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost



'''
    Function to read board state from the given input file
'''
def readBoard(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    state = []                                          # Initialise a list
    for line in lines:
        if (line.split(' ')[0] == 'END'):               # Ignore last line that says 'END OF FILE'
            break

        row = line.split()
        state.append([int(i) for i in row])             # Put them in grouped list of 3 each
    return state


'''
    Function to reorganize the state of the board in a more meaningful way
'''
def saveState(file, state):
    with open(file, 'w') as f:
        for row in state:
            line = ' '.join([str(i) for i in row])
            f.write(line+'\n')


def getBlankPosition(state):
    """
    Returns the position of the blank tile (represented by 0) in the given state.

    Args:
        state (list): A 2D list representing the state of the puzzle.

    Returns:
        A tuple representing the row and column of the blank tile.
    """
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:                        # Blank tile is represented by 0
                return i, j                             # Return position of blank tile


'''
    Function to Log the progress
    Different algorithms require different parameters to be logged
    Therefore the if-elif-else case
'''
def logProgress(visited, fringe, algorithm):
    with open("log_file.txt", 'w') as f:
        f.write('Closed nodes:\n')

        for n in visited:
            f.write(str(n)+'\n')

        f.write('Fringe:\n')

        if algorithm == 'BFS' or algorithm == 'DFS':
            for n in fringe:
                f.write(str(n.state)+'\n')

        elif algorithm == 'IDS' or algorithm == 'DLS':
            for level in fringe:
                for n in level:
                    f.write(str(n.state)+'\n')

        elif algorithm == 'UCS' or algorithm == 'GFS':
            for n in list(fringe.queue):
                f.write(str(n[1].state)+'\n')


'''
    Function to get the child nodes of each node
    Generates child nodes by swapping the blank tile with the tiles above, below, to the left, and to the right of the blank tile, respectively
    Each swap generates a new state for the puzzle, represented by child_state
'''
def getChildren(node):
    children = []
    state = node.state
    # Get the position of the blank tile
    i, j = getBlankPosition(state)
    # Determine the possible child states by moving the blank tile in all legal directions
    if i > 0:
        child_state = [row[:] for row in state]             # 2D list
        child_state[i][j], child_state[i - 1][j] = child_state[i-1][j], child_state[i][j]
        # Create a new node for the child_state
        children.append(Node(child_state, node, 'Up'))
    if j > 0:
        child_state = [row[:] for row in state]             # 2D list
        child_state[i][j], child_state[i][j - 1] = child_state[i][j-1], child_state[i][j]
        # Create a new node for the child_state
        children.append(Node(child_state, node, 'Left'))
    if i < 2:
        child_state = [row[:] for row in state]             # 2D list
        child_state[i][j], child_state[i + 1][j] = child_state[i+1][j], child_state[i][j]
        # Create a new node for the child_state
        children.append(Node(child_state, node, 'Down'))
    if j < 2:
        child_state = [row[:] for row in state]             # 2D list
        child_state[i][j], child_state[i][j + 1] = child_state[i][j+1], child_state[i][j]
        # Create a new node for the child_state
        children.append(Node(child_state, node, 'Right'))
    return children

'''
    Function to explore child nodes of given node
'''
def exploreChildren(node, visited, fringe, end_state, algorithm):

    nodes_expanded = 0                                      # default value
    nodes_generated = 1                                     # default value

    children = getChildren(node)                           # Get all child nodes for give node

    for child in children:                                  # Explore child nodes for the given node

        if tuple(map(tuple, child.state)) not in visited:   # Explore only unvisited nodes and ignore the visited ones
            visited.add(tuple(map(tuple, child.state)))

            if algorithm == "IDS" or algorithm == "DLS":
                fringe[-1].append(child)                    # Insert into fringe

            elif algorithm == "UCS":
                fringe.put((child.cost, child))             # Insert into fringe in order of cost cost

            elif algorithm == "GFS":
                cost = manhattan_distance(child.state, end_state)
                fringe.put(cost, child)                     # Insert into fringe in order of heuristic value

            elif algorithm == "ASS":
                cost = child.cost + manhattan_distance(child.state, end_state)
                fringe.put(cost, child)                     # Insert into fringe in order of sum of heuristic value and cost

            else:
                fringe.append(child)                        # Insert into fringe

            nodes_generated += 1                            # Increment count

    nodes_expanded += 1                                     # Increment count

    return nodes_generated, nodes_expanded

'''
    Function to reconstruct the path taken to arrive at the present state
'''
def reconstruct_path(node, algorithm):

    depth = 0                                               # Initialise values
    cost = 0                                                # Initialise values
    moves = []                                              # Initialise values

    while node.parent is not None:
        depth += 1                                          # Update values
        if algorithm == 'BFS' or algorithm == 'DFS':
            cost += 1                                       # Update values
        elif algorithm == 'IDS' or algorithm == 'UCS' or algorithm == 'GFS' or algorithm == 'ASS' or algorithm == 'DLS':
            cost += node.cost                               # Update cost
        moves.append(node.move)                             # Update values
        node = node.parent
    moves.reverse()                                         # Reconstruct the path taken to reach present state

    return depth, cost, moves


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
            else:
                try:
                    row, col = divmod(goal_state.index(state[i][j]), 3)
                    distance += abs(i - row) + abs(j - col)
                except ValueError:
                    continue
    return distance
