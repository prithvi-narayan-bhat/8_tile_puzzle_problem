from queue import PriorityQueue
import argparse

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
def load_state(file):
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
def save_state(file, state):
    with open(file, 'w') as f:
        for row in state:
            line = ' '.join([str(i) for i in row])
            f.write(line+'\n')


'''
    Function to get the position of the blank tile
'''
def get_blank_pos(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:                        # Blank tile is represented by 0
                return i, j

'''
    Function to Log the progress
'''
def LOG(visited, fringe):
    with open("log_file.txt", 'w') as f:
        f.write('Closed nodes:\n')

        for n in visited:
            f.write(str(n)+'\n')

        f.write('Fringe:\n')

        for n in fringe:
            f.write(str(n.state)+'\n')

'''
    Function to get the child nodes of each node
    Generates child nodes by swapping the blank tile with the tiles above, below, to the left, and to the right of the blank tile, respectively
    Each swap generates a new state for the puzzle, represented by child_state
'''
def get_children(node):
    children = []
    state = node.state
    i, j = get_blank_pos(state)                             # Get the position of the blank tile
    # Determine the possible child states by moving the blank tile in all legal directions
    if i > 0:
        child_state = [row[:] for row in state]             # 2D list
        child_state[i][j], child_state[i-1][j] = child_state[i-1][j], child_state[i][j]
        children.append(Node(child_state, node, 'Up'))      # Create a new node for the child_state
    if j > 0:
        child_state = [row[:] for row in state]             # 2D list
        child_state[i][j], child_state[i][j-1] = child_state[i][j-1], child_state[i][j]
        children.append(Node(child_state, node, 'Left'))    # Create a new node for the child_state
    if i < 2:
        child_state = [row[:] for row in state]             # 2D list
        child_state[i][j], child_state[i+1][j] = child_state[i+1][j], child_state[i][j]
        children.append(Node(child_state, node, 'Down'))    # Create a new node for the child_state
    if j < 2:
        child_state = [row[:] for row in state]             # 2D list
        child_state[i][j], child_state[i][j+1] = child_state[i][j+1], child_state[i][j]
        children.append(Node(child_state, node, 'Right'))   # Create a new node for the child_state
    return children

'''
    Function to implement the Breadth First Search algorithm
'''
def bfs(start_state, end_state, log_file=None):

    start_node = Node(start_state)                      # The starting state of the board
    end_node = Node(end_state)                          # The goal state for the board
    visited = set()                                     # Set of all nodes that have been visited
    visited.add(tuple(map(tuple, start_node.state)))    # Keep appending to it
    fringe = [start_node]                               # Fringe that includes all nodes, visited and unvisited
    max_fringe_size = 1                                 # default value
    nodes_popped = 0                                    # default value
    nodes_expanded = 0                                  # default value
    nodes_generated = 1                                 # default value

    # Execute until fringe is empty
    while len(fringe) > 0:
        max_fringe_size = max(max_fringe_size, len(fringe))
        node = fringe.pop(0)                            # Pop from fringe
        nodes_popped += 1                               # Increment the popped nodes count

        if node.state == end_node.state:                # Check if the goal has been reached
            depth = 0                                   # Initialise values
            cost = 0                                    # Initialise values
            moves = []                                  # Initialise values

            while node.parent is not None:
                depth += 1                              # Increment values
                cost += 1                               # Increment values
                moves.append(node.move)                 # Increment values
                node = node.parent
            moves.reverse()                             # Find the path taken so far

            # Log if flag is provided
            if log_file is not None:
                LOG(visited, fringe)

                # with open(log_file, 'w') as f:
                #     f.write('Closed nodes:\n')

                #     for n in visited:
                #         f.write(str(n)+'\n')

                #     f.write('Fringes:\n')

                #     for n in fringe:
                #         f.write(str(n.state)+'\n')

            # return search statistics to calling function
            return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

        children = get_children(node)                   # Get all child nodes for give node

        for child in children:                          # Explore all child nodes for the given node
            if tuple(map(tuple, child.state)) not in visited:
                visited.add(tuple(map(tuple, child.state)))
                fringe.append(child)
                nodes_generated += 1                    # Increment count
        nodes_expanded += 1                             # Increment count
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
            depth = 0                                   # Initialise values
            cost = 0                                    # Initialise values
            moves = []                                  # Initialise values
            while node.parent is not None:
                depth += 1                              # Update values
                cost += 1                               # Update values
                moves.append(node.move)                 # Update values
                node = node.parent
            moves.reverse()                             # Reconstruct the path taken to reach present state

            if log_file is not None:

                with open(log_file, 'w') as f:
                    f.write('Closed nodes:\n')

                    for n in visited:
                        f.write(str(n)+'\n')

                    f.write('Fringes:\n')

                    for n in fringe:
                        f.write(str(n.state)+'\n')

            return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

        children = get_children(node)
        for child in children:
            if tuple(map(tuple, child.state)) not in visited:
                visited.add(tuple(map(tuple, child.state)))
                fringe.append(child)
                nodes_generated += 1
        nodes_expanded += 1
    return None


# def ids(start_state, end_state, log_file=None):
#     start_node = Node(start_state)
#     end_node = Node(end_state)
#     max_depth = 100  # maximum allowed depth
#     max_fringe_size = 1
#     nodes_popped = 0
#     nodes_expanded = 0
#     nodes_generated = 1
#     for depth_limit in range(max_depth):
#         stack = [[start_node]]
#         while len(stack) > 0:
#             max_fringe_size = max(max_fringe_size, len(stack[-1]))
#             try:
#                 node = stack[-1].pop()
#                 nodes_popped += 1                               # Increment the popped nodes count
#             except:
#                 break
#             if node.state == end_node.state:
#                 depth = 0
#                 cost = 0
#                 moves = []
#                 while node.parent is not None:
#                     depth += 1
#                     cost += 1
#                     moves.append(node.move)
#                     node = node.parent
#                 moves.reverse()

#                 if log_file is not None:

#                     with open(log_file, 'w') as f:
#                         f.write('Fringes:\n')

#                         for s in stack:
#                             for n in s:
#                                 f.write(str(n.state)+'\n')

#                 return depth, cost, moves, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size

#             if node.depth < depth_limit:
#                 children = get_children(node)
#                 for child in children:
#                     stack.append([child])
#                     nodes_generated += 1
#             nodes_expanded += 1

#         if log_file is not None:
#             with open(log_file, 'w') as f:
#                 f.write('Fringes:\n')

#                 for s in stack:
#                     for n in s:
#                         f.write(str(n.state)+'\n')

#     return None

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

                if log_file is not None:

                    with open(log_file, 'w') as f:
                        f.write('Closed nodes:\n')

                        for n in visited:
                            f.write(str(n)+'\n')

                        f.write('Fringes:\n')

                        for level in stack:
                            for n in level:
                                f.write(str(n.state)+'\n')

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


def greedy(start_state, end_state, heuristic, log_file=None):
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


parser = argparse.ArgumentParser()
parser.add_argument('start', help='start state file')
parser.add_argument('end', help='end state file')
parser.add_argument('--log', help='log file')
args = parser.parse_args()

start_state = load_state(args.start)
end_state = load_state(args.end)
# heuristic_manhattan = manhattan_distance(start_state, end_state)

# result = bfs(start_state, end_state, 0, args.log)
result = bfs(start_state, end_state, args.log)
# result = astar(start_state, end_state, heuristic_manhattan, args.log)

if result is None:
    print('No solution found.')
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
