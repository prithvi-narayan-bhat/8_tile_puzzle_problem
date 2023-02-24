# 8_tile_puzzle_problem
An attempt to build an agent to solve a modified version of the 8 puzzle problem.

This code code is available on Github at the following link
#### https://github.com/prithvi-narayan-bhat/8_tile_puzzle_problem

To clone the repository, enter the follwoing command on a Git supported machine
#### git clone git@github.com:prithvi-narayan-bhat/8_tile_puzzle_problem.git

## Algorithms implemented
1. Breadth First Search
2. Depth First Search
3. Iteratively Deepening Search
4. Uniform Cost Search
5. Greedy First Search
6. A* Search
7. Depth Limited Search

## Execution
### System Requirements
The project was developed and tested on a Linux Mint machine (Kernel 5.15.0-56-generic) with Python3 (Version 3.10.6).
However, I am positive it can be run without any modifications on any compatible system
### Required Files
Ensure the the following files are all present in the same directory when executing
1. expense_8_puzzle.py
2. auxillary.py
3. goal_state.txt
4. start_state.txt
### Command Line Execution
Enter the following to view the CLI help interface
##### python3 expense_8_puzzle.py -h[--help]

To run the application and perform any particular algorithm, run the following
##### python3 expense_8_puzzle.py [puzzle_start_state] [puzzle_goal_state] [algorithm] [log_flag]
#### [puzzle_start_state]
A text file that includes the grid of tiles (tested for 3x3). It could look like this
2 3 6
1 0 7
4 8 5
END OF FILE
#### [puzzle_goal_state]
A text file that includes the grid of tiles (tested for 3x3). It could look like this
1 2 3
4 5 6
7 8 0
END OF FILE

#### [algorithm]
Acceptable algorithms parameters are the following
1. BFS -> Breadth First Search
2. DFS -> Depth First Search
3. IDS -> Iteratively Deepening Search
4. UCS -> Uniform Cost Search
5. GFS -> Greedy First Search
6. ASS -> A* Search
7. DLS -> Depth Limited Search (will receive a second prompt for the maximum permissable depth (Integer value))

##### NOTE: At any given time only one algorithm may be implemented

#### [log_flag]
Flag to either log the progress into a file or not
Acceptable values
1. l -> log = True
2. nl -> log = False

