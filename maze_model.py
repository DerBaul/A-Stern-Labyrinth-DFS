#agent
#lebendig oder tot
#x,y

#model
#grid -> singlegrid
#scheduler: simultanousActivation

from turtle import st
import mesa
import mesa.agent
from networkx import neighbors
import numpy as np
import random
import heapq


#Hilfs funktionen

def create_maze(width, height):
    # Initialize the grid with walls
    maze = [[1 for _ in range(width)] for _ in range(height)]
    
    # Directions for moving in the grid (right, left, down, up)
    directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
    
    def is_valid(x, y):
        return 0 < x < height-1 and 0 < y < width-1
    
    def carve_path(x, y):
        maze[x][y] = 0  # Mark the cell as part of the path
        
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and maze[nx][ny] == 1:
                maze[nx-dx//2][ny-dy//2] = 0  # Carve a passage
                carve_path(nx, ny)
    
    # Start carving the maze from (1, 1)
    carve_path(1, 1)
    
    # Make sure the starting point is open
    maze[1][1] = 0  # Start point
    
    return maze

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(maze, start, goal):
    height = len(maze)
    width = len(maze[0])
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            reconstruct_path(came_from, current, maze)
            return True
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
                if maze[neighbor[0]][neighbor[1]] == 1:
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return False

def reconstruct_path(came_from, current, maze):
    while current in came_from:
        current = came_from[current]
        if maze[current[0]][current[1]] != 0:  # To avoid overwriting the start point
            maze[current[0]][current[1]] = 4

def print_maze(maze):
    for row in maze:
        print(''.join(str(cell) for cell in row))

def place_stuff(start, obj):
    place_life_at = list()
    for x in range(len(obj[0])):
        for y in range(len(obj) ):
            if obj[y][x]:
                place_life_at.append((x + start[0], y + start[1]))
    return place_life_at

class WallAgent(mesa.Agent):
    def __init__(self, model, id, state=1):
        super().__init__(id, model)
        self.state = state
    
    

class MarkerAgent(mesa.Agent):
    def __init__(self, model, id, state=1):
        super().__init__(id, model)
        self.state = state
        

class MazeAgent(mesa.Agent):
    def __init__(self, model, id, state=0):
        super().__init__(id, model)
        self.state = state
        self.next_state = state

    def find_end(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )

        possible_steps = []
        for neighbor in neighbors:
            inhalt = self.model.grid.get_cell_list_contents([neighbor])
            if inhalt:
                if isinstance(inhalt[0], WallAgent):  # Hier das erste Element überprüfen. Eigentlich nicht nötig, aber zur sicherheit drinnen.
                    continue

                else:
                    possible_steps.append(neighbor)
            else:
                possible_steps.append(neighbor)
        
        #Es wird von den Erlaubten zügen zufällig einer gewält.
        new_position = self.random.choice(possible_steps) 
        marker_agent = MarkerAgent(self.model, self.model.next_id(), 1)
        self.model.grid.place_agent(marker_agent, self.pos)
        self.model.grid.move_agent(self, new_position)
        

    def step(self):
        self.find_end()

    def advance(self):
        self.state = self.next_state

        
class MazeModel(mesa.Model):
    def __init__(self, prob, width, height):
        self.width = width
        self.height = height
        super().__init__()
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.scheduler = mesa.time.SimultaneousActivation(self)
        self.wall_counter = 0
        self.marker_counter = 0
        self.agent_counter = 0
        self.maze_map = np.array(create_maze(width, height))
        self.maze = self.maze_map
        self.maze[1,1] = 0
        self.maze[width-2, height-2] = 0

        #Build maze on canvas
        for x in range(self.width):
            for y in range(self.height):
                state = self.maze[x,y]
                if state:
                    a = WallAgent(self, self.wall_counter, state)
                    #self.scheduler.add(a)
                    #add to grid
                    self.grid.place_agent(a, (x,y))
                    self.wall_counter += 1
        #Place Marker for start and end
        m = MarkerAgent(self, self.marker_counter, 0)
        self.grid.place_agent(m, (1,1))
        self.marker_counter += 1
        m = MarkerAgent(self, self.marker_counter, 0)
        self.grid.place_agent(m, (width-2, height-2))
        
        #Place Maze_Agent
        a = MazeAgent(self, self.agent_counter, 0)
        self.scheduler.add(a)
        self.grid.place_agent(a, (1,1))
        
    def step(self):
        self.scheduler.step()
        

cv = MazeModel(0.4, 5, 5)
cv.step()


#Hausaufgabe
#Ein paar Objekte definieren mit denen man Startet 

[(0, 2), (1, 0), (1, 2), (2, 1), (2, 2)]
