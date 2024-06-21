#agent
#lebendig oder tot
#x,y

#model
#grid -> singlegrid
#scheduler: simultanousActivation

from turtle import st
import mesa
import mesa.agent
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import math

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
        
    def make_move(self, move):
        marker_agent = MarkerAgent(self.model, self.model.next_id(), 1)
        self.model.grid.place_agent(marker_agent, self.pos)
        self.model.grid.move_agent(self, move)

    def manhattan_distance(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def a_star(self, start, end):
        G = self.model.maze_to_graph()
        open_list = set([start])
        closed_list = set([])
        g = {start: 0}
        parents = {start: start}

        while len(open_list) > 0:
            n = None
            for v in open_list:
                if n == None or g[v] + self.manhattan_distance(v, end) < g[n] + self.manhattan_distance(n, end):
                    n = v
            if n == end or G[n] == end:
                path = []
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.append(start)
                path.reverse()
                return path
            open_list.remove(n)
            closed_list.add(n)
            for (nx, ny) in G.neighbors(n):
                if (nx, ny) in closed_list:
                    continue
                candidate_g = g[n] + 1
                if (nx, ny) not in open_list or candidate_g < g[(nx, ny)]:
                    g[(nx, ny)] = candidate_g
                    parents[(nx, ny)] = n
                    if (nx, ny) not in open_list:
                        open_list.add((nx, ny))
        return None


    def step(self):
        nex_move = self.a_star(self.pos, self.model.get_end())
        print(self.a_star(self.pos, self.model.get_end()))
        self.make_move(nex_move[1])
        #self.find_end()

    def advance(self):
        self.state = self.next_state
        
class MazeModel(mesa.Model):
    def __init__(self, prob, width, height):
        self.width = width
        self.height = height
        super().__init__()
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.scheduler = mesa.time.SimultaneousActivation(self)
        self.agent_counter = 0
        self.agent_counter = 0
        self.agent_counter = 0
        self.maze_map = np.array(create_maze(width, height))
        self.maze = self.maze_map
        self.maze[1,1] = 0
        self.maze[width-2, height-2] = 0
        self.maze_graph = nx.Graph()
        self.agent_at_goal = 0
        self.all_wall_agents = list()
        self.all_marker_agents = list()


        #Place Maze_Agent
        self.Maze_Agent = MazeAgent(self, self.agent_counter, 0)
        self.scheduler.add(self.Maze_Agent)
        self.grid.place_agent(self.Maze_Agent, (1,1))
        self.agent_counter += 1

        #Build maze on canvas
        for x in range(self.width):
            for y in range(self.height):
                state = self.maze[x,y]
                if state:
                    a = WallAgent(self, self.agent_counter, state)
                    #self.scheduler.add(a)
                    #add to grid
                    self.all_wall_agents.append(a)
                    self.grid.place_agent(a, (x,y))
                    self.agent_counter += 1

        #Place Marker for start and end
        m = MarkerAgent(self, self.agent_counter, 0)
        self.all_marker_agents.append(m)
        self.grid.place_agent(m, (1,1))
        self.agent_counter += 1
        m = MarkerAgent(self, self.agent_counter, 0)
        self.all_marker_agents.append(m)
        self.grid.place_agent(m, (width-2, height-2))
        self.agent_counter += 1

        #makes maze_map to maze_graph
        self.maze_graph = self.maze_to_graph()
        print(self.maze_graph)
        
        pos = {(x, y): (x, -y) for x, y in self.maze_graph.nodes()}

        plt.figure(figsize=(8, 8))
        nx.draw(self.maze_graph, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold", edge_color='gray')
        plt.gca().invert_yaxis()  # Invert y axis to match the grid layout
        plt.title("Maze Graph")
        plt.show()

    def maze_to_graph(self):
        graph = nx.Graph()
        width, height = self.maze_map.shape

        for x in range(width):
            for y in range(height):
                if self.maze_map[x, y] == 0:  # Only consider paths
                    pos = (x, y)
                    graph.add_node(pos)
                    # Check the 4 possible neighbors (up, down, left, right)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        neighbor = (x + dx, y + dy)
                        if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                            if self.maze_map[neighbor[0], neighbor[1]] == 0:  # Only connect paths
                                graph.add_edge(pos, neighbor)
        
        return graph
    
    def check_agent_goal(self):
        print(self.Maze_Agent.pos)
        if ((self.width-2, self.height-2) == self.Maze_Agent.pos):
            self.agent_at_goal = 1
    
    def remove_agent(self, agent):
        self.grid.remove_agent(agent)

    def delet_walls(self):
        for agent in self.all_wall_agents:
            self.remove_agent(agent)
    
    def delet_marker(self):
        for agent in self.all_marker_agents:
            self.remove_agent(agent)

    def step(self):
        self.check_agent_goal()
        if self.agent_at_goal:
            self.delet_marker()
            self.delet_walls()
        self.scheduler.step()

    def get_end(self):
        return (self.width-2, self.height-2)
cv = MazeModel(0.4, 5, 5)
cv.step()


#Hausaufgabe
#Ein paar Objekte definieren mit denen man Startet 

[(0, 2), (1, 0), (1, 2), (2, 1), (2, 2)]
