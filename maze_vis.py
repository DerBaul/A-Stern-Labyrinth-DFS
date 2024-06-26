from maze_model import MarkerAgent, MazeAgent, mesa, MazeModel


def agent_portrayal(agent):
    portrayal = {"Shape": "rect",
                  "w": 1,
                  "h":1,
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "black",
                 "r": 0.5}
    if isinstance(agent, MarkerAgent):
        if agent.state == 0:
            portrayal["Color"] = "red"
        elif agent.state == 1:
            portrayal["Color"] = "pink"
            portrayal["w"]= 0.21
            portrayal["h"]=0.2
        
    elif agent.state == 3:
        portrayal["Color"] = "blue"
    if isinstance (agent, MazeAgent):
        if agent.state == 2:
            portrayal["Color"] = "black"
        elif agent.money > 5:
            portrayal["Color"] = "blue"
        elif agent.money < 3:
            portrayal["Color"] = "gray"
        else:
            portrayal["Color"] = "orange"
    return portrayal

square = 15
grid = mesa.visualization.CanvasGrid(agent_portrayal, square, square, 1000, 1000)
server = mesa.visualization.ModularServer(MazeModel,
                       [grid],
                       "Maze Model",
                       {"width":square, "height":square})
server.port = 8521 # The default
server.launch()