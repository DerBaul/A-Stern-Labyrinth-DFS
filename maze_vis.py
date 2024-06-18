from gol_model import MarkerAgent, MazeAgent, mesa, MazeModel


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
        portrayal["Color"] = "yellow"
    return portrayal

grid = mesa.visualization.CanvasGrid(agent_portrayal, 21, 21, 1000, 1000)
server = mesa.visualization.ModularServer(MazeModel,
                       [grid],
                       "Maze Model",
                       {"prob":0.4, "width":21, "height":21})
server.port = 8521 # The default
server.launch()