
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle, Rectangle

from src.environment import NormalMoveEnv
from src.rl import Agent, Trajectory

class Experiment:
    def __init__(self, **params):
        self.env_class = params.get('env_class') or NormalMoveEnv
        self.env_params = params.get('env_params') or {}

        self.agents_config = [agent for agent in params.get('agents')]

        self.episodes_size_limit = params.get('episodes_size_limit') or 300
        self.runs =  params.get('runs') or 100
        self.replays =  params.get('replays') or 1

        template = {'run':[], 'play':[]}
        template.update({f'model_{i}':[] for i,_ in enumerate(self.agents_config)})
        self.data = pd.DataFrame(template)

    def reset_run(self):
        self.env = self.env_class(**self.env_params)
        self.agents = [Agent(self.env, **config) for config in self.agents_config]

    def run(self):
        trajectories_sizes = []
        for agent in self.agents:
            try:
                agent.reset()
                agent.episode(size_limit=self.episodes_size_limit)
                trajectories_sizes.append(agent.trajectory.run.shape[0])
            except:
                trajectories_sizes.append(np.nan)
        return trajectories_sizes
    
    def execute(self):
        for n in range(self.runs):
            self.reset_run()
            for m in range(self.replays):
                template = {'run':[n], 'play':[m]}
                data = self.run()
                template.update({f'model_{i}':[d] for i,d in enumerate(data)})
                self.data = pd.concat([self.data, pd.DataFrame(template)])

        
