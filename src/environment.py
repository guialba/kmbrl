from typing import Optional, Tuple, List, Callable, Iterable

import numpy as np

import gym
from gym import spaces

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# TPoint = Tuple[float, float]
# TSquaredRegion = Tuple[TPoint, TPoint]
# TRegion = Callable[[TPoint], int]
# TTransition = Callable[[TPoint, TPoint, int], TPoint]
# TParams = Iterable[Iterable[float]]


# def default_transition(S:TPoint, A:TPoint, region:int = 0) -> TPoint:
#     tau = (1., -1.)
#     sigma = (.01, .3)
#     tr = lambda s,a, t,v: (s + a*t + np.random.normal(0, v, 2)).astype(np.float32)
#     return tuple(tr(np.array(S), np.array(A), tau[region], sigma[region]))


# class MultiRegionContinuosGridWorldEspec:
#     def __init__(self,
#         size: TPoint = (10., 10.),
#         start: TSquaredRegion = ([-5., -9.5], [9.5, -2.]), 
#         goals: List[TSquaredRegion] = [ ([7.5, 7.5], [10., 10.]) ], 
#         region: TRegion = lambda s: (s[1]>-4. and s[1]<4.),
#         params: TParams = [ (1., .01), (-1, .3) ],
#         transition: TTransition = default_transition
#     ):
#     pass


class NormalMoveEnv():
    def __init__(self, 
            wolrd_size=10., 
            punishment=-100, 
            sigma=(.01, .3), 
            tau=(1., -1.), 
            start=([-5., -9.5],[9.5, -2.]), 
            goals=[([7.5,7.5],[10.,10.])], 
            walls=[([-8.,-1.],[10., 1.])],
            beta=None,
            transition=None
        ):
        self.wolrd_size = wolrd_size
        self.start = start
        self.punishment = punishment
        self.actions = np.array([[1,0], [-1,0], [0,1], [0,-1]])
        self.sigma, self.tau = sigma, tau
        self.theta = [None, sigma, tau]
        self.transition = transition or (lambda s,a,t,sig: (s + self.actions[a]*t + np.random.normal(0, sig, 2)).astype(np.float32))
        self.beta = beta or (lambda s: s[1]>1.0)
        self.mu = lambda s,a: self.transition(s,a,self.tau[self.beta(s)], self.sigma[self.beta(s)]).astype(np.float32)
        self.walls = [spaces.Box(low=np.array(wall[0], dtype=np.float32), high=np.array(wall[1], dtype=np.float32), dtype=np.float32) for wall in walls]
        self.goals = [spaces.Box(low=np.array(goal[0], dtype=np.float32), high=np.array(goal[1], dtype=np.float32), dtype=np.float32) for goal in goals]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-wolrd_size, high=wolrd_size, shape=(2,), dtype=np.float32)
        self.state = None
        self.initial_state = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # self.initial_state = np.array([0.,-8.], dtype=np.float32)
        self.initial_state = np.random.uniform(low=self.start[0], high=self.start[1], size=2)
        self.state = self.initial_state
        return self.state
        
    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        
        # m = self.beta(self.state)
        new_state = self.mu(self.state, action, )

        out_bound = (not self.observation_space.contains(new_state))
        in_wall = any([wall.contains(new_state) for wall in self.walls])
        if not out_bound and not in_wall:
            self.state = new_state

        on_goal = any([goal.contains(self.state) for goal in self.goals])
        # terminated = on_goal or out_bound
        reward = not(on_goal) * (self.punishment if out_bound or in_wall else -1)

        return self.state, reward, on_goal
    
    def plot(self, ax=None, background=True):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        # ax.invert_yaxis()
        if background:
            size = 10
            res = 50
            lin = np.linspace(-size, size, res).reshape(-1,1)
            X,Y = np.meshgrid(lin,lin)
            t = [self.beta(s) for row in np.dstack((X,Y)) for s in row]

            p = ax.imshow(np.array(t).reshape(res, res), extent=(int(min(lin))-1, int(max(lin))+1, int(max(lin))+1, int(min(lin))-1), vmin = 0, vmax = 1)
            plt.colorbar(p)
            ax.invert_yaxis()

        diff = lambda a,b: b-a

        width = diff(*list(zip(self.observation_space.low, self.observation_space.high))[0])
        height = diff(*list(zip(self.observation_space.low, self.observation_space.high))[1])
        ax.add_patch(Rectangle(self.observation_space.low, width, height, edgecolor='black', linewidth=1.5, fill=False))

        for wall in self.walls:
            width = diff(*list(zip(wall.low, wall.high))[0])
            height = diff(*list(zip(wall.low, wall.high))[1])
            ax.add_patch(Rectangle(wall.low, width, height, edgecolor='black', facecolor='black', fill=True))

        for goal in self.goals:
            width = diff(*list(zip(goal.low, goal.high))[0])
            height = diff(*list(zip(goal.low, goal.high))[1])
            ax.add_patch(Rectangle(goal.low, width, height, edgecolor='brown', facecolor='green', fill=True, alpha=.5))
            ax.text(*goal.low, f'G', alpha=.5)

        width = diff(*list(zip(self.start[0], self.start[1]))[0])
        height = diff(*list(zip(self.start[0], self.start[1]))[1])
        ax.add_patch(Rectangle(self.start[0], width, height, edgecolor='grey', facecolor='white', fill=True, alpha=.5))
        ax.text(*self.start[0], f'I', alpha=.5)
        if self.initial_state is not None:
            ax.add_patch(Circle(self.initial_state, radius=.2, edgecolor='white', facecolor='white', fill=True))

        return ax
    

class NormalMoveGatedEnv():
    def __init__(self, 
            wolrd_size=10., 
            punishment=-100, 
            sigma=(.01, .3), 
            tau=(1., -1.), 
            start=([-5., -9.5],[9.5, -2.]), 
            goals=[([7.5,7.5],[10.,10.])], 
            walls=[([-8.,-1.],[10., 1.])],
            gates=[([-10.,-1.],[-8., 1.])],
            buttons=[([8.,-2.],[9.,-.5])],
            beta=None,
            transition=None
        ):
        self.wolrd_size = wolrd_size
        self.start = start
        self.punishment = punishment
        self.actions = np.array([[1,0], [-1,0], [0,1], [0,-1]])
        self.sigma, self.tau = sigma, tau
        self.theta = [None, sigma, tau]
        self.transition = transition or (lambda s,a,t,sig: (s + self.actions[a]*t + np.random.normal(0, sig, 2)).astype(np.float32))
        self.beta = beta or (lambda s: s[1]>1.0)
        self.mu = lambda s,a: self.transition(s,a,self.tau[self.beta(s)], self.sigma[self.beta(s)]).astype(np.float32)
        self.walls = [spaces.Box(low=np.array(wall[0], dtype=np.float32), high=np.array(wall[1], dtype=np.float32), dtype=np.float32) for wall in walls]
        self.gates = [spaces.Box(low=np.array(gate[0], dtype=np.float32), high=np.array(gate[1], dtype=np.float32), dtype=np.float32) for gate in gates]
        self.buttons = [spaces.Box(low=np.array(button[0], dtype=np.float32), high=np.array(button[1], dtype=np.float32), dtype=np.float32) for button in buttons]
        self.pressed_buttons = [False for _ in buttons]
        self.goals = [spaces.Box(low=np.array(goal[0], dtype=np.float32), high=np.array(goal[1], dtype=np.float32), dtype=np.float32) for goal in goals]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-wolrd_size, high=wolrd_size, shape=(2,), dtype=np.float32)
        self.state = None
        self.initial_state = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # self.initial_state = np.array([0.,-8.], dtype=np.float32)
        self.initial_state = np.random.uniform(low=self.start[0], high=self.start[1], size=2)
        self.state = self.initial_state
        self.pressed_buttons = [False for _ in self.buttons]
        return self.state      


    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        
        # m = self.beta(self.state)
        new_state = self.mu(self.state, action)

        out_bound = (not self.observation_space.contains(new_state))
        in_wall = any([wall.contains(new_state) for wall in self.walls])
        in_closed_gate = any([gate.contains(new_state) for i,gate in enumerate(self.gates) if not self.pressed_buttons[i]])
        if not out_bound and not in_wall and not in_closed_gate:
            self.state = new_state

        for i, button in enumerate(self.buttons):
            if button.contains(self.state):
                self.pressed_buttons[i] = True

        on_goal = any([goal.contains(self.state) for goal in self.goals])
        # terminated = on_goal or out_bound
        reward = not(on_goal) * (self.punishment if out_bound or in_wall else -1)

        return self.state, reward, on_goal
    
    def plot(self, ax=None, background=True):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        # ax.invert_yaxis()
        if background:
            size = 10
            res = 50
            lin = np.linspace(-size, size, res).reshape(-1,1)
            X,Y = np.meshgrid(lin,lin)
            t = [self.beta(s) for row in np.dstack((X,Y)) for s in row]

            p = ax.imshow(np.array(t).reshape(res, res), extent=(int(min(lin))-1, int(max(lin))+1, int(max(lin))+1, int(min(lin))-1), vmin = 0, vmax = 1)
            plt.colorbar(p)
            ax.invert_yaxis()

        diff = lambda a,b: b-a

        width = diff(*list(zip(self.observation_space.low, self.observation_space.high))[0])
        height = diff(*list(zip(self.observation_space.low, self.observation_space.high))[1])
        ax.add_patch(Rectangle(self.observation_space.low, width, height, edgecolor='black', linewidth=1.5, fill=False))

        for wall in self.walls:
            width = diff(*list(zip(wall.low, wall.high))[0])
            height = diff(*list(zip(wall.low, wall.high))[1])
            ax.add_patch(Rectangle(wall.low, width, height, edgecolor='black', facecolor='black', fill=True))

        for i, gate in enumerate(self.gates):
            width = diff(*list(zip(gate.low, gate.high))[0])
            height = diff(*list(zip(gate.low, gate.high))[1])
            fill = not self.pressed_buttons[i]
            ax.add_patch(Rectangle(gate.low, width, height, edgecolor='red', facecolor='black', fill=fill, linewidth=2))
        
        for i, button in enumerate(self.buttons):
            width = diff(*list(zip(button.low, button.high))[0])
            height = diff(*list(zip(button.low, button.high))[1])
            fill = not self.pressed_buttons[i]
            ax.add_patch(Rectangle(button.low, width, height, edgecolor='red', facecolor='red', fill=fill))

        for goal in self.goals:
            width = diff(*list(zip(goal.low, goal.high))[0])
            height = diff(*list(zip(goal.low, goal.high))[1])
            ax.add_patch(Rectangle(goal.low, width, height, edgecolor='brown', facecolor='green', fill=True, alpha=.5))
            ax.text(*goal.low, f'G', alpha=.5)

        width = diff(*list(zip(self.start[0], self.start[1]))[0])
        height = diff(*list(zip(self.start[0], self.start[1]))[1])
        ax.add_patch(Rectangle(self.start[0], width, height, edgecolor='grey', facecolor='white', fill=True, alpha=.5))
        ax.text(*self.start[0], f'I', alpha=.5)
        if self.initial_state is not None:
            ax.add_patch(Circle(self.initial_state, radius=.2, edgecolor='white', facecolor='white', fill=True))

        return ax
    

    