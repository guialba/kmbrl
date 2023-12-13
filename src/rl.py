from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.nn import Model

class Trajectory:
    def __init__(self, env=None, size=None, policy=None):
        self.policy = (lambda _: np.random.choice(4, 1)[0]) if policy is None else policy
        self.env = env
        self.size = size or np.inf

        self.run = pd.DataFrame({'step':[], 's':[], 'a':[], 'r':[], 's_':[], 'end':[]})
        
        if env is not None:
            self.generate()

    def step(self, step, s, a, r, s_, end):
        self.run = pd.concat([self.run, pd.DataFrame({'step':[int(step)], 's':[s], 'a':[int(a)], 'r':[int(r)], 's_':[s_], 'end':[int(end)]})])

    def generate(self):   
        i, end = 0, False
        s = self.env.reset()
        while not end:
            a = self.policy(s)
            s_, r, end = self.env.step(a)
            i += 1
            self.step(i, s, a, r, s_, end)
            # self.run = pd.concat([self.run, pd.DataFrame({'step':[int(i)], 's':[s], 'a':[int(a)], 'r':[int(r)], 's_':[s_], 'end':[int(end)]})])
            # self.run.append((i, s,a,r,s_ end))
            end = end or (i>=self.size)
            s = s_
        return self.run
    
    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        
        df_s = pd.DataFrame(self.run.s.to_list() + self.run.s_.to_list()[-1:], columns=['x','y'])
        ax.plot(df_s.x, df_s.y, color='red')

        return ax
    
    
class Node:
    def __init__(self, value, parent=None, cost=None):
        self.value = value
        self.cost = cost
        self.parent = parent
        self.children = []

class RRT:
    def __init__(self, start, transition, goal_dist):
        """
        start: strat point in numpy array format [x,y]
        transition: transition function (state, action) -> state  
        """
        self.tree = Node(start)
        self.start = start
        self.transition = transition
        self.goal_dist = goal_dist
        self.actions = list(range(4))

    # def explore(self, max_iterations=100):
    #     new_node = self.tree
    #     for _ in range(max_iterations):
    #         random_action = np.random.choice(4, 1)[0]
    #         random_point = self.transition(new_node.value, random_action)
    #         nearest_node = self.find_nearest(random_point)
    #         new_node = self.extend(nearest_node, random_point, random_action)

    ## In goal's direction
    # def explore(self, max_iterations=100):
    #     new_node = self.tree
    #     for _ in range(max_iterations):
    #         moves = [self.transition(new_node.value, a) for a in self.actions]
    #         best = np.argmin(np.array([self.goal_dist(p) for p in moves]))
    #         nearest_node = self.find_nearest(moves[best])
    #         new_node = self.extend(nearest_node, moves[best], self.actions[best])
    #         if self.goal_dist(moves[best]) <= 0.1:
    #             break

    def explore(self, max_iterations=100, node=None):
        old_node = node or self.tree
        if max_iterations>0:
            moves = [self.transition(old_node.value, a) for a in self.actions]
            for i,move in enumerate(moves):
                # best = np.argmin(np.array([self.goal_dist(p) for p in moves]))
                # nearest_node = self.find_nearest(move)
                new_node = self.extend(old_node, move, self.actions[i])
                if self.goal_dist(move) <= 0.1:
                    return
                else:
                    self.explore(max_iterations-1, new_node)



    def get_path(self, point):
        node = self.find_nearest(point)
        path = [node] if node.cost is not None else []
        while node.parent is not None:
            node = node.parent
            if node.cost is not None: 
                path.append(node)
        return path

    def find_nearest(self, point, tree=None):
        node = tree or self.tree
        best = node
        best_distance = np.linalg.norm(np.array(best.value) - np.array(point))
        for child in node.children:
            node = self.find_nearest(point, child)
            distance = np.linalg.norm(np.array(node.value) - np.array(point))
            if distance < best_distance:
                best_distance = distance
                best = node
        return best

    def extend(self, from_node, to_point, action):
        new_node = Node(to_point, from_node, action)
        from_node.children.append(new_node)
        # from_node.cost = action
        return new_node
    
    def plot(self, ax=None, path=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.plot(self.start[0], self.start[1], 'go', markersize=10)
        # ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10)
        
        def plt_node(tree):
            ax.plot(tree.value[0], tree.value[1], 'bo', markersize=2)
            if tree.parent is not None:
                ax.plot([tree.parent.value[0], tree.value[0]], [tree.parent.value[1], tree.value[1]], 'b-', linewidth=.5)
            for child in tree.children:
                plt_node(child)

        plt_node(self.tree)
        if path is not None:
            path_acts = []
            for node in path:
                ax.plot(node.value[0], node.value[1], 'ro', markersize=2)
                if node.parent is not None:
                    ax.plot([node.parent.value[0], node.value[0]], [node.parent.value[1], node.value[1]], 'r-', linewidth=.5)
                if node.cost is not None:
                    path_acts.append(f'{node.cost}')
            ax.text(-10,-14, '-'.join(path_acts))
        return ax

   
class Agent:
    def __init__(self, env, td_model_steps=5, memory_size=0, max_plan_size=100, state_space_size=50, model=Model, **model_params):
        self.env = env
        self.td_model_steps = td_model_steps
        self.memory_size = memory_size
        self.max_plan_size = max_plan_size
        self.model_params = model_params
        self.model =  model(self.env, **model_params)
        self.lls =  []

        self.state_space_size = state_space_size
        
        self.reset()
        
    def reset(self):
        self.trajectory = None
        self.plan_trees = []
        self.plans = []
        self.plan_step = 0

    def discretize_state(self, s):
        bounds = [self.env.observation_space.low.tolist(), self.env.observation_space.high.tolist()]
        discretizer = KBinsDiscretizer(n_bins=self.state_space_size, encode='ordinal', strategy='uniform')
        discretizer.fit(bounds)
        discrete_state = discretizer.transform([s])[0]
        return tuple(map(int, discrete_state))

        # return (np.array(s)*100).astype(int)  

    def simple_plan(self, s):
        calc_dist = lambda s_: min([((goal.low[0]-s_[0])**2 + (goal.low[1]-s_[1])**2)**(1/2) for goal in self.env.goals])
        sigma, tau = self.model.infer(s)
        dists = np.array([calc_dist(self.env.transition(s,a,tau,sigma)) for a in range(4)])

        return np.argmin(dists)
    
    def plan(self, s):
        def transition(s,a):
            sigma, tau = self.model.infer(s)
            return self.env.transition(s,a,tau,sigma)
        def goal_dist(s):
            mid_point = lambda goal: np.array([goal.low[0] + ((goal.low[0] - goal.high[0])**2)**.5, goal.low[1] + ((goal.low[1] - goal.high[1])**2)**.5])
            return min([np.linalg.norm(mid_point(goal)-s) for goal in self.env.goals])

        has_no_plan = (len(self.plans)==0) or (self.plan_step >= len(self.plans[-1]))
        if has_no_plan:
            self.plan_step = 1
            plan_tree = RRT(s, transition, goal_dist)
            plan_tree.explore(max_iterations=self.max_plan_size)

            closest_goal = np.argmin(np.array([np.linalg.norm(goal.low-s) for goal in self.env.goals]))
            goal_point = self.env.goals[closest_goal].low
            actual_plan = plan_tree.get_path(goal_point)

            self.plan_trees.append(plan_tree)
            self.plans.append(actual_plan)
        
        try:
            step = self.plans[-1][::-1][self.plan_step]
            self.plan_step += 1
            return step.cost
        except:
            print('---ERRO---')
            print('self.plans: ', len(self.plans))
            if len(self.plans) > 0:
                print('self.plans[-1]: ', len(self.plans[-1]))
                print('self.plan_step: ', self.plan_step)
            return 0
        
 
    def episode(self, size_limit=None, log=False):
        if self.trajectory is None:
            step, terminated,  = 0, False,
            trajectory = Trajectory()

            S = self.env.reset()
        else:
            step, terminated,  = 0, self.trajectory.run.iloc[-1].end,
            trajectory = self.trajectory

            S = self.trajectory.run.iloc[-1].s_

        if log:
            print(S)

        while not terminated:
            # take action
            A = self.plan(S)
            S_, R, terminated = self.env.step(A)
            trajectory.step(step, S, A, R, S_, terminated)
            
            if log:
                print(step, S,A,R, S_)

            if step>0 and not(step % self.td_model_steps):
                self.train(trajectory)

            S=S_
            step += 1   
            if size_limit is not None:
                terminated = terminated or (step >= size_limit)

        self.trajectory = trajectory  
        return trajectory

    def train(self, trajectory, epochs=1000, log=False, seed=0):
        np.random.seed(seed)

        try:
            ll = self.model.batch_train(trajectory.run[-self.memory_size:], epochs)
            self.lls += ll
            # ll = self.model.batch_train(trajectory.run, epochs)
            if log:
                print(ll[0], ll[-1])
                # self.model.plot(trajectory.plot(self.env.plot(background=False))).get_figure().savefig(f'logs\epi-{epi}.png')
        except:
            print('-------Erro no Treinamento-----------')

    def plot(self, kind='values', **params):
        fig = None
        def plot_background(axs):
            for ax in axs:
                ax = self.env.plot(ax, background=False)
                ax = self.trajectory.plot(ax)

        if  str.lower(kind)=='values':
            fig, ax = plt.subplots(ncols=self.model.n_params, figsize=(self.model.n_params*5, 5))
            plot_background(ax)
            self.model.plot_values(ax)

        elif  str.lower(kind)=='probs':
            fig, ax = plt.subplots(ncols=self.model.k, figsize=(self.model.k*5, 5))
            plot_background(ax)
            self.model.plot_probs(ax)

        elif  str.lower(kind)=='plan':
            p = params.get('plan') or -1
            fig, ax = plt.subplots(figsize=(5, 5))
            ax = [ax]
            plot_background(ax)
            self.plan_trees[p].plot(ax[0], path=self.plans[p])
        
        return fig

    

        
    
    