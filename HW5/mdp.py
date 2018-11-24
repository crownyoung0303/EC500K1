from __future__ import division
import math
import sys
import random

import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt
import pylab



class MDP(object):

    """A Markov Decision Process.
    Define class members
    S: [int] The number of states;
    A: [int] The number of acions;
    T: [array]
        Transition matrices. The simplest way is using a numpy 
        array that has the shape ``(A, S, S)``. Each element with index
        [a,s,s'] represent the transition probability T(s, a, s'). 
        When state or action space is overwhelmingly large and sparse,
        then ``scipy.sparse.csr_matrix`` matrices can be used.
    R: [array]
        Reward matrices or vectors. Let's use the simplest form with the 
        shape ``(S,)``. Each element with index s is the reward R(s).
        Still ``scipy.sparse.csr_matrix`` can be used instead of numpy arrays.    
    gamma: [float] Discount factor. The per time-step discount factor on future
        rewards. The value should be greater than 0 up to and including 1.
        If the discount factor is 1, then convergence cannot be assumed and a
        warning will be displayed. 
    epsilon : [float]
        Error bound. The maximum change in the value function at each
        iteration is compared against. Once the change falls below
        this value, then the value function is considered to have converged to
        the optimal value function.
    max_iter : [int]
        Maximum number of iterations. The algorithm will be terminated once
        this many iterations have elapsed. 
    """

    def __init__(self, num_states, num_actions, transitions, rewards, discount, epsilon, max_iter):
        # Set the number of states and number of actions
        self.S = int(num_states)
        self.A = int(num_actions)
        
        # Set the maximum iteration number
        if max_iter is not None:
            self.max_iter = int(max_iter)
            assert self.max_iter > 0, (
                "Warning: the maximum number of iterations must be greater than 0.")
        else:
            self.max_iter = 10000
            
        # Set the discount factor
        if discount is not None:
            self.gamma = float(discount)
            assert 0.0 < self.gamma <= 1.0, (
                "Warning: discount rate must be in (0, 1]")
        else:
            self.gamma = 0.99
        # check that error bound is approperiate
        
        if epsilon is not None:
            self.epsilon = float(epsilon)
            assert self.epsilon > 0, (
            "Warning: epsilon must be greater than 0.")
        else:
            self.epsilon = 1E-5
            
        if transitions is not None:
            self.T = np.asarray(transitions)
            assert self.T.shape == (self.A, self.S, self.S), (
            "Warning: the shape of transition function does not match with state and action space")
        else:
            self.T = np.zeros([self.A, self.S, self.S])
            
        if rewards is not None:
            self.R = np.asarray(rewards).astype(float)
            assert self.R.shape == (self.S, ), (
                "Warning: the shape of reward function does not match with state space")
        else:
            self.R = np.random.random(self.S)
            
        # Reset the initial iteration number to zero
        self.iter = 0
        
        # Reset value matrix to None
        # Since value function is mapping from state space to real value. 
        # When it is initialized, it should be a numpy array with shape (S,)
        self.V = None
        
        # Reset Q matrix to None
        # When it is initialized, it should be a numpy array with shape (A, S)
        self.Q = None
        
        # Reset policy matrix to None
        # It should have the shape (S,). Each element is the choosen action
        self.policy = None
    def BellmanUpdate(self):
        pass

class gridworld(object):
    # Firsly define the MDP for the gridworld. 
    # The MDP should have 8*8=64 states to represent all the states.
    # There should be 5 actions: moving left, moving up, moving right, moving down, staying.
    # Firstly initialize the transition and reward function with an all zero matrix
    def __init__(self, dimension = 8, probability = 0.8):
        self.dim = dimension
        self.prob = probability
        self.M = MDP(num_states = dimension**2, num_actions = 5, transitions = np.zeros([5, dimension**2, dimension**2]), 
                     rewards = np.zeros([dimension**2]), discount = 0.999, epsilon = 1e-4, max_iter = 100) 
        
        self.__build_transitions__()
        self.__build_rewards__()
    
    def __coord_to_index__(self, coord):
        # Then translate the coordinate to index
        index = 0
        base = 1
        for i in range(len(coord)):
            index += coord[len(coord) - 1 - i] * base 
            base *= self.dim
        return int(index)  
    
    def __index_to_coord__(self, index):
        # Then translate the state index to coord
        return [int(index/self.dim),int(index)%int(self.dim) ]
    
    def __build_transitions__(self):
        self.M.T = list()
        for a in range(self.M.A):
            self.M.T.append(np.zeros([self.M.S, self.M.S]).astype(float))
            for y in range(self.dim):
                for x in range(self.dim):
                    s = self.__coord_to_index__([y, x])
                    if a == 0:
                        # Action 0 means staying
                        self.M.T[a][s, s] = 1.0
                        continue
                    # 20% probability of moving in random direction
                    self.M.T[a][s, s] += (1 - self.prob)/5.0
                    
                    # Action 4 means going up, y is reduced by 1, x doesn't change 
                    s_ = self.__coord_to_index__([abs(y-1), x])
                    self.M.T[a][s, s_] += (1 - self.prob)/5.0 + int(a == 4) * self.prob
                    
                    # Action 3 means going down, y doesn't change, x is reduced by 1  
                    s_ = self.__coord_to_index__([y, abs(x-1)])
                    self.M.T[a][s, s_] += (1 - self.prob)/5.0 + int(a == 3) * self.prob

                    # Action 2 means going down, y add 1, x doesn't change 
                    s_ = self.__coord_to_index__([self.dim - 1 - abs(self.dim - 1  - y - 1), x])
                    self.M.T[a][s, s_] += (1 - self.prob)/5.0 + int(a == 2) * self.prob

                    # Action 1 means going right, y does not change, x add 1
                    s_ = self.__coord_to_index__([y, self.dim - 1 - abs(self.dim - 1 - x - 1)])
                    self.M.T[a][s, s_] += (1 - self.prob)/5.0 + int(a == 1) * self.prob
         
        self.M.T = np.asarray(self.M.T)
        
    def __build_rewards__(self):
        # The 64th cell with coord [7, 7] has the highest reward
        # The reward function is a radial basis function
        self.M.R = np.asarray(range(self.M.S))
        for s in range(self.M.S):
            coord = self.__index_to_coord__(s)
            self.M.R[s] = - 1.0 * np.linalg.norm(np.array(coord).astype(float) 
                          - np.array([self.dim - 1, self.dim - 1]).astype(float), ord = 2)
        self.M.R = np.exp(self.M.R).astype(float)
        
        for s in range(self.M.S):
            coord = self.__index_to_coord__(s)
            self.M.R[s] = -1.0 * np.linalg.norm(np.array(coord).astype(float) 
                          - np.array([self.dim/2 - 1, self.dim/2 - 1]).astype(float), ord = 2)
        self.M.R = self.M.R - np.exp(self.M.R).astype(float)
        
        self.M.R = self.M.R/(np.max(self.M.R) + np.min(self.M.R))


    def draw_grids(self, rewards = None, title = None):
        # Draw the reward mapping of the grid world with grey scale
        if rewards is None:
            rewards = self.M.R
        R = np.zeros([self.dim, self.dim])
        for i in range(self.dim):
            for j in range(self.dim):
                R[i, j] = rewards[self.__coord_to_index__([i, j])]
        if title is None:
            title = 'Reward mapping'
        pylab.title(title)
        pylab.set_cmap('gray')
        pylab.axis([0, self.dim, self.dim, 0])
        c = pylab.pcolor(R, edgecolors='w', linewidths=1)
        pylab.show()
    
    def draw_plot(self, rewards = None, values = None, title = None):
        # Draw the reward or value plot with state indices being the x-axle
        if rewards is not None:
            plt.ylabel('Reward')
            plt.plot(range(self.M.S), rewards, 'r--') 
        if values is not None:
            plt.ylabel('Value')
            plt.plot(range(self.M.S), values, 'b--')
        plt.xlabel('State Index')
        plt.show()
        
   
    def draw_policy(self, policy = None):
        # Draw the policy mapping of the grid world
        if policy is None:
            policy = self.M.policy
        fig, ax = plt.subplots()
        plt.axis([0, self.dim, self.dim, 0])
        
        colors = ['black', 'red', 'yellow', 'green', 'blue']
        actions = ['stay', 'right', 'down', 'left', 'up']
        for a in range(len(colors)):
            x = list()
            y = list()
            states = (policy==a).nonzero()[0]
            
            for s in states:
                [y_, x_] = self.__index_to_coord__(s)
                y.append(y_ + 0.5)
                x.append(x_ + 0.5)
            ax.scatter(x, y, c=colors[a], s=100, label=actions[a],
                       alpha=0.8, edgecolors='none')

        ax.legend()
        ax.grid(True)

        plt.show()

class PolicyIteration():
    
    ##Design a Policy Iteration algorithm for a given MDP
    
    def __init__(self, MDP, policy_init = None):
        ## Reset the current policy
        
        self.M = MDP
        self.iter = 0
        
        # Check if the user has supplied an initial policy.
        if policy_init is None:
            # Initialise a policy that greedily maximises the one-step reward
            self.M.policy, _ = self.M.BellmanUpdate(np.zeros(self.M.S))
        else:
            # Use the provided initial policy
            self.M.policy = np.array(policy_init)
            # Check the shape of the provided policy
            assert self.policy.shape in ((self.M.S, ), (self.M.S, 1), (1, self.M.S)), \
                ("Warning: initial policy must be a vector with length S.")
            # reshape the policy to be a vector
            self.M.policy = self.M.policy.reshape(self.M.S)
            # The policy must choose from the action space
            msg = "Warning: action out of range."
            assert not np.mod(self.M.policy, 1).any(), msg
            assert (self.M.policy >= 0).all(), msg
            assert (self.M.policy < self.M.A).all(), msg
        # set the initial values to zero
        self.M.V = np.zeros(self.M.S)
    
    
    def TransitionUpdate(self):
        # Compute the transition matrix under the current policy.
        #
        # The transition function MDP.T is a (A, S, S) tensor,
        # The actions in the first dimension are undeterministic.
        #
        # Now the action is determined by the policy
        # The transition function becomes a (S,S) matrix, named P
        #
        # Use the current policy to find out P
        P = np.empty((self.M.S, self.M.S))
        for a in range(self.M.A):
            indices = (self.M.policy == a).nonzero()[0]
            if indices.size > 0:
                P[indices, :] = self.M.T[a][indices, :]
        return P
    
    def ValueUpdate(self):
        pass

    def iterate(self):
        # Run the policy iteration algorithm.
        V_ = np.zeros([self.M.S])
        while True:
            self.iter += 1
            # Calculate the value function resulted from the curretn policy
            # attribute
            self.ValueUpdate()
            
            # Make one step improvement on the policy based on current value function.
            policy_, _ = self.M.BellmanUpdate()
            #print(policy_)
            #print(self.M.V)
            #print(V_)
            # calculate the difference between the newly generated policy and the current policy
            err = (policy_ != self.M.policy).sum()
            #err = np.absolute(self.M.V - V_).max()
            
            # If the difference is smaller than the error bound MDP.epsilon, then stop;
            # Otherwise if the maximum number of iterations has been reached, then stop;
            # Otherwise update the current policy with the newly generated policy
            if err <= self.M.epsilon:
                break
            elif self.iter == self.M.max_iter:
                break
            else:
                self.M.policy = policy_
                V_ = self.M.V.copy()

class wrapper(object):
    def __init__(self, game):
        self.s = 0
        self.M = game.M
        self.game = game
    def num_actions(self):
        return self.M.A
    def num_states(self):
        return self.M.S
    
    def reset(self):
        self.s = 0
        return np.array([self.s])
    def step(self, a):
        if isinstance(a, np.ndarray):
            a = a.flatten()[0]
        assert 0.0 <= a <= self.num_actions(), ("Warning: discount rate must be in (0, 1]")
        p = np.reshape(self.M.T[a, self.s], [self.M.S])
        s_ = np.random.choice(self.M.S, 1, p = p)
        if isinstance(s_, np.ndarray):
            s_ = s_.flatten()[0]
        self.s = s_
        return np.array([self.s]), np.array([self.M.R[int(self.s)]]), self.s == self.M.S - 1, None
    
    def render_rewards(self):
        self.game.draw_grids()
    def render_policy(self, policy):
        self.game.draw_policy(np.asarray(policy))
    
    def close(self):
        self.game = None
        self.M = None
