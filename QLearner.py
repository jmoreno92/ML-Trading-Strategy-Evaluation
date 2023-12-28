""""""
"""  		  	   		  	  		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  	  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  	  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  	  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  	  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  		  		  		    	 		 		   		 		  
or edited.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  	  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  	  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Student Name: Jorge Salvador Aguilar Moreno  		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jmoreno62	  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845924		   	   		  	  		  		  		    	 		 		   		 		  
"""

import random as rand

import numpy as np


class QLearner(object):
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    This is a Q learner object.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param num_states: The number of states to consider.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type num_states: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		  	  		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    """
    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False,
    ):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        self.s = 0      # row*10 + column index
        self.a = 0      # 0: north, 1: east, 2: south, 3: west

        self.count = 0  # count for iterations of Q table update

        self.exp = []

        # Initialize with small random numbers
        self.Q = np.random.random((num_states, num_actions)) * 0.000001      # Q table

        # Variables for Dyna, T and R matrices
        if dyna >0:
            self.T = np.empty([num_states, num_actions, num_states])      # T matrix
            self.Tc = np.ones([num_states, num_actions, num_states]) * 0.000001      # T count matrix
            self.R = np.ones([num_states, num_actions]) * 0.000001       # Reward matrix


    def querysetstate(self, s):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Update the state without updating the Q-table  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param s: The new state  		  	   		  	  		  		  		    	 		 		   		 		  
        :type s: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """

        # Roll the dice to know whether I have to take a random action or not.
        # Do not update Q-table nor rar
        if rand.uniform(0.0, 1.0) <= self.rar:  # random action if number is <= than rar
            a = rand.randint(0, self.num_actions - 1)    # Choose random action
        else:
            a = np.argmax(self.Q[s])    # Choose action with the highest Q value

        # Update variables in constructor
        self.s = s
        self.a = a

        if self.verbose:
            print(f"s = {s}, a = {a}")

        return a


    def query(self, s_prime, r):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Update the Q table and return an action  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param s_prime: The new state  		  	   		  	  		  		  		    	 		 		   		 		  
        :type s_prime: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :param r: The immediate reward  		  	   		  	  		  		  		    	 		 		   		 		  
        :type r: float  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """

        # 1) Update Q-table
        # I have to remember s and a. From the function, I know s' and r
        # Local variables to remember state and action we were before
        s = self.s
        a = self.a
        # a = self.querysetstate(self.s)
        self.updateQ(s, a, s_prime, r)

        # Before I pick action
        if self.dyna > 0:
            self.dynaplanning(s,a,s_prime,r)

        # 2) roll the dice to know whether I have to take a random action or not.
        # We start by forcing the system to explore, then random action frequency decays
        # until we are not using random actions at all.
        a_prime = self.querysetstate(s_prime)
        self.rar = self.rar * self.radr     # After each update, decrease rar
        self.count += 1                     # Update variables in constructor
        # self.exp.append( (s, a, s_prime, r) )   # Append experience tuple, commented out to make code faster

        if self.verbose:
            print(f"s = {s_prime}, a = {a_prime}, r={r}")


        return a_prime


    def updateQ(self, s, a, s_prime, r):
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + \
                                self.alpha * (r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime])] )


    def dynaplanning(self,s,a,s_prime, r):
        # Update T'[s,a,s'] and R'[s,a]
        self.Tc[s, a, s_prime] += 1     # increment count of Tc
        self.T[s,a,s_prime] = self.Tc[s,a,s_prime] / ( self.Tc[s, a, :].sum() / self.num_states)    # Update T matrix
        self.R[s,a] = (1 - self.alpha) * self.R[s,a] + (self.alpha * r)  # Update R matrix

        # Good reference for keeping track pf previous states and previovus actions within previous states
        # https://stackoverflow.com/questions/16094563/numpy-get-index-where-value-is-true
        # np.any test whether any array element along a given axis evaluates to True
        # np.nonzero return indices of elements that are non-zero

        # self.Tc.sum(axis=2)      # converts Tc from 3d to 2d, to count actions and states
        past_exp = self.Tc.sum(axis=2) >= 1   # 2D Boolean matrix, (num_states x num_action) dim  for past experiences
        prev_s = np.nonzero(np.any(past_exp, axis = 1))
        # prev_a = np.nonzero(past_exp[prev_s].flatten())

        for i in range(self.dyna):
            s = rand.choice(prev_s[0])    # Choose random state from previously observed state
            prev_a = np.nonzero(past_exp[s].flatten())  # previous actions taken in previously observed state
            a = rand.choice(prev_a[0])   # Choose random action previously taken in previous state
            s_prime = np.argmax(self.T[s,a])
            r = self.R[s,a]
            self.updateQ(s, a, s_prime, r)


    def author(self):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        return "jmoreno62"


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jmoreno62"

if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
