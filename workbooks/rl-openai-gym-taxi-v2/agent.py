import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=.01, gamma=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.epsilon = 1.0
        self.episode = 1
        self.gamma = gamma
        self.alpha = alpha
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
    
    def get_policy(self, state):
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        policy_s[np.argmax(self.Q[state])] = 1 - self.epsilon + (self.epsilon / self.nA)
        return policy_s
    
    def q_learning_next_q(self, state):
        return max(self.Q[state])

    def expected_sarsa_next_q(self, state):
        return np.dot(self.Q[state], self.get_policy(state))

    def get_next_q(self, state):
        return self.expected_sarsa_next_q(state)
    
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy_s = self.get_policy(state)
        return np.random.choice(np.arange(self.nA), p=policy_s)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        if done:
            self.episode += 1
            self.epsilon = 1.0 / (self.episode)

        Qs = self.Q[state][action]
        self.Q[state][action] = Qs + self.alpha * (reward + self.gamma * self.get_next_q(next_state) - Qs)
        