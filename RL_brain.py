import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=.9, alpha=0.9, beta=0.1):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.alpha = alpha
        self.beta = beta

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))  # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            # print("Random")
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)
        q_predict = self.q_table.loc[s, a]

        q_target = r + self.gamma * self.q_table.loc[s_, :].max()-q_predict  # next state is not terminal
        if q_target>=0:
            self.q_table.loc[s, a] +=q_target*self.alpha
        else:
            self.q_table.loc[s, a] += q_target * self.beta
            # with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
            #     print(self.q_table.loc['(0.5, 0.1)', :], "KKKKKKKKOIUUI")


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
        # print(len(self.q_table))
