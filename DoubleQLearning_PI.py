import numpy as np
import pandas as pd

'''
Initialize Q(s,a), all s in S, a in A(s), arbitrarily, and Q(terminal-state, .) = 0
Repeat (for each episode):
    Initialize S
    Repeat (for each step of episode):
        Choose A from S using policy derived from Q (e.g. epsilon-greedy)
        Take action A, observe R, S'
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a(Q(S',A)) - Q(S,A)]
        S <- S'
    until S is terminal
'''

class DoubleQLearning:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table1 = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table2 = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="Double Q-Learning"


    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation):
        self.check_state_exist_q1(observation)
        self.check_state_exist_q2(observation)

        #BUG: Epsilon should be .1 and signify the small probability of NOT choosing max action
        if np.random.uniform() >= self.epsilon:
            state_action1 = self.q_table1.loc[observation, :]
            state_action2 = self.q_table2.loc[observation, :]
            prob1 = np.max(state_action1)
            prob2 = np.max(state_action2)
            if prob1 > prob2:
                action = np.random.choice(state_action1[state_action1 == np.max(state_action1)].index)
            else:
                action = np.random.choice(state_action2[state_action2 == np.max(state_action2)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    '''Choose the next best action given the state'''
    def choose_best_action_q1(self, observation):
        self.check_state_exist_q1(observation)
        state_action = self.q_table1.loc[observation, :]
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action

    '''Choose the next best action given the state'''
    def choose_best_action_q2(self, observation):
        self.check_state_exist_q2(observation)
        state_action = self.q_table2.loc[observation, :]
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action

    '''Update the Q(S,A) state-action value table using the latest experience
       This is a not a very good learning update 
    '''
    def learn(self, s, a, r, s_):
        self.check_state_exist_q1(s_)
        self.check_state_exist_q2(s_)
        update_q1 = False
        if np.random.random() >= 0.5:
            update_q1 = True

        if s_ != 'terminal':
            if update_q1:
                # update Q1
                best_action = self.choose_best_action_q1(str(s_))
                self.q_table1.loc[s, a] = self.q_table1.loc[s, a] + self.lr * (r + self.gamma * self.q_table2.loc[s_, best_action] - self.q_table1.loc[s, a]);
            else:
                # update Q2
                best_action = self.choose_best_action_q2(str(s_))
                self.q_table2.loc[s, a] = self.q_table2.loc[s, a] + self.lr * (r + self.gamma * self.q_table1.loc[s_, best_action] - self.q_table2.loc[s, a]);
        else:
            if update_q1:
                self.q_table1.loc[s, a] = self.q_table1.loc[s, a] + self.lr * (r + self.q_table2.loc[s, a])  # next state is terminal
            else:
                self.q_table2.loc[s, a] = self.q_table2.loc[s, a] + self.lr * (r + self.q_table1.loc[s, a])  # next state is terminal
        a_ = self.choose_action(s_);
        return s_, a_

    '''States are dynamically added to the Q(S,A) table as they are encountered'''
    def check_state_exist_q1(self, state):
        if state not in self.q_table1.index:
            # append new state to q table
            self.q_table1 = self.q_table1.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table1.columns,
                    name=state,
                )
            )
        return

    '''States are dynamically added to the Q(S,A) table as they are encountered'''
    def check_state_exist_q2(self, state):
        if state not in self.q_table2.index:
            # append new state to q table
            self.q_table2 = self.q_table2.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table2.columns,
                    name=state,
                )
            )
        return
