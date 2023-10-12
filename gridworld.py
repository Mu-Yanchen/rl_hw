# -*- coding: utf-8 -*-
# @Time : 2023/10/10 11:56
# @Author : Mu Yanchen
# @Email : 2627264809@qq.com
# @File : gridworld.py
# @Project : 强化学习第二次实验

import numpy as np

class Gridworld:
    def __init__(self):
        # define the necessary paras about the gridworld
        # which has 4 cols and 4 rows and 4 actions
        self.num_rows = 4
        self.num_cols = 4
        self.num_states = self.num_rows * self.num_cols
        self.actions = ['Left', 'Right', 'Up', 'Down']
        self.state_values = np.zeros(self.num_states)
        self.discount_factor = 0.9
        

    # policy_iteration, which involves evaluation and improve policy
    def policy_iteration(self):
        policy = np.random.choice(self.actions, size=self.num_states)
    
        for i in range(10000):
            self.evaluate_policy(policy)
            if_policy_stable, policy = self.improve_policy(policy)
            print(f"iteration {i}:\npolicy:\n{policy.reshape(4,4)};\n value:\n{self.state_values.reshape(4,4)}")
            if if_policy_stable:
                break
                
        self.print_results(policy)
        
        print("\nFinal Policy:")
        print(policy.reshape(4,4))

        print("\nFinal Value:")
        print(self.state_values.reshape(4,4))

    # evaluate policy and renew the state_value
    def evaluate_policy(self, policy, theta = 0.001):
        while True:
            deltaw = 0
            for state in range(self.num_states):
                value = self.state_values[state]
                action = policy[state]
                next_state, reward = self.get_next_state_and_reward(state, action)
                self.state_values[state] = reward + self.discount_factor * self.state_values[next_state] 
                # state value's renew
                deltaw = max(deltaw, np.abs(value - self.state_values[state]))
                # judge whether it should be stop or continue(w/for iterate too much)
            if deltaw < theta:
                break
    
    # judge the policy's correctness and improve pi according values and return ifstable+new policy
    def improve_policy(self, policy):
        if_policy_stable = True
        for state in range(self.num_states):
            old_action = policy[state]
            max_value = float('-inf')
            best_action = None
            for action in self.actions:
                next_state, reward = self.get_next_state_and_reward(state, action)
                value = reward + self.discount_factor * self.state_values[next_state]
                if value > max_value:
                    max_value = value
                    best_action = action 
            policy[state] = best_action
            if old_action != best_action:
                if_policy_stable = False        
        return if_policy_stable, policy
    
    def get_next_state_and_reward(self, state, action):
        row = state // self.num_cols
        col = state % self.num_cols
        if action == 'Left':
            col = max(col - 1, 0)
        elif action == 'Right':
            col = min(col + 1, self.num_cols - 1)
        elif action == 'Up':
            row = max(row - 1, 0)
        elif action == 'Down':
            row = min(row + 1, self.num_rows - 1)
            
        next_state = row * self.num_cols + col
        if state == 0 or state == self.num_states - 1:
            reward = 0
            next_state = state
        else:
            reward = -1
        
        return next_state, reward
    
    # print position and value+policy action
    def print_results(self, policy):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                state = row * self.num_cols + col
                action = policy[state]
                print(f"({row},{col}): Value={self.state_values[state]}\t, Action={action}\t",end=';')
            print()        

    # value_iteration
    def value_iteration(self, theta = 0.001):
        while True:
            deltaw = 0
            for state in range(self.num_states):
                value = self.state_values[state]
                max_value = float('-inf')
                for action in self.actions:
                    next_state, reward = self.get_next_state_and_reward(state, action)
                    value = reward + self.discount_factor * self.state_values[next_state]
                    max_value = max(max_value, value)
                
                self.state_values[state] = max_value
                deltaw = max(deltaw, np.abs(value - self.state_values[state]))
            
            if deltaw < theta:
                break
        
        
        
        policy = self.get_optimal_policy()
        self.print_results(policy)

        print("\nFinal Policy:")
        print(policy.reshape(4,4))

        print("\nFinal Value:")
        print(self.state_values.reshape(4,4))

    def get_optimal_policy(self):
        policy = np.random.choice(self.actions, size=self.num_states) 
        for state in range(self.num_states):
            max_value = float('-inf')
            best_action = None
            
            for action in self.actions:
                next_state, reward = self.get_next_state_and_reward(state, action)
                value = reward + self.discount_factor * self.state_values[next_state]
                
                if value > max_value:
                    max_value = value
                    best_action = action
            
            policy[state] = best_action
        
        return policy
    
if __name__ == '__main__':
    # policy iteration
    gw = Gridworld()
    print("Policy Iteration:")
    gw.policy_iteration()

    print('=='*90)

    # value iteration
    gw = Gridworld()
    print("Value Iteration:")
    gw.value_iteration()

    
        
    