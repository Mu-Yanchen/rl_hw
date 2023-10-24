# import the 4 necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

    
# Creates a table of V_values (state-action) initialized with zeros
# Initialize V(s, a), for all s ∈ S, a ∈ A(s), arbitrarily, and Q(terminal-state, ·) = 0.
def createV_table(rows = 4, cols = 12):
    """
    Implementation of creating a table for the V(s) state value
    
    Args:
        rows -- type(int) Number of rows the simple grid world
        cols -- type(int) Number of columns in the simple grid world
    
    Returns:
        v_table -- type(np.array) 2D representation of state value table 
                                     Rows are actions and columns are the states. 
    """
    # initialize the v_table with all zeros for each state and action
    v_table = np.zeros(cols * rows)

    # define an action dictionary to access corresponding state-action pairs fast
    # action_dict =  {"UP": q_table[0, :],"LEFT": q_table[1, :], "RIGHT": q_table[2, :], "DOWN": q_table[3, :]}
    
    return v_table

# Choosing action using policy
# Sutton's code pseudocode: Choose A from S using policy derived from Q (e.g., ε-greedy)
# %10 exploration to avoid stucking at a local optima
def epsilon_greedy_policy(agent, v_table, epsilon = 0.1):
    """
    Epsilon greedy policy implementation takes the current state and q_value table 
    Determines which action to take based on the epsilon-greedy policy
    
    Args:
        epsilon -- type(float) Determines exploration/explotion ratio
        state -- type(int) Current state of the agent value between [0:47]
        v_table -- type(np.array) Determines state value
        
    Returns:
        action -- type(int) Choosen function based on V(s) pairs & epsilon
    """
    # choose a random int from an uniform distribution [0.0, 1.0) 
    decide_explore_exploit  = np.random.random()
    
    if(decide_explore_exploit < epsilon):
        action = np.random.choice(4) # UP = 0, LEFT = 1, RIGHT = 2, DOWN = 3
    else:
        action = [0,1,2,3]
        agent = [move_agent(agent, i) for i in action]
        # observe next state value
        next_state = [get_state(i) for i in agent]
        action = np.argmax([v_table[i] for i in next_state]) # Choose the action with largest Q-value (state value)
        
    return action

def visited_env(agent, env):
    """
        Visualize the path agent takes
        
    """
    (posY, posX) = agent
    env[posY][posX] = 1
    return env

def get_state(agent):
    """
    Determine the state and state value given agent's position, the relation between state and agent
    
    Args:
        agent -- type(tuple) x, y coordinate of the agent on the grid
        v_table -- type(np.array) Determines state value
        
    Returns:
        state -- type(int) state value between [0,47]
        max_state_value -- type(float) maximum state value at the position of the agent
    """
    # get position of the agent
    (posX , posY) = agent
    
    # obtain the state value
    state = 12 * posX + posY

    # # get maximum state value from the table
    # state_action = v_table[:, int(state)]
    # maximum_state_value = np.amax(state_action) # return the state value with for the highest action
    return state    # , maximum_state_value

def update_vTable(v_table, state,  reward, next_state_value, gamma_discount = 0.9, alpha = 0.5):
    """
    Update the q_table based on observed rewards and maximum next state value
    Sutton's Book pseudocode:  V(S) <- V(S) + [alpha * (reward + (gamma * V(S')) -  V(S) ]
    
    Args:
        v_table -- type(np.array) Determines state value
        state -- type(int) state value between [0,47]
        reward -- type(int) reward in the corresponding state 
        next_state_value -- type(float) maximum state value at next state
        gamma_discount -- type(float) discount factor determines importance of future rewards
        alpha -- type(float) controls learning convergence
        
    Returns:
        v_table -- type(np.array) Determines state value
    """
    update_v_value = v_table[state] + alpha * (reward + (gamma_discount * next_state_value) - v_table[state])
    v_table[state] = update_v_value

    return v_table  

def move_agent(agent, action):
    """
    Moves the agent based on action to take
    
    Args:
        agent -- type(tuple) x, y coordinate of the agent on the grid, attention: <agent is a tuple and state is a int, which has a rela>
        action -- type(int) updates agent's position 
        
    Returns:
        agent -- type(tuple) new coordinate of the agent
    """
    # get position of the agent
    (posX , posY) = agent
    # UP 
    if ((action == 0) and posX > 0):
        posX = posX - 1
    # LEFT
    if((action == 1) and (posY > 0)):
        posY = posY - 1
    # RIGHT
    if((action == 2) and (posY < 11)):
        posY = posY + 1
    # DOWN
    if((action) == 3 and (posX < 3)):
        posX = posX + 1
    agent = (posX, posY)
    
    return agent  

def get_reward(state):
    """
    Function returns reward in the given state
    
    Args:
        state -- type(int) state value between [0,47]
        
    Returns: 
        reward -- type(int) Reward in the corresponding state 
        game_end -- type(bool) Flag indicates game end (falling out of cliff / reaching the goal)
    """
    # game continues
    game_end = False
    # all states except cliff have -1 value
    reward = -1
    # goal state
    if(state == 47):
        game_end = True
        reward = 0
    # cliff
    if(state >= 37 and state <= 46):
        game_end = False
        # Penalize the agent if agent encounters a cliff
        reward = -100

    return reward, game_end

def td_0(num_episodes = 500, gamma_discount = 0.9, alpha = 0.5, epsilon = 0.1):
    """
    Implementation of td(0) algorithm. (Sutton's book), adapted from qlearning
    
    Args:
        num_episodes -- type(int), 500 acts' number of games to train agent
        gamma_discount -- type(float) discount factor determines importance of future rewards
        alpha -- type(float) determines convergence rate of the algorithm (can think as updating states fast or slow)
        epsilon -- type(float) explore/ exploit ratio (exe: default value 0.1 indicates %10 exploration)
        
    Returns:
        v_table -- type(np.array) Determines state value
        reward_cache -- type(list) contains cumulative_reward, which is used to plot a draw
    """
    # initialize all states to 0
    # Terminal state cliff_walking ends
    reward_cache = list()
    step_cache = list()
    v_table = createV_table()
    agent = (3, 0) # 1. starting from left down corner
    # start iterating through the episodes
    for episode in range(0, num_episodes):
        env = np.zeros((4, 12))
        env = visited_env(agent, env)
        agent = (3, 0) # starting from left down corner
        game_end = False
        reward_cum = 0 # cumulative reward of the episode
        step_cum = 0 # keeps number of iterations untill the end of the game
        while(game_end == False):
            # get the state from agent's position
            state = get_state(agent)
            state_value=v_table[state]
            # choose action using epsilon-greedy policy
            action = epsilon_greedy_policy(agent,v_table)
            # move agent to the next state
            agent = move_agent(agent, action)
            env = visited_env(agent, env) # mark the visited path
            step_cum += 1
            # observe next state value
            next_state = get_state(agent)
            max_next_state_value = v_table[state]
            # observe reward and determine whether game ends
            reward, game_end = get_reward(next_state)
            reward_cum += reward 
            # update q_table
            v_table = update_vTable(v_table, state, reward, max_next_state_value, gamma_discount, alpha)
            # update the state
            state = next_state
        reward_cache.append(reward_cum)
        if(episode > 498):
            print("Agent trained with td(0) after 500 iterations")
            print(env) # display the last 2 path agent takes 
        step_cache.append(step_cum)
    return v_table, reward_cache, step_cache

def plot_number_steps(step_cache_qlearning):
    """
        Visualize number of steps taken

    Args:
        step_cache_td  --  type(list) 

    Returns:
        nothing but draw a linepicture
    """    
    cum_step_q = []
    steps_mean = np.array(step_cache_qlearning).mean()
    steps_std = np.array(step_cache_qlearning).std()
    count = 0 # used to determine the batches
    cur_step = 0 # accumulate reward for the batch
    for cache in step_cache_qlearning:
        count = count + 1
        cur_step += cache
        if(count == 10):
            # normalize the sample, sample every 10 epochs
            normalized_step = (cur_step - steps_mean)/steps_std
            cum_step_q.append(normalized_step)
            cur_step = 0
            count = 0
            
       
    # prepare the graph    
    plt.plot(cum_step_q, label = "q_learning")
    plt.ylabel('Number of iterations')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("TD(0) Iteration number untill game ends")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

def plot_cumreward_normalized(reward_cache_qlearning):
    """
    Visualizes the reward convergence
    
    Args:
        reward_cache -- type(list) contains cumulative_reward
        
    Returns:
        nothing but draw a linepicture
    """
    cum_rewards_q = []
    rewards_mean = np.array(reward_cache_qlearning).mean()
    rewards_std = np.array(reward_cache_qlearning).std()
    count = 0 # used to determine the batches
    cur_reward = 0 # accumulate reward for the batch
    for cache in reward_cache_qlearning:
        count = count + 1
        cur_reward += cache
        if(count == 10):
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean)/rewards_std
            cum_rewards_q.append(normalized_reward)
            cur_reward = 0
            count = 0
            
    # prepare the graph    
    plt.plot(cum_rewards_q, label = "q_learning")
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("TD(0) Convergence of Cumulative Reward")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

def generate_heatmap(q_table,kind):
    """
        Generates heatmap to visualize agent's learned actions on the environment

    Args:
        q_table -- type(np.array) 2D representation of state-action pair table 
                                     Rows are actions and columns are the states. 
        kind    -- type(str) kind of strategy

    Returns:
        nothing but draw a heatmap
    """
    sns.set()
    # display mean of environment values using a heatmap
    data = np.mean(q_table, axis = 0)
    print(data)
    data = data.reshape((4, 12))
    ax = sns.heatmap(np.array(data))
    plt.title(f"Heatmap of {kind} the Features", fontsize = 10)
    plt.show()
    return ax

def generate_heatmap(v_table,kind):
    """
        Generates heatmap to visualize agent's learned actions on the environment

    Args:
        q_table -- type(np.array) 2D representation of state-action pair table 
                                     Rows are actions and columns are the states. 
        kind    -- type(str) kind of strategy

    Returns:
        nothing but draw a heatmap
    """
    sns.set()
    # display mean of environment values using a heatmap
    data = v_table.reshape((4, 12))
    ax = sns.heatmap(np.array(data))
    plt.title(f"Heatmap of {kind} the Features", fontsize = 10)
    plt.show()
    return ax

v_table_td,reward_cache_td,step_cache_td = td_0()
plot_number_steps(step_cache_td)
plot_cumreward_normalized(reward_cache_td)
generate_heatmap(v_table_td,'TD(0)')
# ax_q = generate_heatmap(v_table_td,'TD(0)')
# print(ax_q)