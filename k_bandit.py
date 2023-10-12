# -*- coding: utf-8 -*-
# @Time : 2023/9/20 20:16
# @Author : Mu Yanchen
# @Email : 2627264809@qq.com
# @File : k_bandit.py
# @Project : 强化学习第一次实验

import numpy as np
import matplotlib.pyplot as plt
#the real mean value of each ation's reward
qa_star = np.array(range(1,16))
#the vars of each action's reward
var_qa = np.ones(15)

# the action is to choose an arm
# Qa is the evaluation of the rewards of actions
# qa_star is the real mean rewards of actions
# reward is the total reward of the policy from the 1st step to the last step

# iteration epoches
steps = 20000

# number of K-arm
armNum = 15
# value_function update strategy
alpha = 0.1
# UCB's rate
c=2
totalReward = np.zeros(steps)
totalReward1= np.zeros(steps)
totalReward2= np.zeros(steps)
totalReward3= np.zeros(steps)

# the evaluation of the rewards of actions during each step
Qa = np.zeros(armNum)
# 乐观初始值
# Qa = np.ones(armNum)
# the times each action had been taken
actionTimes = np.zeros(armNum)


# use a random choice
def selectAnArm():
    temp = np.random.randint(0, 15)
    return int(temp)

# get Reward value via a gauss distribution
def getReward(selectedAction):
    meanQa = qa_star[selectedAction]
    print("meanQa=", meanQa)
    varQa = var_qa[selectedAction]
    print("varQa=", varQa)
    temp = np.random.normal(meanQa, varQa, 1)
    return temp[0]

# uodate function
def updateQa(selectedAction, t, Ra):
    Qa[selectedAction] = Qa[selectedAction] + alpha * (Ra - Qa[selectedAction])
    actionTimes[selectedAction] = actionTimes[selectedAction] + 1


def main():
    for t in range(1, steps):
        # for the 1st time, select an arm randomly because we have the same Qa array
        if t == 1:
            selectedAction = selectAnArm()
        else:
            # choose an action randomly in a probabilty of 0.01，where we use a probability of 0.01 to choose new action
            select = np.random.randint(1000)
            if 1==0:
                selectedAction = selectAnArm()
                # print("keci happened...")
            else:
                # choose the action with the biggest reward in a probabilty of 0.9
                # print("choose the best action")
                temp = Qa
                # if there are more than 1 maximum, choose one randomly
                index = np.where(temp == np.max(temp))
                # print("index=",index)
                ishape = np.shape(index)
                numMax = ishape[1]
                # if there are more than 1 maximum, choose one randomly
                if numMax > 1:
                    i = np.random.randint(0, numMax - 1)
                    selectedAction = index[0][i]
                else:
                    selectedAction = index[0][0]
        print("t=", t, ", action=", selectedAction)
        # get the reward
        Ra = getReward(selectedAction)
        # print("...after get reward")
        # print("selectedAction=",selectedAction)
        # use the selected action to update its Qa
        updateQa(selectedAction, t, Ra)
        # print("...after update Qa")
        # print(Qa)
        totalReward[t] = ((t - 1) / t) * totalReward[t - 1] + Ra / t


def main1():
    for t in range(1, steps):
        # for the 1st time, select an arm randomly because we have the same Qa array
        
        # choose an action randomly in a probabilty of 0.01，where we use a probability of 0.01 to choose new action
        select = np.random.randint(1000)
        if select>100 and select<200: # a possibility of 0.01
            selectedAction = selectAnArm()
            # print("keci happened...")
        else:
            # choose the action with the biggest reward in a probabilty of 0.9
            # print("choose the best action")
            temp = Qa
            # if there are more than 1 maximum, choose one randomly
            index = np.where(temp == np.max(temp))
            # print("index=",index)
            ishape = np.shape(index)
            numMax = ishape[1]
            # if there are more than 1 maximum, choose one randomly
            if numMax > 1:
                i = np.random.randint(0, numMax - 1)
                selectedAction = index[0][i]
            else:
                selectedAction = index[0][0]
        print("t=", t, ", action=", selectedAction)
        # get the reward
        Ra = getReward(selectedAction)
        # print("...after get reward")
        # print("selectedAction=",selectedAction)
        # use the selected action to update its Qa
        updateQa(selectedAction, t, Ra)
        # print("...after update Qa")
        # print(Qa)
        totalReward1[t]=totalReward1[t-1]+(Ra-totalReward1[t-1]) / t

def main2():
    actionTimes = np.ones(armNum)*0.001
    for t in range(1, steps):
        # for the 1st time, select an arm randomly because we have the same Qa array
        
        # if t == 1:
        #     selectedAction = selectAnArm()
        # else:


        # choose the action with the biggest reward in a probabilty of 0.9
        # print("choose the best action")
        temp = Qa + c * np.sqrt(np.log(t+1)/actionTimes)
        # if there are more than 1 maximum, choose one randomly
        index = np.where(temp == np.max(temp))
        # print("index=",index)
        ishape = np.shape(index)
        numMax = ishape[1]
        # if there are more than 1 maximum, choose one randomly
        if numMax > 1:
            i = np.random.randint(0, numMax - 1)
            selectedAction = index[0][i]
        else:
            selectedAction = index[0][0]
        print("t=", t, ", action=", selectedAction)
        # get the reward
        actionTimes[selectedAction] = actionTimes[selectedAction] + 1
        Ra = getReward(selectedAction)
        
        
        # print("...after get reward")
        # print("selectedAction=",selectedAction)
        # use the selected action to update its Qa
        updateQa(selectedAction, t, Ra)
        # print("...after update Qa")
        # print(Qa)
        totalReward2[t]=totalReward2[t-1]+(Ra-totalReward2[t-1]) / t
 

def main3():
    for t in range(1, steps):
        # for the 1st time, select an arm randomly because we have the same Qa array
        if t == 1:
            selectedAction = selectAnArm()
        else:
            # choose an action randomly in a probabilty of 0.01，where we use a probability of 0.01 to choose new action
            select = np.random.randint(1000)
            if 1==0:
                selectedAction = selectAnArm()
                # print("keci happened...")
            else:
                # choose the action with the biggest reward in a probabilty of 0.9
                # print("choose the best action")
                temp = Qa
                # if there are more than 1 maximum, choose one randomly
                index = np.where(temp == np.max(temp))
                # print("index=",index)
                ishape = np.shape(index)
                numMax = ishape[1]
                # if there are more than 1 maximum, choose one randomly
                if numMax > 1:
                    i = np.random.randint(0, numMax - 1)
                    selectedAction = index[0][i]
                else:
                    selectedAction = index[0][0]
        print("t=", t, ", action=", selectedAction)
        # get the reward
        Ra = getReward(selectedAction)
        # print("...after get reward")
        # print("selectedAction=",selectedAction)
        # use the selected action to update its Qa
        updateQa(selectedAction, t, Ra)
        # print("...after update Qa")
        # print(Qa)
        totalReward3[t]=totalReward3[t-1]+(Ra-totalReward3[t-1]) / t
    x = np.linspace(1, steps, steps)
    plt.plot(x, totalReward,label='greedy(ε=0)')
    plt.plot(x, totalReward1,label='ε-greedy(ε=0.01)')
    plt.plot(x, totalReward2,label='UCB')
    plt.plot(x, totalReward3,label='optimistic initialization')
    plt.legend()
    print(Qa)
    plt.show()

if __name__ == '__main__':
    main()
    Qa = np.zeros(armNum)
    main1()
    Qa = np.zeros(armNum)
    # Qa = np.ones(armNum)*5 # 乐观初始值方法
    main2()
    Qa = np.ones(armNum)*5 # 乐观初始值方法
    main3()


