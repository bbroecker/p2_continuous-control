[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Install
This project requires **Python 3.5** and the following libaries:

- [NumPy](http://www.numpy.org/)
- [Torch](https://pytorch.org)
- [UnityAgents](https://github.com/Unity-Technologies/ml-agents)
- [OpenAI Gym](https://gym.openai.com)

### Instructions

Navigate to the `p2_continuous-control/notebooks/` directory and start jupyter notebook:

```shell
$ ipython3 notebook
```
Chose `A2C_Continuous_Control.ipynb` or `DDPG_Continuous_Control.ipynb` to test the A2C or DDPG implementations, respectively.
Follow the instruction in the notebooks to train your own agent or test the best performing weights.

### Report

Read a detailed report, describing appied approaches and achieved results [here](https://github.com/bbroecker/p2_continuous-control/blob/master/Report.pdf) 


![Result](https://github.com/bbroecker/p2_continuous-control/blob/master/figures/ddpg.png)
