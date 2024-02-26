import gymnasium as gym
import numpy as np

import utils

# OpenAI gym is a library containing a number of environmets for reinforcement learning.
# It main advantage is that it provides the same interface to all of the environments.
# The implementation of various types of reinforcement learning methods can thus be simple
# and general.

# We can implement a simple class for a random agent -- you can use it as a base
# for the reinforcement learning agent.
class RandomAgent:

    def __init__(self, actions):
        self.actions = actions
        self.train = True

    def act(self, observe, reward, done):
        return self.actions.sample()

    def reset(self):
        pass


# Let's create a simple environment I have shown in the lecture -
# MountainCar (https://gym.openai.com/envs/MountainCar-v0/).
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    # We can get the observation space and action space with these few lines of code.
    print('observation space:', env.observation_space)
    print('observation space low:', env.observation_space.low)
    print('observation space high:', env.observation_space.high)
    print('action space:', env.action_space)

    # Before starting the simulation, we need to reset the environment, this also
    # gives us the first observation.
    obs, info = env.reset()
    print('initial observation:', obs)

    # OpenAI gym allows us to quickly get a random action using the sample() method
    action = env.action_space.sample()

    # We can apply the action in the environment using the step(action) method. It returns
    # a new observation, the reward for this step, information about the end of the simulation
    # (two values - terminated, truncated) and potentially additional info as a dictionary.
    obs, r, terminated, truncated, info = env.step(action)
    print('next observation:', obs)
    print('reward:', r)
    print('terminated:', terminated)
    print('truncated:', truncated)
    print('info:', info)

    #import solution  # this class will be provided later, once you try implementing the agent yourself

    #agent = solution.QLearningAgent(env.action_space, solution.StateDiscretizer(
    #    list(zip(env.observation_space.low, env.observation_space.high)), [15, 15]), True)

    agent = RandomAgent(env.action_space)

    # We will need a lot of iterations to train the agent, we can use a technique similar to the one below.

    total_rewards = []
    for i in range(1000):  # 5000 runs in the environments
        obs, info = env.reset()
        agent.reset()

        done = False
        terminated = False
        r = 0
        R = 0  # total reward - it is used only for logging
        t = 0  # step number - also used only for logging

        while not (done or terminated):
            action = agent.act(obs, r, done or terminated)
            obs, r, done, terminated, _ = env.step(action)
            R += r
            t += 1

        total_rewards.append(R)
        print(f"Iteration: {i}, reward: {R}")

    agent.train = False
    import matplotlib.pyplot as plt

    plt.imshow(agent.Q.max(axis=1).reshape(15, 15))
    plt.show()

    env = gym.make('MountainCar-v0', render_mode='human')

    utils.show_animation(agent, env, steps=1000, episodes=5)

    # display the learning progress - show() ensures that the window will stay open until we close it
    plt.plot(utils.moving_average(total_rewards, 10))
    plt.show()

    env.close()
