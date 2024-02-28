import numpy as np


class StateDiscretizer:

    def __init__(self, ranges, states):
        self.ranges = ranges
        self.states = states
        self.bins = []

        for r, s in zip(self.ranges, self.states):
            self.bins.append(np.linspace(r[0], r[1], s + 1))

        self.num_states = np.prod(self.states)

    def transform(self, obs):
        locs = [np.digitize(o, b) for (o, b) in zip(obs, self.bins)]

        state_num = 0
        for l, s in zip(locs, self.states):
            state_num = state_num * s + l - 1

        if state_num < 0 or state_num >= self.num_states:
            raise UserWarning('the observation was outside the specified range')

        return state_num if 0 <= state_num < self.num_states else 0  # observation outside specified ranges


class QLearningAgent:

    def __init__(self, actions, state_transformer, train=False, alpha=0.1, gamma=0.9, eps=0.1):
        self.actions = actions
        self.state_transformer = state_transformer
        self.Q = np.zeros((state_transformer.num_states, self.actions.n))
        self.train = train
        self.last_state = None
        self.last_action = None
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def act(self, observe, reward, done):
        current_state = self.state_transformer.transform(observe)
        current_action = np.argmax(self.Q[current_state, :])

        if self.train and self.last_state:
            if np.random.random() < self.eps:
                current_action = self.actions.sample()
            self.Q[self.last_state, self.last_action] = (1 - self.alpha) * self.Q[
                self.last_state, self.last_action] + self.alpha * (reward + self.gamma * self.Q[
                current_state, current_action])

        self.last_state = current_state
        self.last_action = current_action

        return current_action

    def reset(self):
        self.last_state = None
        self.last_action = None
