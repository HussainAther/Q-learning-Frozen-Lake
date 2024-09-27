import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_decay=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.qtable = np.zeros((state_size, action_size))
        
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.qtable[state])
        
    def update_qtable(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.qtable[next_state])
        td_target = reward + self.gamma * self.qtable[next_state, best_next_action]
        td_error = td_target - self.qtable[state, action]
        self.qtable[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, 0)

    def save(self, filename="qtable.npy"):
        np.save(filename, self.qtable)
    
    def load(self, filename="qtable.npy"):
        self.qtable = np.load(filename)

