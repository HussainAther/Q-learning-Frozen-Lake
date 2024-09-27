import gym
import numpy as np
import itertools
import yaml

# Load configuration from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Define grid search parameters
alpha_values = [0.1, 0.3, 0.5, 0.7]
gamma_values = [0.5, 0.7, 0.9]
epsilon_values = [1.0, 0.8, 0.6]
epsilon_decay_values = [0.001, 0.005, 0.01]
episodes = config["hyperparameters"]["episodes"]

# Initialize environment
env_name = config["environment"]["name"]
is_slippery = config["environment"]["is_slippery"]
environment = gym.make(env_name, is_slippery=is_slippery)

# Perform grid search
best_success_rate = 0
best_params = None

for alpha, gamma, epsilon, epsilon_decay in itertools.product(alpha_values, gamma_values, epsilon_values, epsilon_decay_values):
    qtable = np.zeros((environment.observation_space.n, environment.action_space.n))
    successes = 0

    # Training loop
    for _ in range(episodes):
        state = environment.reset()
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = environment.action_space.sample()
            else:
                action = np.argmax(qtable[state])

            new_state, reward, done, _ = environment.step(action)
            qtable[state, action] += alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
            state = new_state

        epsilon = max(epsilon - epsilon_decay, 0)

    # Evaluation
    for _ in range(100):
        state = environment.reset()
        done = False
        while not done:
            action = np.argmax(qtable[state])
            state, reward, done, _ = environment.step(action)
            successes += reward

    success_rate = successes / 100
    if success_rate > best_success_rate:
        best_success_rate = success_rate
        best_params = (alpha, gamma, epsilon, epsilon_decay)

# Display best parameters
print(f"Best Success Rate: {best_success_rate:.2f}%")
print(f"Best Parameters: alpha={best_params[0]}, gamma={best_params[1]}, epsilon={best_params[2]}, epsilon_decay={best_params[3]}")

