import gym
import numpy as np
import yaml

# Load configuration from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract hyperparameters from config
env_name = config["environment"]["name"]
alpha = config["hyperparameters"]["alpha"]
gamma = config["hyperparameters"]["gamma"]
epsilon = config["hyperparameters"]["epsilon"]
epsilon_decay = config["hyperparameters"]["epsilon_decay"]
episodes = config["hyperparameters"]["episodes"]

# Initialize the environment with slippery condition
environment = gym.make(env_name, is_slippery=True)

# Initialize Q-table
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# Training process
for episode in range(episodes):
    state = environment.reset()
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = environment.action_space.sample()
        else:
            action = np.argmax(qtable[state])

        # Perform the action
        new_state, reward, done, _ = environment.step(action)

        # Update Q-table using the Q-learning formula
        qtable[state, action] += alpha * (
            reward + gamma * np.max(qtable[new_state]) - qtable[state, action]
        )

        # Update state
        state = new_state

    # Decay epsilon
    epsilon = max(epsilon - epsilon_decay, 0)

# Save the trained Q-table for slippery environment
np.save("slippery_qtable.npy", qtable)
print("Training complete. Q-table saved to slippery_qtable.npy.")

