import gym
import numpy as np
import matplotlib.pyplot as plt

# Initialize the Frozen Lake environment (non-slippery)
environment = gym.make("FrozenLake-v1", is_slippery=False)

# Hyperparameters
alpha = 0.5           # Learning rate
gamma = 0.9           # Discount factor
epsilon = 1.0         # Epsilon for the epsilon-greedy algorithm
epsilon_decay = 0.001 # Decay rate for epsilon
episodes = 1000       # Number of training episodes

# Initialize the Q-table
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# List to store the outcomes
outcomes = []

# Training loop
for _ in range(episodes):
    state = environment.reset()
    done = False
    outcomes.append("Failure")  # Default outcome is failure

    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = environment.action_space.sample()  # Random action
        else:
            action = np.argmax(qtable[state])  # Best action based on Q-table
        
        # Take the action and observe the new state and reward
        new_state, reward, done, _ = environment.step(action)

        # Update the Q-value using the Q-learning formula
        qtable[state, action] = qtable[state, action] + alpha * (
            reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        # Update the state
        state = new_state

        # If we received a reward, mark the outcome as a success
        if reward:
            outcomes[-1] = "Success"

    # Decay epsilon
    epsilon = max(epsilon - epsilon_decay, 0)

# Plotting outcomes
plt.figure(figsize=(12, 5))
plt.xlabel("Episode Number")
plt.ylabel("Outcome")
plt.bar(range(len(outcomes)), outcomes, width=1.0)
plt.show()

# Evaluation of the trained agent
nb_success = 0
evaluation_episodes = 100

for _ in range(evaluation_episodes):
    state = environment.reset()
    done = False

    while not done:
        action = np.argmax(qtable[state])  # Choose best action
        new_state, reward, done, _ = environment.step(action)
        state = new_state
        nb_success += reward

# Print the success rate
print(f"Success rate = {nb_success / evaluation_episodes * 100:.2f}%")

