import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import yaml

# Load configuration from config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract hyperparameters from config
env_name = config["environment"]["name"]
episodes = config["hyperparameters"]["episodes"]
gamma = config["hyperparameters"]["gamma"]
epsilon = config["hyperparameters"]["epsilon"]
epsilon_decay = config["hyperparameters"]["epsilon_decay"]
alpha = config["hyperparameters"]["alpha"] # Learning rate for neural network

# Initialize environment
environment = gym.make(env_name, is_slippery=False)
state_shape = environment.observation_space.n
action_shape = environment.action_space.n

# Build Deep Q-Network model
model = tf.keras.Sequential([
    layers.Dense(24, activation='relu', input_shape=(state_shape,)),
    layers.Dense(24, activation='relu'),
    layers.Dense(action_shape, activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha), loss='mse')

# Training process
for episode in range(episodes):
    state = environment.reset()
    state = np.reshape(state, [1, state_shape])
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = environment.action_space.sample()
        else:
            action_values = model.predict(state)
            action = np.argmax(action_values[0])

        next_state, reward, done, _ = environment.step(action)
        next_state = np.reshape(next_state, [1, state_shape])
        target = reward + gamma * np.max(model.predict(next_state)[0])
        target_values = model.predict(state)
        target_values[0][action] = target
        model.fit(state, target_values, epochs=1, verbose=0)

        state = next_state

    # Decay epsilon
    epsilon = max(epsilon - epsilon_decay, 0)

# Save trained model
model.save("dqn_model.h5")
print("Training complete. Model saved to dqn_model.h5.")

