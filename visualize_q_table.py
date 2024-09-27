import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained Q-table
qtable = np.load("qtable.npy")

# Visualization settings
plt.figure(figsize=(12, 8))
sns.heatmap(qtable, annot=True, cmap="viridis", cbar=True, linewidths=0.5)
plt.title("Q-table Heatmap")
plt.xlabel("Actions")
plt.ylabel("States")
plt.show()

