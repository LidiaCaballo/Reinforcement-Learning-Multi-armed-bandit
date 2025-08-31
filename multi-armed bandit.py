import numpy as np
import matplotlib.pyplot as plt

class MultiArmedBandit:
    def __init__(self, n_arms=10, epsilon=0.1, plays=2000):
        # Initialise the multi-armed bandit problem
        self.n_arms = n_arms  # Number of arms (actions)
        self.epsilon = epsilon  # Exploration rate (probability of choosing a random action)
        self.plays = plays  # Number of plays (time steps)
        self.q_true = np.random.normal(0, 1, self.n_arms)  # True action values (unknown to the agent)
        self.q_estimates = np.zeros(self.n_arms)  # Estimated action values (initialised to 0)
        self.action_counts = np.zeros(self.n_arms)  # Number of times each action has been taken

    def select_action(self):
        # Select an action using the epsilon-greedy strategy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)  # Exploration: choose a random action
        else:
            return np.argmax(self.q_estimates)  # Exploitation: choose the action with the highest estimated value

    def get_reward(self, action):
        # Simulate a reward for the chosen action
        # Reward is drawn from a normal distribution with mean = true action value and variance = 1
        return np.random.normal(self.q_true[action], 1)

    def update_estimates(self, action, reward):
        # Update the estimated value of the chosen action using the sample-average method
        self.action_counts[action] += 1  # Increment the count of the chosen action
        # Update the estimate: new estimate = old estimate + (reward - old estimate) / count
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]

    def run(self):
        # Run the multi-armed bandit simulation for a fixed number of plays
        rewards = np.zeros(self.plays)  # Store rewards for each play
        optimal_actions = np.zeros(self.plays)  # Store whether the chosen action was optimal (1) or not (0)
        optimal_action = np.argmax(self.q_true)  # Identify the optimal action (arm with the highest true value)

        for i in range(self.plays):
            action = self.select_action()  # Choose an action
            reward = self.get_reward(action)  # Get the reward for the chosen action
            self.update_estimates(action, reward)  # Update the action value estimate
            rewards[i] = reward  # Store the reward
            if action == optimal_action:
                optimal_actions[i] = 1  # Mark if the chosen action was optimal

        return rewards, optimal_actions  # Return the rewards and optimal action indicators


def experiment(n_arms=10, epsilon_values=[0, 0.01, 0.1], plays=2000, runs=2000):
    # Run the multi-armed bandit experiment for different epsilon values
    avg_rewards = {eps: np.zeros(plays) for eps in epsilon_values}  # Store average rewards for each epsilon
    optimal_action_perc = {eps: np.zeros(plays) for eps in epsilon_values}  # Store percentage of optimal actions for each epsilon

    for eps in epsilon_values:
        for _ in range(runs):
            # Create a new bandit instance for each run
            bandit = MultiArmedBandit(n_arms, epsilon=eps, plays=plays)
            rewards, optimal_actions = bandit.run()  # Run the bandit simulation
            avg_rewards[eps] += rewards  # Accumulate rewards across runs
            optimal_action_perc[eps] += optimal_actions  # Accumulate optimal action indicators across runs
        avg_rewards[eps] /= runs  # Compute the average reward over all runs
        optimal_action_perc[eps] = (optimal_action_perc[eps] / runs) * 100  # Compute the percentage of optimal actions

    return avg_rewards, optimal_action_perc  # Return the results


def plot_results(avg_rewards, optimal_action_perc, epsilon_values, plays):
    # Plot the results of the experiment
    plt.figure(figsize=(12, 5))

    # Plot average rewards over time
    plt.subplot(1, 2, 1)
    for eps in epsilon_values:
        plt.plot(avg_rewards[eps], label=f"ε = {eps}")  # Plot average rewards for each epsilon
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.title("Average Reward Over Time")

    # Plot percentage of optimal actions over time
    plt.subplot(1, 2, 2)
    for eps in epsilon_values:
        plt.plot(optimal_action_perc[eps], label=f"ε = {eps}")  # Plot optimal action percentage for each epsilon
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.legend()
    plt.title("Optimal Action Selection Over Time")

    plt.show()  # Display the plots


# Run the experiment and plot the results
n_arms = 10  # Number of arms
plays = 1000  # Number of plays (time steps)
epsilon_values = [0, 0.01, 0.1]  # Different epsilon values to test
avg_rewards, optimal_action_perc = experiment(n_arms=n_arms, epsilon_values=epsilon_values, plays=plays)  # Run the experiment
plot_results(avg_rewards, optimal_action_perc, epsilon_values, plays)  # Plot the results
