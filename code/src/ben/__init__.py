def main():
    from gridworld import GridWorld
    from .env import GridWorldEnv
    from .model import BENConfig, BayesianExplorationNetwork
    from .train import TrainingConfig, train_ben

    # Create a simple grid world
    layout = """
    #######
    #s   g#
    # o o #
    #     #
    #######
    """

    # Initialize environment
    grid_world = GridWorld(layout=layout)
    env = GridWorldEnv(grid_world)

    # Initialize BEN
    config = BENConfig(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        rnn_hidden_dim=64,
        num_flows=2,
        learning_rate=1e-4,
        gamma=0.99
    )

    ben = BayesianExplorationNetwork(config)

    # Train
    training_config = TrainingConfig(
        num_episodes=1000,
        batch_size=32,
        replay_capacity=10000,
        min_experiences=100
    )

    rewards = train_ben(env, ben, training_config)

    # Plot learning curve
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
