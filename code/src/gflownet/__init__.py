from gridworld import GridWorld
from .env import GridWorldEnv
from .config import GFlowNetConfig
from .model import GFlowNet
from .train import GFlowNetTrainingConfig, train, visualize_flows

def main():
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

    # Initialize GFlowNet
    config = GFlowNetConfig(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        learning_rate=1e-4
    )

    gflownet = GFlowNet(config)

    # Training configuration
    training_config = GFlowNetTrainingConfig(
        num_episodes=1000,
        batch_size=32,
        replay_capacity=10000,
        min_experiences=100
    )

    # Train
    rewards = train(env, gflownet, training_config)

    # Plot learning curve
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('GFlowNet Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

    # Visualize flows for a few states
    print("\nFinal Flow Values:")
    test_state = env.reset().state
    visualize_flows(env, gflownet, test_state)
