from n_chain import NChain

from .model import GFlowNet
from .train import evaluate_gflownet, train_gflownet


def main():
    # Create environment and model
    env = NChain(n=5)
    model = GFlowNet(
        n_states=env.get_state_space_size(),
        n_actions=env.get_action_space_size(),
        hidden_dim=64,
    )

    # Train
    losses = train_gflownet(env, model)

    # Evaluate
    success_rate = evaluate_gflownet(env, model)
    print(f"Final success rate: {success_rate:.2%}")
