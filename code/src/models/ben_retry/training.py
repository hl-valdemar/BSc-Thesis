import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from setuptools._distutils.util import strtobool
from torch import Tensor, nn, optim

from environments.nchain import NChainEnv

from .AleatoricNetwork import construct_aleatoric_flow
from .BayesModel import BayesModel
from .ConditionerMLP import ConditionerMLP
from .EmpiricalTarget import EmpiricalTarget
from .EpistemicNetwork import construct_epsitemic_flow
from .posterior_updating import posterior_update
from .prior_initialization import prior_initialization
from .QBayesNet import QBayesNetwork

torch.set_grad_enabled(True)

# Needed for stability reasons for abs flow
torch.set_default_dtype(torch.float64)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    parser.add_argument("--lr_q", type=float, default=1e-4, help="learning rate for q")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="device to evaluate on",
    )
    parser.add_argument(
        "--lr_phi", type=float, default=1e-4, help="learning rate for epistemic"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    # parser.add_argument(
    #     "--cuda",
    #     type=lambda x: bool(strtobool(x)),
    #     default=False,
    #     nargs="?",
    #     const=False,
    #     help="if toggled, cuda will be enabled by default",
    # )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--prior-initialization",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="pretrain the q network",
    )
    args = parser.parse_args()
    return args


class QMyopicNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 120)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(84, output_size)

    def forward(self, otp1):
        cc = torch.Tensor([[otp1]], device=self.fc1.weight.device)
        x1 = self.fc1(cc)
        x1 = self.relu1(x1)
        x2 = self.fc2(x1)
        x2 = self.tanh(x2)
        return self.out(x2)

    def reset_net(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.out.reset_parameters()


obs_prob = 0.85
horizon = 11
learning_rate = 0.02
memory = 10
N_episodes = 100
N_pretrain = 100
n_msbbe_steps = 10

# Collect metrics during training
all_q_losses: List[float] = []
all_epistemic_losses: List[float] = []
all_rewards: List[int] = []
all_branches: List[int] = []
cumulative_returns: List[int] = []


def train(chain_length: int = 5, max_steps_per_episode: int = 10):
    max_steps_per_episode = max(max_steps_per_episode, chain_length * 2)

    args = parse_args()
    gamma = args.gamma

    cumulative_return = 0

    # Additional config params
    # args.device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # Initialise Env
    env = NChainEnv(n=chain_length, rewards=[-200, 10, 100])
    env.reset()
    (aleatoric_flow, num_params_aleatoric, base_dist_al) = construct_aleatoric_flow(
        6, 2, device=args.device
    )
    epistemic_flow, prior_ep = construct_epsitemic_flow(
        num_params_aleatoric, 2, aleatoric_flow, device=args.device
    )
    q_net = QBayesNetwork(
        input_size=np.prod(env.current_state.to_tensor().shape) + 2,
        output_size=env.num_actions,
    ).to(args.device)
    q_net = q_net.to(args.device)
    hidden_size = q_net.rnn.hidden_size
    q_net.init_hidden()
    q_net.init_params()
    conditioner_net = ConditionerMLP(
        num_params_aleatoric + 1 + hidden_size,
        num_params_aleatoric,
        num_params_aleatoric,
        device=args.device,
    ).to(args.device)
    target_dist = EmpiricalTarget(q_net, env)

    model = BayesModel(
        conditioner_net,
        epistemic_flow,
        aleatoric_flow,
        target_dist,
    )

    # Initialise Q-Network
    optimiser_q = optim.Adam(q_net.parameters(), lr=args.lr_q)
    optimiser_phi = optim.Adam(model.parameters(), lr=args.lr_phi)

    # Initialise Policy

    # Initialise environment
    env.reset()
    o = env.current_state
    o_init = o.to_tensor().clone()

    # initialise history
    h_o: List[Tensor] = []
    h_r: List[Tensor] = []
    h_a: List[Tensor] = []

    # Train
    optimiser_q.zero_grad()
    optimiser_phi.zero_grad()
    episode = 0
    t = 0
    hidden = q_net.init_hidden()

    action_init = (
        torch.tensor(int(env.sample_valid_actions(k=1)[0]), dtype=int)
        .view(1, -1)
        .to(args.device)
    )
    reward_init = torch.tensor(0.0).view(1, -1).to(args.device)
    state_init = o.to_tensor(dtype=torch.float64).view(1, -1).to(args.device)

    inputs = torch.cat(
        (state_init, action_init, reward_init),
        dim=1,
    )

    q_val, _ = q_net(inputs, hidden)
    if args.prior_initialization:
        loss_q, q_net, optimiser_q = prior_initialization(
            envs=env,
            prior_phi=prior_ep,
            base_dist_al=base_dist_al,
            action_init=action_init,
            reward_init=reward_init,
            s_0=state_init,
            q_network=q_net,
            model=model,
            hist_length=hidden_size,
            N_pretrain=N_pretrain,
            gamma=args.gamma,
            device=args.device,
            args=args,
        )

    while episode < N_episodes:
        print(f"\nEpisode {episode + 1}")

        # commented out prior adjustment of log scales.... for epistemic and aleatoric as test
        model = BayesModel(
            conditioner_net,
            epistemic_flow,
            aleatoric_flow,
            target_dist,
        ).to(args.device)

        env.reset()
        o = env.current_state
        o = o.to_tensor(dtype=torch.float64).view(1, -1).to(args.device)

        h_o = []
        h_r = []
        h_a = []

        h_o.append(o)
        h_a.append(action_init)
        h_r.append(reward_init)
        t = 0

        hidden = q_net.init_hidden()

        reward_init = torch.tensor(0.0, dtype=torch.float64).view(1, -1).to(args.device)
        a = action_init

        for t in range(max_steps_per_episode):
            o = env.current_state

            q_net, model, losses_q, losses_epistemic = posterior_update(
                h_init=hidden,
                t=t,
                model=model,
                q_network=q_net,
                h_states=h_o,
                h_actions=h_a,
                h_rewards=h_r,
                action_init=action_init,
                reward_init=reward_init,
                args=args,
                env=env,
                state_init=o,
                N_update=n_msbbe_steps,
                memory=memory,
                device=args.device,
            )

            all_q_losses.extend(losses_q)
            all_epistemic_losses.extend(losses_epistemic)

            if isinstance(a, torch.Tensor):
                a = a.item()

            # print("\nStepping:")
            # print(f"  position: {o.position}")
            # print(f"  action: {a}")
            next_state, reward, done = env.step(a)
            assert reward is not None and a is not None

            inputs = torch.cat(
                (
                    o.to_tensor(dtype=torch.float64).view(1, -1).to(args.device),
                    torch.tensor(a, dtype=torch.float64).view(1, -1).to(args.device),
                    torch.tensor(reward, dtype=torch.float64)
                    .view(1, -1)
                    .to(args.device),
                ),
                dim=1,
            )
            qs, hidden = q_net.forward(inputs, hidden)
            # print(f"\nQ-values shape: {qs.shape}")
            # print(f"  Values before mask: {qs}")

            valid_actions = env.get_valid_actions()
            mask = torch.zeros((qs.shape[0], env.num_actions), dtype=torch.bool)
            mask[:, valid_actions] = True

            qs = qs.masked_fill(~mask, float("-inf"))
            # print(f"  Values after mask: {qs}")

            a = torch.argmax(qs, -1)

            h_a.append(a.clone().detach().view(1, -1).to(args.device))
            h_r.append(
                torch.tensor(reward, dtype=torch.float64).view(1, -1).to(args.device)
            )
            h_o.append(
                env.current_state.to_tensor(dtype=torch.float64)
                .view(1, -1)
                .to(args.device)
            )

            x_vals = [1, 0]
            y_vals = [1, 0]

            t += 1

            if done:
                all_rewards.append(reward)
                all_branches.append(env.current_state.branch)

                cumulative_return += reward
                cumulative_returns.append(cumulative_return)

                episode += 1

                q_net.reset_net()
                hidden = q_net.init_hidden()
                t = 0
                env.reset()
                o = env.current_state
                h_o = []
                h_r = h_r[-1]
                h_a = h_a[-1]

                # Break out when done
                break

    plot_ben_metrics(
        losses_q=all_q_losses,
        losses_epistemic=all_epistemic_losses,
        rewards=all_rewards,
        branches=all_branches,
        cumulative_returns=cumulative_returns,
    )


def plot_ben_metrics(
    losses_q: List[float],
    losses_epistemic: List[float],
    rewards: List[float],
    branches: List[int],
    cumulative_returns: List[int],
    window_size: int = 50,
):
    """
    Plot key metrics from BEN training including MSBBE loss, ELBO loss,
    rewards, and branch choices.

    Args:
        losses_q: List of MSBBE (Q-network) losses
        losses_epistemic: List of ELBO (epistemic) losses
        rewards: List of rewards received
        branches: List tuples of branches chosen (episode, branch)
        window_size: Size of window for running average
    """
    # Convert inputs to numpy arrays
    losses_q = np.array(losses_q)
    losses_epistemic = np.array(losses_epistemic)
    rewards = np.array(rewards)
    branches = np.array(branches)

    # Calculate a usable window
    def window(x: NDArray[any] | List[any]) -> int:
        return min(window_size, len(x) // 2)

    # Calculate running averages
    def running_average(data: np.ndarray, window: int) -> np.ndarray:
        return np.convolve(data, np.ones(window) / window, mode="valid")

    ra_losses_q = running_average(losses_q, window(losses_q))
    ra_losses_epistemic = running_average(losses_epistemic, window(losses_epistemic))
    ra_rewards = running_average(rewards, window(rewards))

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle("BEN Training Metrics", fontsize=16)

    # Plot MSBBE Loss
    ax1.plot(losses_q, alpha=0.3, color="blue", label="Raw")
    ax1.plot(
        np.arange(window(losses_q) - 1, len(losses_q)),
        ra_losses_q,
        color="orange",
        label=f"Running Avg (window={window(losses_q)})",
    )
    ax1.set_title("MSBBE Loss Over Time")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot ELBO Loss
    ax2.plot(losses_epistemic, alpha=0.3, color="red", label="Raw")
    ax2.plot(
        np.arange(window(losses_epistemic) - 1, len(losses_epistemic)),
        ra_losses_epistemic,
        color="orange",
        label=f"Running Avg (window={window(losses_epistemic)})",
    )
    ax2.set_title("ELBO Loss Over Time")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot Rewards
    ax3.plot(rewards, alpha=0.3, color="green", label="Raw")
    ax3.plot(
        np.arange(window(rewards) - 1, len(rewards)),
        ra_rewards,
        color="orange",
        label=f"Running Avg (window={window(rewards)})",
    )
    ax3.set_title("Rewards Over Time")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Reward")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot Cumulative Rewards
    ax4.plot(cumulative_returns, alpha=0.3, color="green", label="Raw")
    ax4.plot(
        np.arange(window(cumulative_returns) - 1, len(cumulative_returns)),
        ra_rewards,
        color="orange",
        label=f"Running Avg (window={window(cumulative_returns)})",
    )
    ax4.set_title("Cumulative Returns")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Return")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot Branch Choices
    steps = np.arange(len(branches))
    scatter = ax5.scatter(steps, branches, c=branches, cmap="viridis", alpha=0.6, s=20)
    ax5.set_title("Branch Choices Over Time")
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Branch")
    ax5.set_yticks([-1, 0, 1])
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
