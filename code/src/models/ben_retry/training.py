import argparse
import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
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
    # parser.add_argument(
    #     "--wandb-project-name",
    #     type=str,
    #     default="cleanRL",
    #     help="the wandb's project name",
    # )
    # parser.add_argument(
    #     "--wandb-entity",
    #     type=str,
    #     default="rl-world",
    #     help="the entity (team) of wandb's project",
    # )
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


matplotlib.rcParams.update({"figure.autolayout": True})
matplotlib.rcParams.update({"figure.figsize": (10, 8)})
matplotlib.rcParams.update({"figure.dpi": 1200})
plt.figure(dpi=1200)
# Import classes
fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
# import wandb

av_cumulative_reward_list: List[float] = []
twentyfifth_percentile_list = []
seventyfifth_percentile_list = []
spaghetti_cumulative_returns = []


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
N_episodes = 5
memory = 10

# Collect metrics during training
all_q_losses = []
all_epistemic_losses = []
all_rewards = []
all_branches = []


def train(max_steps_per_episode: int = 20):
    args = parse_args()
    # wandb.init(project=args.wandb_project_name, entity=args.wandb_entity)
    # wandb.config.update(args, allow_val_change=True)
    gamma = args.gamma
    n_msbbe_steps_multiple = [10]  # 10, #50]# 2, 3, 10, 20]  # 50
    for n_msbbe_steps in n_msbbe_steps_multiple:
        # Additional config params
        cumulative_return = 0
        # args.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        # Initialise Env
        gridwidth = 7
        time_limit = 5 * gridwidth**2
        env = NChainEnv(n=5)
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

        cumulative_returns = np.zeros((N_episodes, max_steps_per_episode))

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
                N_pretrain=5,
                gamma=args.gamma,
                device=args.device,
                args=args,
            )
            # wandb.log({"prior_loss_q": loss_q})

        end_of_episode_cumulative_returns = []
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

            current_loss = []
            hidden = q_net.init_hidden()
            cumulative_returns = np.zeros((N_episodes, max_steps_per_episode))
            # action_init = 3 # What the fuck is this doing here?
            reward_init = (
                torch.tensor(0.0, dtype=torch.float64).view(1, -1).to(args.device)
            )
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

                # wandb.log({"memory": memory, "n_msbbe_steps": n_msbbe_steps})
                current_loss.append(losses_q[-1])
                # wandb.log({"loss_q": losses_q[-1]})
                if isinstance(a, torch.Tensor):
                    a = a.item()
                # action_name = env.translate_action(a)
                # next_state, reward, done = env.step(action_name)
                # print("\nStepping:")
                # print(f"  position: {o.position}")
                # print(f"  action: {a}")
                next_state, reward, done = env.step(a)
                # wandb.log({"reward": reward})
                # wandb.log({"episode": episode})
                # wandb.log({"step": t})
                # wandb.log({"action_name": env.translate_action(a)})
                # wandb.log({"next_state_position": next_state.position})
                # wandb.log({"next_state_branch": next_state.branch})
                # wandb.log({"current_state_position": env.current_state.position})
                # wandb.log({"current_state_branch": env.current_state.branch})
                # wandb.log({"victims_saved": env.rescue_number})
                # wandb.log({"victims_left": env.number_of_victims - env.rescue_number})
                # wandb.log({"hazards_hit": env.hazard_number})

                all_rewards.append(reward)
                # all_branches.append(env.current_state.branch)

                cumulative_returns[episode, t] = reward

                assert reward is not None and a is not None

                inputs = torch.cat(
                    (
                        o.to_tensor(dtype=torch.float64).view(1, -1).to(args.device),
                        torch.tensor(a, dtype=torch.float64)
                        .view(1, -1)
                        .to(args.device),
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

                # qs.masked_fill()
                a = torch.argmax(qs, -1)
                h_a.append(a.clone().detach().view(1, -1).to(args.device))
                h_r.append(
                    torch.tensor(reward, dtype=torch.float64)
                    .view(1, -1)
                    .to(args.device)
                )
                h_o.append(
                    env.current_state.to_tensor(dtype=torch.float64)
                    .view(1, -1)
                    .to(args.device)
                )
                cumulative_return += reward
                cumulative_returns[episode, t] = cumulative_return
                # wandb.log({"cumulative_return": cumulative_return})
                # wandb.log({"action": a})
                if t > 0:
                    # wandb.log({"epistemic_loss": losses_epistemic[-1]})
                x_vals = [1, 0]
                y_vals = [1, 0]

                cumulative_returns[episode, t] = cumulative_return
                t += 1
                if done:
                    cumulative_return_for_ep = cumulative_returns[episode, -1]
                    indexed_cumulative_return = [episode, cumulative_return_for_ep]
                    end_of_episode_cumulative_returns.append(indexed_cumulative_return)
                    # wandb.log({"cumulative_return_for_ep": cumulative_return_for_ep})
                    # table = wandb.Table(
                    #     data=end_of_episode_cumulative_returns,
                    #     columns=["episode", "cumulative return for episode"],
                    # )
                    # wandb.log(
                    #     {
                    #         "custom_plot": wandb.plot.line(
                    #             table,
                    #             "Episodes",
                    #             "Cumulative Returns for Episode",
                    #             title="Cumulative Returns for Episode",
                    #         )
                    #     }
                    # )
                    episode += 1
                    q_net.reset_net()
                    hidden = q_net.init_hidden()
                    cumulative_return = 0
                    t = 0
                    env.reset()
                    o = env.current_state
                    h_o = []
                    h_r = h_r[-1]
                    h_a = h_a[-1]

                    # Break out when done
                    break

        # av_cumulative_reward_list.append(np.median(cumulative_returns, axis=0))

    # wandb.finish()

    plot_ben_metrics(
        losses_q=all_q_losses,
        losses_epistemic=all_epistemic_losses,
        rewards=all_rewards,
    )


def plot_ben_metrics(
    losses_q: List[float],
    losses_epistemic: List[float],
    rewards: List[float],
    # branches: List[int],
    window_size: int = 50,
):
    """
    Plot key metrics from BEN training including MSBBE loss, ELBO loss,
    rewards, and branch choices.

    Args:
        losses_q: List of MSBBE (Q-network) losses
        losses_epistemic: List of ELBO (epistemic) losses
        rewards: List of rewards received
        branches: List of branches chosen (-1=pre-split, 0=left, 1=right)
        window_size: Size of window for running average
    """
    # Convert inputs to numpy arrays
    losses_q = np.array(losses_q)
    losses_epistemic = np.array(losses_epistemic)
    rewards = np.array(rewards)
    # branches = np.array(branches)

    # Calculate running averages
    def running_average(data: np.ndarray, window: int) -> np.ndarray:
        return np.convolve(data, np.ones(window) / window, mode="valid")

    ra_losses_q = running_average(losses_q, window_size)
    ra_losses_epistemic = running_average(losses_epistemic, window_size)
    ra_rewards = running_average(rewards, window_size)

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("BEN Training Metrics", fontsize=16)

    # Plot MSBBE Loss
    ax1.plot(losses_q, alpha=0.3, color="blue", label="Raw")
    ax1.plot(
        np.arange(window_size - 1, len(losses_q)),
        ra_losses_q,
        color="blue",
        label=f"Running Avg (window={window_size})",
    )
    ax1.set_title("MSBBE Loss Over Time")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot ELBO Loss
    ax2.plot(losses_epistemic, alpha=0.3, color="red", label="Raw")
    ax2.plot(
        np.arange(window_size - 1, len(losses_epistemic)),
        ra_losses_epistemic,
        color="red",
        label=f"Running Avg (window={window_size})",
    )
    ax2.set_title("ELBO Loss Over Time")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot Rewards
    ax3.plot(rewards, alpha=0.3, color="green", label="Raw")
    ax3.plot(
        np.arange(window_size - 1, len(rewards)),
        ra_rewards,
        color="green",
        label=f"Running Avg (window={window_size})",
    )
    ax3.set_title("Rewards Over Time")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Reward")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot Branch Choices
    # steps = np.arange(len(branches))
    # scatter = ax4.scatter(steps, branches, c=branches, cmap="viridis", alpha=0.6, s=20)
    # ax4.set_title("Branch Choices Over Time")
    # ax4.set_xlabel("Step")
    # ax4.set_ylabel("Branch (-1=pre-split, 0=left, 1=right)")
    # ax4.set_yticks([-1, 0, 1])
    # ax4.grid(True, alpha=0.3)
    # plt.colorbar(scatter, ax=ax4)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    #############################

    # Add this at the end of your training loop:
    # """
    # # Collect metrics during training
    # all_q_losses = []
    # all_epistemic_losses = []
    # all_rewards = []
    # all_branches = []
    #
    # # In training loop:
    # all_q_losses.extend(losses_q)
    # all_epistemic_losses.extend(losses_epistemic)
    # all_rewards.append(reward)
    # all_branches.append(env.current_state.branch)
    #
    # # After training:
    # plot_ben_metrics(all_q_losses,
    #                 all_epistemic_losses,
    #                 all_rewards,
    #                 all_branches)
    # """

    ##########################

    #     timesteps = np.array(range(max_steps_per_episode))
    #     spaghetti_cumulative_returns.append(cumulative_returns)
    #     twentyfifth_percentile = np.percentile(cumulative_returns, 30, axis=0)
    #     seventyfifth_percentile = np.percentile(cumulative_returns, 70, axis=0)
    #     twentyfifth_percentile_list.append(twentyfifth_percentile)
    #     seventyfifth_percentile_list.append(seventyfifth_percentile)
    #     # print(av_cumulative_reward_list)
    #     # clist = matplotlib.rcParams["axes.prop_cycle"]
    #     # plt.rcParams.update({"font.size": 16})
    #
    # # Set the figure size and font parameters first
    # plt.rcParams.update({"font.size": 16})
    # colors = ["#ff4500", "#ff0000", "#dc143c", "#aa0000", "#8b0000"]
    # baseline_colors = ["#003366", "#66ccff"]
    #
    # # Create figure with adjusted size and margins
    # fig, ax = plt.subplots(
    #     figsize=(12, 8)
    # )  # Changed from (10, 10) to give more horizontal space
    # plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)  # Adjust margins
    #
    # # ax.fill_between(
    # #     timesteps,
    # # for i, n_msbbe_steps in enumerate(n_msbbe_steps_multiple):
    # #     if i < len(n_msbbe_steps_multiple) - 1:
    # #         ax.plot(
    # #             timesteps,
    # #             av_cumulative_reward_list,
    # #             ":",
    # #             label=f"{n_msbbe_steps}",
    # #             linewidth=3,
    # #             color=colors[-i - 1],
    # #         )
    # #         ax.fill_between(
    # #             timesteps,
    # #             twentyfifth_percentile_list[i],
    # #             seventyfifth_percentile_list[i],
    # #             alpha=0.2,
    # #             interpolate=False,
    # #             color=colors[-i - 1],
    # #         )
    # #     else:
    # #         ax.plot(
    # #             timesteps,
    # #             av_cumulative_reward_list[-1],
    # #             label=f"{n_msbbe_steps}",
    # #             linewidth=3,
    # #             color=colors[0],
    # #         )
    # #         ax.fill_between(
    # #             timesteps,
    # #             twentyfifth_percentile_list[-1],
    # #             seventyfifth_percentile_list[-1],
    # #             alpha=0.2,
    # #             interpolate=False,
    # #             color=colors[0],
    # #         )
    # #
    # #         ax.set_xlim(0, 10)
    # #         ax.tick_params(axis="both", which="major", labelsize=16, length=6, width=2)
    # #
    # #         ax.legend(loc="best", title="MSBBE Iterations", prop={"size": 10})
    # #         ax.set_xlabel("Timesteps", fontsize=16)
    # #         ax.set_ylabel("Cumulative Return", fontsize=16)
    # #         ax.set_title("Median, 30th, 70th Percentile Cumulative Return vs. Timestep")
    # # # Blues in hexadecimal 6
    # #
    # # # for i, policy in enumerate(policy_list):
    # # #                    interpolate = True)
    # #
    # # #
    # # # fig.suptitle(f"Median, 30th, 70th Percentile Cumulative Return vs. Timesteps over {N_episodes} Episodes",
    # #
    # # # Set the font weight to bold
    # #
    # # # Update the legend
    # # fig.show()
    # # fig.savefig("msbbe_steps.svg")
    # # fig.savefig("msbbe_steps.pdf", format="pdf")
    # # plt.show()
    # #
    # # wandb.finish()
    #
    # # Plot data
    # for i, n_msbbe_steps in enumerate(n_msbbe_steps_multiple):
    #     if i < len(n_msbbe_steps_multiple) - 1:
    #         ax.plot(
    #             timesteps,
    #             av_cumulative_reward_list,
    #             ":",
    #             label=f"{n_msbbe_steps}",
    #             linewidth=3,
    #             color=colors[-i - 1],
    #         )
    #         ax.fill_between(
    #             timesteps,
    #             twentyfifth_percentile_list[i],
    #             seventyfifth_percentile_list[i],
    #             alpha=0.2,
    #             interpolate=False,
    #             color=colors[-i - 1],
    #         )
    #     else:
    #         ax.plot(
    #             timesteps,
    #             av_cumulative_reward_list[-1],
    #             label=f"{n_msbbe_steps}",
    #             linewidth=3,
    #             color=colors[0],
    #         )
    #         ax.fill_between(
    #             timesteps,
    #             twentyfifth_percentile_list[-1],
    #             seventyfifth_percentile_list[-1],
    #             alpha=0.2,
    #             interpolate=False,
    #             color=colors[0],
    #         )
    #
    # # Set axis properties
    # ax.set_xlim(0, 10)
    # ax.tick_params(axis="both", which="major", labelsize=16, length=6, width=2)
    #
    # # Add legend and labels with adjusted positions
    # ax.legend(
    #     loc="upper right",
    #     title="MSBBE Iterations",
    #     prop={"size": 12},
    #     bbox_to_anchor=(0.98, 0.98),
    # )
    # ax.set_xlabel("Timesteps", fontsize=16, labelpad=10)
    # ax.set_ylabel("Cumulative Return", fontsize=16, labelpad=10)
    # ax.set_title("Median, 30th, 70th Percentile Cumulative Return vs. Timestep", pad=20)
    #
    # # Save and show
    # plt.savefig("msbbe_steps.svg", bbox_inches="tight", dpi=300)
    # plt.savefig("msbbe_steps.pdf", format="pdf", bbox_inches="tight", dpi=300)
    # plt.show()
