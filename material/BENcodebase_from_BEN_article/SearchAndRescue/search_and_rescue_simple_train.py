# Core modules

# 3rd party modules
import argparse
from distutils.util import strtobool

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
from torch import nn, optim

import AleatoricNetwork as al
import EpistemicNetwork as ep
import BayesModel
from ConditionerMLP import ConditionerMLP
from EmpiricalTarget import EmpiricalTarget
from QBayesNet import QBayesNetwork
from SearchAndRescue.search_and_rescue import SearchRescueEnv
from prior_initialization import prior_initialization
from posterior_updating import posterior_update
import os

torch.set_grad_enabled(True)

# Needed for stability reasons for abs flow
torch.set_default_tensor_type(torch.DoubleTensor)


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
    parser.add_argument("--lr_phi", type=float, default=1e-4, help="learning rate for epistemic")
    parser.add_argument(
        "--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment"
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
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="rl-world", help="the entity (team) of wandb's project")
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
import wandb

av_cum_reward_list = []
twentyfifth_percentile_list = []
seventyfifth_percentile_list = []
spaghetti_cum_returns = []


class QMyopicNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 120)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(84, output_size)

    def forward(self, otp1):
        cc = th.Tensor([[otp1]], device=self.fc1.weight.device)
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
N_episodes = 500
memory = 10

if __name__ == "__main__":
    args = parse_args()
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity)
    wandb.config.update(args, allow_val_change=True)
    gamma = args.gamma
    n_msbbe_steps_multiple = [10]  # 10, #50]# 2, 3, 10, 20]  # 50
    for n_msbbe_steps in n_msbbe_steps_multiple:
        # Additional config params
        cum_return = 0
        #args.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        # Initialise Env
        gridwidth = 7
        time_limit = 5 * gridwidth **2 
        env = SearchRescueEnv(gridwidth=gridwidth, number_of_hazards=8, number_of_victims=4, max_steps_per_episode=time_limit)
        env.reset()
        (aleatoric_flow, num_params_aleatoric, base_dist_al) = al.construct_aleatoric_flow(6, 2, device=args.device)
        epistemic_flow, prior_ep = ep.construct_epsitemic_flow(num_params_aleatoric, 2, aleatoric_flow, device=args.device)
        q_net = QBayesNetwork(input_size=np.prod(env.state.shape) + 2, output_size=env.action_space.n).to(args.device)
        q_net = q_net.to(args.device)
        hidden_size = q_net.rnn.hidden_size
        q_net.init_hidden()
        q_net.init_params()
        conditioner_net = ConditionerMLP(
            num_params_aleatoric + 1 + hidden_size, num_params_aleatoric, num_params_aleatoric, device=args.device
        ).to(args.device)
        target_dist = EmpiricalTarget(q_net, env)

        model = BayesModel.BayesModel(conditioner_net, epistemic_flow, aleatoric_flow, target_dist)

        # Initialise Q-Network
        optimiser_q = optim.Adam(q_net.parameters(), lr=args.lr_q)
        optimiser_phi = optim.Adam(model.parameters(), lr=args.lr_phi)

        # Initialise Policy

        # Initialise environment
        env.reset()
        o = env.state
        o_init = o.clone()
        # initialise history
        h_o = []
        h_r = []
        h_a = []
        # Train
        optimiser_q.zero_grad()
        optimiser_phi.zero_grad()
        episode = 0
        t = 0
        hidden = q_net.init_hidden()

        cum_returns = np.zeros((N_episodes, env.max_steps_per_episode))

        action_init = torch.tensor(env.action_space.sample(), dtype=torch.float64).view(1, -1).to(args.device)
        reward_init = torch.tensor(0.0).view(1, -1).to(args.device)
        inputs = torch.cat((torch.tensor(o).view(1, -1).to(args.device), action_init, reward_init), dim=1)
        q_val, _ = q_net(inputs, hidden)
        if args.prior_initialization:
            loss_q, q_net, optimiser_q = prior_initialization(
                env,
                prior_ep,
                base_dist_al,
                torch.tensor(action_init, dtype=torch.float64).view(1, -1).to(args.device),
                torch.tensor(reward_init, dtype=torch.float64).view(1, -1).to(args.device),
                torch.tensor(o, dtype=torch.float64).view(1, -1).to(args.device),
                q_net,
                model,
                1000,
                hidden_size,
                gamma=args.gamma,
                device=args.device,
                args=args,
            )
            wandb.log({"prior_loss_q": loss_q})
        end_of_episode_cum_returns = []
        while episode < N_episodes:
            # commented out prior adjustment of log scales.... for epistemic and aleatoric as test
            model = BayesModel.BayesModel(conditioner_net, epistemic_flow, aleatoric_flow, target_dist).to(args.device)
            env.reset()
            o = env.state
            h_o = []
            h_r = []
            h_a = []
            o = torch.tensor(o, dtype=torch.float64).view(1, -1).to(args.device)
            h_o.append(o)
            h_a.append(action_init)
            h_r.append(reward_init)
            t = 0
            current_loss = []
            hidden = q_net.init_hidden()
            cum_returns = np.zeros((N_episodes, env.max_steps_per_episode))
            action_init = 3
            reward_init = torch.tensor(0.0, dtype=torch.float64).view(1, -1).to(args.device)
            a = action_init
            
            for t in range(env.max_steps_per_episode):
                o = env.state
                #
                q_net, model, losses_q, losses_epistemic = posterior_update(
                    hidden,
                    t,
                    model,
                    q_net,
                    h_o,
                    h_a,
                    h_r,
                    action_init,
                    reward_init,
                    args,
                    env,
                    o,
                    N_update=n_msbbe_steps,
                    memory=memory,
                    device=args.device,
                )
                wandb.log({"memory":memory, "n_msbbe_steps":n_msbbe_steps})
                current_loss.append(losses_q[-1])
                wandb.log({"loss_q": losses_q[-1]})
                if isinstance(a, th.Tensor):
                    a = a.item()
                action_name = env.translate_action(a)
                next_state, reward, done = env.step(action_name)
                wandb.log({"reward": reward})
                wandb.log({"episode": episode})
                wandb.log({"step": t})
                wandb.log({"action_name": action_name})
                wandb.log({"next_state": next_state})
                wandb.log({"current_state": env.state})
                wandb.log({"victims_saved": env.rescue_number})
                wandb.log({"victims_left": env.number_of_victims - env.rescue_number})
                wandb.log({"hazards_hit": env.hazard_number})

                cum_returns[episode, t] = reward
                assert reward is not None and a is not None
                inputs = torch.cat(
                    (
                        torch.tensor(o, dtype=torch.float64).view(1, -1).to(args.device),
                        torch.tensor(a, dtype=torch.float64).view(1, -1).to(args.device),
                        torch.tensor(reward, dtype=torch.float64).view(1, -1).to(args.device),
                    ),
                    dim=1,
                )
                qs, hidden = q_net(inputs, hidden)
                a = th.argmax(qs, -1)
                h_a.append(torch.tensor(a, dtype=torch.float64).view(1, -1).to(args.device))
                h_r.append(torch.tensor(reward, dtype=torch.float64).view(1, -1).to(args.device))
                h_o.append(torch.tensor(env.state, dtype=torch.float64).view(1, -1).to(args.device))
                cum_return += reward
                cum_returns[episode, t] = cum_return
                wandb.log({"cum_return": cum_return})
                wandb.log({"action": a})
                if t > 0:
                    wandb.log({"epistemic_loss": losses_epistemic[-1]})
                x_vals = [1, 0]
                y_vals = [1, 0]

                cum_returns[episode, t] = cum_return
                t += 1
                if done:
                    cum_return_for_ep = cum_returns[episode, -1]
                    indexed_cum_return = [episode, cum_return_for_ep]
                    end_of_episode_cum_returns.append(indexed_cum_return)
                    wandb.log({"cum_return_for_ep": cum_return_for_ep})
                    table = wandb.Table(
                        data=end_of_episode_cum_returns, columns=["episode", "cumulative return for episode"]
                    )
                    wandb.log(
                        {
                            "custom_plot": wandb.plot.line(
                                table, "Episodes", "Cumulative Returns for Episode", title="Cumulative Returns for Episode"
                            )
                        }
                    )
                    episode += 1
                    q_net.reset_net()
                    hidden = q_net.init_hidden()
                    cum_return = 0
                    t = 0
                    env.reset()
                    o = env.state
                    h_o = []
                    h_r = h_r[-1]
                    h_a = h_a[-1]

        timesteps = np.array(range(env.max_steps_per_episode))
        av_cum_returns = np.median(cum_returns, axis=0)
        av_cum_reward_list.append(av_cum_returns)
        spaghetti_cum_returns.append(cum_returns)
        twentyfifth_percentile = np.percentile(cum_returns, 30, axis=0)
        seventyfifth_percentile = np.percentile(cum_returns, 70, axis=0)
        twentyfifth_percentile_list.append(twentyfifth_percentile)
        seventyfifth_percentile_list.append(seventyfifth_percentile)
        print(av_cum_reward_list)
        clist = matplotlib.rcParams["axes.prop_cycle"]
        plt.rcParams.update({"font.size": 16})


    colors = ["#ff4500", "#ff0000", "#dc143c", "#aa0000", "#8b0000"]
    baseline_colors = ["#003366", "#66ccff"]
    fig, ax = plt.subplots(figsize = (10, 10))
    # ax.fill_between(
    #     timesteps,
    for i, n_msbbe_steps in enumerate(n_msbbe_steps_multiple):
        if i < len(n_msbbe_steps_multiple) - 1:
            ax.plot(timesteps, av_cum_reward_list, ":", label=f"{n_msbbe_steps}", linewidth=3, color=colors[-i - 1])
            ax.fill_between(
                timesteps,
                twentyfifth_percentile_list[i],
                seventyfifth_percentile_list[i],
                alpha=0.2,
                interpolate=False,
                color=colors[-i - 1],
            )
        else:
            ax.plot(timesteps, av_cum_reward_list[-1], label=f"{n_msbbe_steps}", linewidth=3, color=colors[0])
            ax.fill_between(
                timesteps,
                twentyfifth_percentile_list[-1],
                seventyfifth_percentile_list[-1],
                alpha=0.2,
                interpolate=False,
                color=colors[0],
            )

            ax.set_xlim(0, 10)
            ax.tick_params(axis="both", which="major", labelsize=16, length=6, width=2)

            ax.legend(loc="best", title="MSBBE Iterations", prop={"size": 10})
            ax.set_xlabel("Timesteps", fontsize=16)
            ax.set_ylabel("Cumulative Return", fontsize=16)
            ax.set_title("Median, 30th, 70th Percentile Cumulative Return vs. Timestep")
    # Blues in hexadecimal 6

    # for i, policy in enumerate(policy_list):
    #                    interpolate = True)

    #
    # fig.suptitle(f"Median, 30th, 70th Percentile Cumulative Return vs. Timesteps over {N_episodes} Episodes",

    # Set the font weight to bold

    # Update the legend
    fig.show()
    fig.savefig("msbbe_steps.svg")
    fig.savefig("msbbe_steps.pdf", format="pdf")
    plt.show()

    wandb.finish()
