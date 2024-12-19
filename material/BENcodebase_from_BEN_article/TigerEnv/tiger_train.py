# Core modules
import logging.config

# 3rd party modules
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch as th
from torch import nn
from torch import optim
import matplotlib
from policies import BayesOptimalPolicy, ContextualOptimalPolicy

matplotlib.rcParams.update({"figure.autolayout": True})
matplotlib.rcParams.update({"figure.figsize": (10, 8)})
matplotlib.rcParams.update({"figure.dpi": 1200})
plt.figure(dpi=1200)
# Import classes
fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)

av_cum_reward_list = []
twentyfifth_percentile_list = []
seventyfifth_percentile_list = []
spaghetti_cum_returns = []


class TigerEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, reward_tiger=-100, reward_gold=10, reward_listen=-1, state_accuracy=0.9, max_steps_per_episode=100
    ):
        """
        OpenAI Gym environment for the partially stateervable Tiger game.

        Parameters
        ----------
        reward_tiger : numeric
            Reward for opening the door with the tiger.
        reward_gold : numeric
            Reward for opening the door with the gold.
        reward_listen : numeric
            Reward for taking the listen action.
        state_accuracy : numeric
            Number b/w 0 and 1. The accuracy of the growl. I.e. state_accuracy of
            1 means that a GROWL_LEFT implies TIGER_LEFT 100% of the time.
        max_steps_per_episode : int, default=100
            Maximum allowed steps per episode. This will define how long an
            episode lasts, since the Tiger game does not end otherwise.

        Attributes
        ----------
        curr_episode : int
            Current episode as a count.
        action_episode_memory : list<int>
            History of actions taken in episode.
        state_episode_memory : list<int>
            History of states stateerved in episode.
        reward_episode_memory : list<int>
            History of rewards stateerved in episode.
        curr_step : int
            Current timestep in episode, as a count.
        action_space : gym.spaces.Discrete
            Action space.
        state_space : gym.spaces.Discrete
            state space.
        """
        self.reward_tiger = reward_tiger
        self.reward_gold = reward_gold
        self.reward_listen = reward_listen
        self.state_accuracy = state_accuracy
        self.max_steps_per_episode = max_steps_per_episode

        self.__version__ = "0.0.2"
        logging.info("TigerEnv - Version {}".format(self.__version__))

        self.curr_episode = -1  # Set to -1 b/c reset() adds 1 to episode
        self.action_episode_memory = []
        self.state_episode_memory = []
        self.reward_episode_memory = []

        self.curr_step = 0

        self.reset()

        # Define what the agent can do: LISTEN, OPEN_LEFT, OPEN_RIGHT
        self.action_space = spaces.Discrete(3)

        # Define agent's states: START, HEAR_GROWL_LEFT, HEAR_GROWL_RIGHT
        self.state = spaces.MultiDiscrete([3])

    @staticmethod
    def str_to_action_idx(_str):
        str_to_action_idx_dct = {"LISTEN": 0, "OPEN_LEFT": 1, "OPEN_RIGHT": 2}
        return str_to_action_idx_dct[_str]

    @staticmethod
    def action_idx_to_str(_str):
        action_idx_to_str_dct = {0: "LISTEN", 1: "OPEN_LEFT", 2: "OPEN_RIGHT"}
        return action_idx_to_str_dct[_str]

    @staticmethod
    def str_to_state_idx(_str):
        str_to_state_idx_dct = {"START": 0, "GROWL_LEFT": 1, "GROWL_RIGHT": 2}
        return str_to_state_idx_dct[_str]

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        state, reward, episode_over, info : tuple
            state : int
                An int in {0,1,2} defining state the agent transitions to
            reward : float
                Amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                Whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : dict
                Diagnostic information useful for debugging. It can sometimes
                be useful for learning (for example, it might contain the raw
                probabilities behind the environment's last state change).
                However, official evaluations of your agent are not allowed to
                use this for learning.
        """
        if not isinstance(action, str):
            action = env.action_idx_to_str(a[0].item())
        else:
            action = a
        done = self.curr_step >= self.max_steps_per_episode
        if done:
            self.reset()
        self.curr_step += 1
        # Recompute done since action may have modified it
        done = self.curr_step >= self.max_steps_per_episode
        self.action_episode_memory[self.curr_episode].append(action)
        reward = self._get_reward()
        state = self._get_state()
        # This has to come after adding the action memory
        self.state_episode_memory[self.curr_episode].append(state)
        self.reward_episode_memory[self.curr_episode].append(reward)

        return state, reward, done  # TODO: Do we need truncations?

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment and returns an initial state.

        Returns
        -------
        object
            The initial state of the space.
        """
        self.curr_step = 0
        self.curr_episode += 1
        self.tiger_left = np.random.randint(0, 2)
        self.tiger_right = 1 - self.tiger_left
        initial_state = "START"
        self.action_episode_memory.append([-1])  # Needs to be offset by 1
        self.state_episode_memory.append([initial_state])
        self.reward_episode_memory.append([0])  # Needs to be offset by 1
        return initial_state, {}

    def render(self, mode="human"):
        return

    def close(self):
        pass

    def _get_reward(self):
        """
        Obtain the reward for the current state of the environment.

        Returns
        -------
        float
            Reward.
        """
        last_action = self.action_episode_memory[self.curr_episode][-1]
        if last_action == "OPEN_LEFT":
            if self.tiger_left:
                return self.reward_tiger
            else:
                return self.reward_gold
        if last_action == "OPEN_RIGHT":
            if self.tiger_right:
                return self.reward_tiger
            else:
                return self.reward_gold
        else:
            return self.reward_listen

    def _get_state(self):
        """
        Obtain the next state for the current action.modules
        Returns
        -------
        list
            state.
        """
        last_action = self.action_episode_memory[self.curr_episode][-1]
        if last_action == "LISTEN":
            # Returns accurate state with probabilty state_accuracy
            sample = np.random.rand()
            if self.tiger_left:
                if sample < self.state_accuracy:
                    return "GROWL_LEFT"
                else:
                    return "GROWL_RIGHT"
            else:
                if sample < self.state_accuracy:
                    return "GROWL_RIGHT"
                else:
                    return "GROWL_LEFT"
        else:
            # Returns agent to start state if door is opened
            return "START"


class QBayesNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.rnn = nn.GRUCell(32, 32)
        self.fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(32, output_size)

    def init_hidden(self):
        # make hidden states on same device as model, and same batch size as input
        return self.fc1.weight.new(1, 32).zero_()

    def forward(self, obs, action, rewards, hidden_state):
        x = torch.cat(
            [
                torch.tensor(obs, dtype=torch.float32, device=self.fc1.weight.device).view(1, -1),
                torch.tensor(action, dtype=torch.float32, device=self.fc1.weight.device).view(1, -1),
                torch.tensor(rewards, dtype=torch.float32, device=self.fc1.weight.device).view(1, -1),
            ],
            dim=1,
        )
        x1 = self.fc1(x)
        x2 = self.relu1(x1)
        h_in = hidden_state.reshape(-1, 32)
        h_out = self.rnn(x2, h_in)
        x3 = self.fc2(h_out)
        x3 = self.relu2(x3)
        x3 = self.out(x3)
        return x3, h_out

    def reset_net(self):
        self.fc1.reset_parameters()
        self.rnn.reset_parameters()
        self.fc2.reset_parameters()


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


class BenPolicy:
    def __init__(self, reward_gold, reward_tiger, reward_listen, state_accuracy):
        # Samples initial policy from a uniform prior at random then learns about the tiger so take optimal policy after
        self.posterior_left = 0.5  # Initialise as prior
        self.reward_gold = reward_gold
        self.reward_tiger = reward_tiger
        self.reward_listen = reward_listen
        self.N_right = 0
        self.N_left = 0
        self.t = 0
        self.door_opened = False
        self.p = float(state_accuracy)

    def update_posterior(self, state, reward, action):
        self.t += 1
        if action != "LISTEN":
            self.door_opened = True
        if state == "GROWL_LEFT":
            self.N_left += 1
            self.update_posterior_left()
        if state == "GROWL_RIGHT":
            self.N_right += 1
            self.update_posterior_left()
        if self.door_opened:
            if reward == self.reward_gold:
                if action == "OPEN_LEFT":
                    self.posterior_left = 0.0
                else:
                    self.posterior_left = 1.0
            if reward == self.reward_tiger:
                if action == "OPEN_LEFT":
                    self.posterior_left = 1.0
                else:
                    self.posterior_left = 0.0

    def update_posterior_left(self):
        num = (self.p**self.N_left) * ((1 - self.p) ** self.N_right)
        evidence = (self.p**self.N_left) * ((1 - self.p) ** self.N_right) + (self.p**self.N_right) * (
            (1 - self.p) ** self.N_left
        )
        self.posterior_left = num / evidence

    def get_posterior_left(self):
        return self.posterior_left

    def take_action(self, Q_bayes):
        return torch.argmax(Q_bayes)

    def return_Bayes_transition_left(self):
        # Returns the Bayesian probability of hearing growl left given taking action listen
        return self.posterior_left * self.p + (1 - self.posterior_left) * (1 - self.p)

    def return_Bayes_reward(self, action):
        if action == "LISTEN":
            return self.reward_listen
        if action == "OPEN_LEFT":
            return self.posterior_left * self.reward_tiger + (1 - self.posterior_left) * self.reward_gold
        if action == "OPEN_RIGHT":
            return self.posterior_left * self.reward_gold + (1 - self.posterior_left) * self.reward_tiger

    def reset(self):
        self.N_right = 0
        self.N_left = 0
        self.update_posterior_left()
        self.door_opened = False
        self.t = 0


obs_prob = 0.85
horizon = 11
learning_rate = 0.02
N_episodes = 50
gamma = 0.9

if __name__ == "__main__":
    n_msbbe_steps_multiple = [1, 2, 5, 10, 20]  # 10, #50]# 2, 3, 10, 20]  # 50
    for n_msbbe_steps in n_msbbe_steps_multiple:
        # Additional config params
        cum_return = 0

        # Initialise Env
        env = TigerEnv(reward_tiger=-500, reward_gold=10, max_steps_per_episode=horizon, state_accuracy=obs_prob)

        policy = BenPolicy(env.reward_gold, env.reward_tiger, env.reward_listen, env.state_accuracy)

        # Initialise Q-Network
        input_size = 1 + 2
        output_size = 3
        q_net = QBayesNetwork(input_size=input_size, output_size=output_size)

        # Initialise Optimiser
        optimiser = optim.Adam(q_net.parameters(), lr=learning_rate)

        # Initialise Policy

        # Initialise environment
        o = env.reset()[0]

        # initialise history
        h_o = []
        h_r = [-1]
        h_a = [-1]
        # Train
        print("Training with n_msbbe_steps = {}".format(n_msbbe_steps))
        env.reset()
        optimiser.zero_grad()
        episode = 0
        t = 0
        hidden = q_net.init_hidden()

        cum_returns = np.zeros((N_episodes, env.max_steps_per_episode))
        while episode < N_episodes:
            h_o.append(env.str_to_state_idx(o))

            # minimise MSBBE
            # Predictive for action 'OPEN_RIGHT'
            r_right = policy.return_Bayes_reward("OPEN_RIGHT")
            b_right = th.tensor(r_right + (horizon - policy.t - 1) * policy.reward_gold)
            # Predictive for action 'OPEN_LEFT'
            r_left = policy.return_Bayes_reward("OPEN_LEFT")
            b_left = th.tensor(r_left + (horizon - policy.t - 1) * policy.reward_gold)

            for j in range(n_msbbe_steps):
                # Traverse Network:
                hidden = q_net.init_hidden()
                for i in range(t - 1):
                    _, hidden = q_net(h_o[i], h_a[i], h_r[i], hidden)
                qs, hidden = q_net(h_o[-1], h_a[-1], h_r[-1], hidden)

                p_growl_left = policy.return_Bayes_transition_left()
                r_listen = policy.return_Bayes_reward("LISTEN")
                qs_left, _ = q_net(
                    env.str_to_state_idx("GROWL_LEFT"), env.str_to_action_idx("LISTEN"), r_listen, hidden
                )
                qs_right, _ = q_net(
                    env.str_to_state_idx("GROWL_RIGHT"), env.str_to_action_idx("LISTEN"), r_listen, hidden
                )
                b_listen = (
                    r_listen
                    + gamma * p_growl_left * th.max(qs_left, -1)[0]
                    + (1 - p_growl_left) * th.max(qs_right, -1)[0]
                )
                bs = th.tensor([b_listen, b_left, b_right])

                loss = ((bs - qs) ** 2).mean()
                """
                    wandb.log({"q_loss": loss.item(),
                    "p_growl_left": p_growl_left,
                    "r_listen": r_listen,
                    "r_right": r_right,
                    "r_left": r_left})
                    """
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            qs, hidden = q_net(h_o[-1], -1, -1, hidden)
            a = th.argmax(qs, -1)
            h_a.append(a)
            o, r, d = env.step(a)
            h_r.append(r)
            cum_return += r
            cum_returns[episode, t] = cum_return
            policy.update_posterior(o, r, env.action_idx_to_str(a[0].item()))
            t += 1
            if d:
                episode += 1
                q_net.reset_net()
                hidden = q_net.init_hidden()
                cum_return = 0
                t = 0
                policy.reset()
                o = env.reset()[0]
                h_o = []
                h_r = [-1]
                h_a = [-1]
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
    ### Multiple policies

    policy_ben = BenPolicy(env.reward_gold, env.reward_tiger, env.reward_listen, env.state_accuracy)
    policy_bayes = BayesOptimalPolicy(
        reward_gold=env.reward_gold,
        reward_tiger=env.reward_tiger,
        reward_listen=env.reward_listen,
        state_accuracy=env.state_accuracy,
        horizon=env.max_steps_per_episode,
    )
    policy_context = ContextualOptimalPolicy(reward_gold=env.reward_gold)
    n_msbbe_steps = 5
    policy_list = [policy_bayes, policy_ben, policy_context]
    av_cum_reward_list_policy = []

    twentyfifth_percentile_list_policy = []

    seventyfifth_percentile_list_policy = []
    for policy in policy_list:
        policy.reset()
        # Initialise Optimiser
        optimiser = optim.Adam(q_net.parameters(), lr=learning_rate)

        # Initialise Policy

        # Initialise environment
        o = env.reset()[0]

        # initialise history
        h_o = []
        h_r = [-1]
        h_a = [-1]
        # Train
        print("Training with n_msbbe_steps = {}".format(n_msbbe_steps))
        env.reset()
        optimiser.zero_grad()
        episode = 0
        t = 0
        hidden = q_net.init_hidden()
        cum_returns = np.zeros((N_episodes, env.max_steps_per_episode))
        if policy == policy_bayes:
            i = 0
            while episode < N_episodes:
                a = policy.take_action()
                s, r, d = env.step(a)
                cum_return += r
                cum_returns[episode, i] = cum_return
                policy.update_posterior(s, r)
                i += 1
                if d:
                    episode += 1
                    cum_return = 0
                    i = 0
                    policy.reset()
            timesteps = np.array(range(env.max_steps_per_episode))
            av_cum_returns = np.median(cum_returns, axis=0)
            av_cum_reward_list_policy.append(av_cum_returns)
            twentyfifth_percentile = np.percentile(cum_returns, 30, axis=0)
            seventyfifth_percentile = np.percentile(cum_returns, 70, axis=0)
            twentyfifth_percentile_list_policy.append(twentyfifth_percentile)
            seventyfifth_percentile_list_policy.append(seventyfifth_percentile)
        if policy == policy_context:
            s = env._get_state()[-1]
            episode = 0
            cum_returns = np.zeros((N_episodes, env.max_steps_per_episode))
            cum_return = 0
            i = 0
            while episode < N_episodes:
                a = policy_context.take_action()
                s, r, d = env.step(a)
                cum_return += r
                cum_returns[episode, i] = cum_return
                policy_context.update_posterior(s, r)
                i += 1
                if d:
                    episode += 1
                    cum_return = 0
                    i = 0
                    policy_context.reset()
            timesteps = np.array(range(env.max_steps_per_episode))
            av_cum_returns = np.median(cum_returns, axis=0)
            av_cum_reward_list_policy.append(av_cum_returns)
            twentyfifth_percentile = np.percentile(cum_returns, 30, axis=0)
            seventyfifth_percentile = np.percentile(cum_returns, 70, axis=0)
            twentyfifth_percentile_list_policy.append(twentyfifth_percentile)
            seventyfifth_percentile_list_policy.append(seventyfifth_percentile)
        if policy == policy_ben:
            optimiser = optim.Adam(q_net.parameters(), lr=learning_rate)
            # Initialize msbbe steps
            n_msbbe_steps_policy = 20
            # Initialise Policy

            # Initialise environment
            o = env.reset()[0]

            # initialise history
            h_o = []
            h_r = [-1]
            h_a = [-1]
            # Train
            print("Training with n_msbbe_steps = {}".format(n_msbbe_steps))
            env.reset()
            optimiser.zero_grad()
            episode = 0
            t = 0
            hidden = q_net.init_hidden()

            cum_returns = np.zeros((N_episodes, env.max_steps_per_episode))
            while episode < N_episodes:
                h_o.append(env.str_to_state_idx(o))

                # minimise MSBBE
                # Predictive for action 'OPEN_RIGHT'
                r_right = policy.return_Bayes_reward("OPEN_RIGHT")
                b_right = th.tensor(r_right + (horizon - policy.t - 1) * policy.reward_gold)
                # Predictive for action 'OPEN_LEFT'
                r_left = policy.return_Bayes_reward("OPEN_LEFT")
                b_left = th.tensor(r_left + (horizon - policy.t - 1) * policy.reward_gold)

                for j in range(n_msbbe_steps_policy):
                    # Traverse Network:
                    hidden = q_net.init_hidden()
                    for i in range(t - 1):
                        _, hidden = q_net(h_o[i], h_a[i], h_r[i], hidden)
                    qs, hidden = q_net(h_o[-1], h_a[-1], h_r[-1], hidden)

                    p_growl_left = policy.return_Bayes_transition_left()
                    r_listen = policy.return_Bayes_reward("LISTEN")
                    qs_left, _ = q_net(
                        env.str_to_state_idx("GROWL_LEFT"), env.str_to_action_idx("LISTEN"), r_listen, hidden
                    )
                    qs_right, _ = q_net(
                        env.str_to_state_idx("GROWL_RIGHT"), env.str_to_action_idx("LISTEN"), r_listen, hidden
                    )
                    b_listen = (
                        r_listen
                        + gamma * p_growl_left * th.max(qs_left, -1)[0]
                        + (1 - p_growl_left) * th.max(qs_right, -1)[0]
                    )
                    bs = th.tensor([b_listen, b_left, b_right])

                    loss = ((bs - qs) ** 2).mean()
                    """
                        wandb.log({"q_loss": loss.item(),
                        "p_growl_left": p_growl_left,
                        "r_listen": r_listen,
                        "r_right": r_right,
                        "r_left": r_left})
                        """
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                qs, hidden = q_net(h_o[-1], -1, -1, hidden)
                a = th.argmax(qs, -1)
                h_a.append(a)
                o, r, d = env.step(a)
                h_r.append(r)
                cum_return += r
                cum_returns[episode, t] = cum_return
                policy.update_posterior(o, r, env.action_idx_to_str(a[0].item()))
                t += 1
                if d:
                    episode += 1
                    q_net.reset_net()
                    hidden = q_net.init_hidden()
                    cum_return = 0
                    t = 0
                    policy.reset()
                    o = env.reset()[0]
                    h_o = []
                    h_r = [-1]
                    h_a = [-1]
            timesteps = np.array(range(env.max_steps_per_episode))
            av_cum_returns = np.median(cum_returns, axis=0)
            twentyfifth_percentile = np.percentile(cum_returns, 30, axis=0)
            seventyfifth_percentile = np.percentile(cum_returns, 70, axis=0)
            av_cum_reward_list_policy.append(av_cum_returns)
            twentyfifth_percentile_list_policy.append(twentyfifth_percentile)
            seventyfifth_percentile_list_policy.append(seventyfifth_percentile)
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ["#ff4500", "#ff0000", "#dc143c", "#aa0000", "#8b0000"]
    baseline_colors = ["#003366", "#66ccff"]
    ax.plot(timesteps, av_cum_reward_list_policy[2], label="Contextual Bayes", linewidth=3, color="cyan")
    ax.fill_between(
        timesteps,
        twentyfifth_percentile_list_policy[2],
        seventyfifth_percentile_list_policy[2],
        alpha=0.2,
        interpolate=False,
        color="cyan",
    )
    for i, n_msbbe_steps in enumerate(n_msbbe_steps_multiple):
        if i != len(n_msbbe_steps_multiple) - 1:
            ax.plot(timesteps, av_cum_reward_list[i], ":", label=f"{n_msbbe_steps}", linewidth=3, color=colors[-i - 1])
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
    ax.plot(timesteps, av_cum_reward_list_policy[0], label="Benchmark (Bayes Optimal)", linewidth=3, color="#003366")
    ax.fill_between(
        timesteps,
        twentyfifth_percentile_list_policy[0],
        seventyfifth_percentile_list_policy[0],
        alpha=0.2,
        interpolate=False,
        color="#003366",
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
