from tqdm import tqdm
from ConditionerMLP import ConditionerMLP
from QBayesNet import QBayesNetwork
import normflows as nf
import gym
from torch import Tensor
import torch
import BayesModel
from random import randrange


def msbbe(b1, b2, q):
    cons = (b1 - q).detach()
    loss = torch.abs(cons * (b2 - q))
    return loss


def prior_initialization(
    envs: gym.vector.SyncVectorEnv,
    prior_phi: nf.distributions.base.DiagGaussian,
    base_dist_al: nf.distributions.base.DiagGaussian,
    action_init: Tensor,
    reward_init: Tensor,
    s_0: Tensor,
    q_network: QBayesNetwork,
    model: BayesModel,
    N_pretrain: int = 1000,
    hist_length: int = 20,
    gamma=0.97,
    args=None,
    device="cuda:0",
):
    """
    :param envs:
    :param prior_phi:
    :param base_dist_al:
    :param s_0:
    :param q_network:
    :param model:
    :param N_pretrain:
    :return:  loss, q_network, optimizer_q
    """
    q_network = q_network.requires_grad_(True).to(device)
    h_0 = q_network.init_hidden()
    hidden_state = h_0
    loss_hist = []
    q_params = q_network.parameters()
    optimizer_q = torch.optim.Adam(q_params, lr=args.lr_q)  # initialize ω weights of q_net and fetch
    actions = []
    states = []
    rewards = []
    # DEBUG
    #
    s_0 = torch.tensor(s_0).view(1, -1).to(args.device)
    s = s_0.clone()
    a = action_init
    model.add_module("q_network", q_network)
    for n in tqdm(range(N_pretrain)):
        # Sample from prior
        envs.reset()
        optimizer_q.zero_grad()
        reward = reward_init
        actions.append(a)
        states.append(s)
        rewards.append(reward)
        hidden_state = h_0.detach()
        b_val_1, b_val_2, q_reshaped = sample_b(
            a, h_0, hidden_state, model, base_dist_al, prior_phi, q_network, reward, s, device=device
        )
        loss = msbbe(b_val_1, b_val_2, q_reshaped)
        border_distance = (envs.gridwidth - 1) / 2
        state_dim = envs.number_of_victims + envs.number_of_hazards + 2
        # Deterministic movement:
        reward = 0
        action = [randrange(5)]
        direction = randrange(4) + 1
        state = torch.zeros(state_dim)
        state[0] = randrange(envs.gridwidth - 2) - border_distance - 1
        state[1] = randrange(envs.gridwidth - 2) - border_distance - 1
        next_state = state.clone()
        if direction == 1:
            # up chosen - move upwards
            next_state[1] = state[1] + 1
        if direction == 2:
            # down chosen - move down
            next_state[1] = state[1] - 1
        if direction == 3:
            # left chosen - move left
            next_state[0] = state[0] - 1
        if direction == 4:
            # right chosen - move right
            next_state[0] = state[0] + 1
        inputs_for_q = torch.cat(
            [
                torch.tensor(state).view(1, -1).to(device),
                torch.tensor(action).view(1, -1).to(device),
                torch.tensor(reward).view(1, -1).to(device),
            ],
            dim=-1,
        )
        hidden_state = h_0.detach()
        q_val, next_hidden_state = q_network(inputs_for_q, hidden_state)
        q_a = q_val[0, direction]
        reward = 0
        next_inputs_for_q = torch.cat(
            [
                torch.tensor(next_state).view(1, -1).to(device),
                torch.tensor(direction).view(1, -1).to(device),
                torch.tensor(reward).view(1, -1).to(device),
            ],
            dim=-1,
        )
        q_val_next, _ = q_network(next_inputs_for_q, next_hidden_state)
        b = 0 + gamma * torch.max(q_val_next)
        loss_movement = (b - q_a) ** 2
        # Prior reward:
        reward = 0
        action = [randrange(5)]
        direction = randrange(4) + 1
        state = torch.zeros(state_dim)
        reward_av_prior = (
            envs.number_of_victims * envs.reward_rescue + envs.number_of_hazards * envs.reward_hazard
        ) / (4 * envs.gridwidth)
        if direction == 1:
            # Upper border - move upwards
            state[1] = border_distance
            state[0] = randrange(envs.gridwidth) - border_distance
        if direction == 2:
            # Lower border - move down
            state[1] = -border_distance
            state[0] = randrange(envs.gridwidth) - border_distance
        if direction == 3:
            # Left border - move left
            state[0] = -border_distance
            state[1] = randrange(envs.gridwidth) - border_distance
        if direction == 4:
            # Right border - move right
            state[0] = border_distance
            state[1] = randrange(envs.gridwidth) - border_distance
        inputs_for_q = torch.cat(
            [
                torch.tensor(state).view(1, -1).to(device),
                torch.tensor(action).view(1, -1).to(device),
                torch.tensor(reward).view(1, -1).to(device),
            ],
            dim=-1,
        )
        hidden_state = h_0.detach()
        q_val, next_hidden_state = q_network(inputs_for_q, hidden_state)
        q_a = q_val[0, direction]
        reward = reward_av_prior
        next_inputs_for_q = torch.cat(
            [
                torch.tensor(state).view(1, -1).to(device),
                torch.tensor(direction).view(1, -1).to(device),
                torch.tensor(reward).view(1, -1).to(device),
            ],
            dim=-1,
        )
        q_val_next, _ = q_network(next_inputs_for_q, next_hidden_state)
        b = reward_av_prior + gamma * torch.max(q_val_next)
        loss_av_reward = (b - q_a) ** 2
        loss = loss + loss_av_reward + loss_movement
        print(f"Loss: {loss}")
        loss.backward()
        # # Debugger
        # if args.debug:
        #     for name, param in q_network.named_parameters():
        #         if param.grad is None:
        #     for name, param in model.named_parameters():
        #         if param.grad is None:
        #         if param.grad is not None:
        optimizer_q.step()
        # Sampling from D
        #     ],
        #     ],
        #
        loss_hist.append(loss.detach().cpu().numpy().ravel())
        # Temporarily detach... eventually want to incorporate
    return loss_hist, q_network, optimizer_q


def sample_b(a, h_0, hidden_state, model, base_dist_al, prior_phi, q_network, reward, s, device="cuda:0"):
    num_params_aleatoric = prior_phi.sample().shape[-1]
    phi_1, phi_2 = torch.split(prior_phi.sample(2).view(1, 2, 1, -1), 1, dim=1)
    z_al_1, z_al_2 = torch.split(
        base_dist_al.sample(
            2,
        ).view(1, 2, 1, -1),
        1,
        dim=1,
    )
    # uniform sampling of actions from action space
    inputs_for_q = torch.cat(
        [torch.tensor(s).view(1, -1), torch.tensor(a).view(1, -1), torch.tensor(reward).view(1, -1)], dim=-1
    )
    conditioner_MLP = ConditionerMLP(
        num_params_aleatoric + 1 + h_0.numel(), num_params_aleatoric, num_params_aleatoric, device=device
    )
    q_val, next_hidden_state = q_network(inputs_for_q, hidden_state)
    q_max, _ = q_val.max(dim=-1, keepdim=True)
    q_reshaped = q_max.view(-1, 1, 1, 1)
    q_reshaped.retain_grad()
    b_val_1 = model.aleatoric_flows.forward(
        z_al_1,  # _mock,
        phi_1,  # _mock,
        conditioner_MLP,
        q_network,
        inputs_for_q,
        hidden_state,
    )
    b_val_1.retain_grad()
    b_val_2 = model.aleatoric_flows.forward(
        z_al_2, phi_2, conditioner_MLP, q_network, inputs_for_q, hidden_state  # _mock,  # _mock
    )
    b_val_2.retain_grad()
    return b_val_1, b_val_2, q_reshaped
