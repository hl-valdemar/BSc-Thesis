import normflows as nf
import torch
from torch import Tensor
from tqdm import tqdm

from environments.nchain import NChainEnv

from .BayesModel import BayesModel
from .ConditionerMLP import ConditionerMLP
from .QBayesNet import QBayesNetwork


def msbbe(b1, b2, q):
    cons = (b1 - q).detach()
    loss = torch.abs(cons * (b2 - q))
    return loss


def prior_initialization(
    # envs: gym.vector.SyncVectorEnv,
    envs: NChainEnv,
    prior_phi: nf.distributions.base.DiagGaussian,
    base_dist_al: nf.distributions.base.DiagGaussian,
    action_init: Tensor,
    reward_init: Tensor,
    s_0: Tensor,
    q_network: QBayesNetwork,
    model: BayesModel,
    N_pretrain: int = 1000,
    hist_length: int = 20,
    device="cpu",
    gamma=0.8,
    args=None,
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
    optimizer_q = torch.optim.Adam(
        q_params, lr=args.lr_q
    )  # initialize omega weights of q_net and fetch
    actions = []
    states = []
    rewards = []
    # DEBUG
    #
    s_0 = s_0.clone().detach().view(1, -1)
    s = s_0.clone()
    a = action_init
    model.add_module("q_network", q_network)

    for n in tqdm(range(N_pretrain), desc="Initializing prior"):
        # Sample from prior
        envs.reset()
        optimizer_q.zero_grad()
        # # for h in range(hist_length):
        reward = reward_init
        actions.append(a)
        states.append(s)
        rewards.append(reward)
        hidden_state = h_0.detach()
        b_val_1, b_val_2, q_reshaped = sample_b(
            a,
            h_0,
            hidden_state,
            model,
            base_dist_al,
            prior_phi,
            q_network,
            reward,
            s,
            device=device,
        )
        # b_val_1, q_val = model.sample_b(q_network, s, action_init, reward, hidden_state, a, phi_1, z_al_1)
        # b_val_1 = b_val_1
        # b_val_2, q_val = model.sample_b(q_network, s, action_init, reward, hidden_state, a, phi_2, z_al_2)
        loss = msbbe(b_val_1, b_val_2, q_reshaped)
        loss.backward()
        # # Debugger
        # if args.debug:
        #     for name, param in q_network.named_parameters():
        #         if param.grad is None:
        #             print(name, "has None gradient")
        #     for name, param in model.named_parameters():
        #         if param.grad is None:
        #             print(name, "has None gradient")
        #         if param.grad is not None:
        #             print(name + ".grad", param.grad)
        #     print(b_val_1.grad)
        #     print(loss.grad)
        optimizer_q.step()
        # Sampling from D
        # i = torch.randint(0, len(actions), (1,))
        # phi_1, phi_2 = torch.split(prior_phi.sample(2).view(1, 2, 1, -1), 1, dim=1)
        # hidden_state = h_0.detach()
        # envs.set_attr("state", states[i])
        # next_state, next_reward, done, _ = envs.step(
        #     [
        #         int(actions[i].item()),
        #     ]
        # )
        # envs.set_attr("state", states[i])
        # next_state_prime, next_reward_prime, done_prime, _ = envs.step(
        #     [
        #         int(actions[i].item()),
        #     ]
        # )
        # inputs_init = torch.cat([states[i], actions[i], rewards[i]], dim=-1)
        # q_val_init, next_hidden_state = q_network(inputs_init, hidden_state)
        # q_val_action = q_val_init.select(dim=-1, index=actions[i])
        # next_action = q_val_init.argmax(dim=-1)
        # # state_dict = q_network.state_dict()
        # next_init_1 = torch.cat(
        #     [
        #         torch.tensor(next_state).view(1, -1),
        #         torch.tensor(next_action).view(1, -1),
        #         torch.tensor(next_reward).view(1, -1),
        #     ],
        #     dim=-1,
        # )
        # q_val_1, next_next_hidden_1 = q_network(next_init_1, next_hidden_state)
        # # q_network.load_state_dict(state_dict)
        # next_init_2 = torch.cat(
        #     [
        #         torch.tensor(next_state_prime).view(1, -1),
        #         torch.tensor(next_action).view(1, -1),
        #         torch.tensor(next_reward_prime).view(1, -1),
        #     ],
        #     dim=-1,
        # )
        # q_val_2, next_next_hidden_2 = q_network(next_init_2, next_hidden_state)
        # b_val_1 = torch.tensor(rewards[i].view(1, -1) + gamma * q_val_1.max(dim=-1)[0].view(1, -1))
        # b_val_2 = torch.tensor(gamma * q_val_2.max(dim=-1)[0].view(1, -1))
        # loss_new = msbbe(b_val_1, b_val_2, q_val_action)
        #
        # print(f"Loss: {loss_new}")
        loss_hist.append(loss.detach().cpu().numpy().ravel())
        # loss_new.backward()
        # optimizer_q.step()

        # Temporarily detach... eventually want to incorporate
    return loss_hist, q_network, optimizer_q


def sample_b(
    a,
    h_0,
    hidden_state,
    model,
    base_dist_al,
    prior_phi,
    q_network,
    reward,
    s,
    device="cpu",
):
    num_params_aleatoric = prior_phi.sample().shape[-1]
    phi_1, phi_2 = torch.split(prior_phi.sample(2).view(1, 2, 1, -1), 1, dim=1)
    # phi_1_mock = torch.ones_like(phi_1)
    # phi_2_mock = torch.ones_like(phi_2)
    z_al_1, z_al_2 = torch.split(
        base_dist_al.sample(
            2,
        ).view(1, 2, 1, -1),
        1,
        dim=1,
    )
    # z_al_1_mock = torch.ones_like(z_al_1)
    # z_al_2_mock = torch.ones_like(z_al_2)
    # uniform sampling of actions from action space
    inputs_for_q = torch.cat(
        [
            s.clone().detach().view(1, -1),
            a.clone().detach().view(1, -1),
            reward.clone().detach().view(1, -1),
        ],
        dim=-1,
    )
    conditioner_MLP = ConditionerMLP(
        num_params_aleatoric + 1 + h_0.numel(),
        num_params_aleatoric,
        num_params_aleatoric,
        device=device,
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
        z_al_2,
        phi_2,
        conditioner_MLP,
        q_network,
        inputs_for_q,
        hidden_state,  # _mock,  # _mock
    )
    b_val_2.retain_grad()
    return b_val_1, b_val_2, q_reshaped
