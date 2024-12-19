import torch
import torch as th

from ConditionerMLP import ConditionerMLP
from QBayesNet import QBayesNetwork
from BayesModel import BayesModel
from prior_initialization import msbbe

LISTEN = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
ACTION_MAP = {
    LISTEN: "LISTEN",
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
}


def sample_b(
    a,
    h_0,
    hidden_state,
    model,
    aleatoric_flow,
    epistemic_flow,
    q_network,
    reward,
    s,
    device="cpu",
):
    num_params_aleatoric = epistemic_flow.sample().shape[-1]
    phi_1, phi_2 = th.split(epistemic_flow.sample(2).view(1, 2, 1, -1), 1, dim=1)

    conditioner_MLP = ConditionerMLP(num_params_aleatoric + 1 + h_0.numel(), num_params_aleatoric, num_params_aleatoric,
                                     device=device)
    q_obs = th.cat(
        [
            th.tensor(s).view(1, -1).to(device),
            th.tensor(a).view(1, -1).to(device),
            th.tensor(reward).view(1, -1).to(device),
        ],
        dim=-1,
    )
    b_val_1 = aleatoric_flow.sample(
        phi_1,
        conditioner_network=conditioner_MLP,
        q_network=q_network,
        inputs_for_q=q_obs,
        hidden_state=hidden_state,
    ).view(1, 1, 1, -1)

    b_val_2 = aleatoric_flow.sample(
        phi_2, conditioner_network=conditioner_MLP, q_network=q_network, inputs_for_q=q_obs, hidden_state=hidden_state
    ).view(1, 1, 1, -1)
    # uniform sampling of actions from action space
    inputs_for_q = th.cat([th.tensor(s).view(1, -1).to(device),
                           th.tensor(a).view(1, -1).to(device),
                           th.tensor(reward).view(1, -1).to(device)], dim=-1)

    q_val, next_hidden_state = q_network(inputs_for_q, hidden_state)
    q_max, _ = q_val.max(dim=-1, keepdim=True)
    q_reshaped = q_max.view(-1, 1, 1, 1)
    q_reshaped.retain_grad()
    #     z_al_1,  # _mock,
    #     phi_1,  # _mock,
    #     conditioner_MLP,
    #     q_network,
    #     inputs_for_q,
    #     hidden_state,
    b_val_1.retain_grad()
    #     z_al_2, phi_2, conditioner_MLP, q_network, inputs_for_q, hidden_state  # _mock,  # _mock
    b_val_2.retain_grad()
    return b_val_1, b_val_2, q_reshaped


def posterior_update(
    h_init,
    t,
    model: BayesModel,
    q_network: QBayesNetwork,
    h_states,
    h_actions,
    h_rewards,
    action_init,
    reward_init,
    args,
    env,
    state_init,
    N_update=5,
    memory=20,
    device="cuda:0" if torch.cuda.is_available()  else "cpu",
):
    optimizer_psi = th.optim.Adam(model.parameters(), lr=args.lr_phi)
    optimizer_q = th.optim.Adam(q_network.parameters(), lr=args.lr_q)
    losses_q = []
    losses_epistemic = []
    for update_period in range(N_update):
        optimizer_q.zero_grad()
        reward_init = reward_init  # reward from randomly sampled - maybe just set to zero?
        action_init = action_init  # again, maybe just set to -1 as a special symbol?
        action_name = ACTION_MAP[action_init]
        action = th.tensor(action_init, dtype=th.float64).view(1, -1).to(args.device)
        reward = th.tensor(reward_init, dtype=th.float64).view(1, -1).to(args.device)
        state = th.tensor(state_init, dtype=th.float64).view(1, -1).to(args.device)
        q_obs = th.cat(
            [
                th.tensor(state_init, dtype=th.float64).view(1, -1).to(device),
                th.tensor(action_init, dtype=th.float64).view(1, -1).to(device),
                th.tensor(reward_init, dtype=th.float64).view(1, -1).to(device),
            ],
            dim=1,
        )
        hidden = h_init.detach()
        last_sample = max(t - memory, 0)
        for i in range(last_sample, t):
            action, reward = h_actions[i], h_rewards[i]
            next_state, next_action, next_reward = h_states[i + 1], h_actions[i + 1], h_rewards[i + 1]
            next_q_obs = th.cat(
                [
                    torch.tensor(next_state, dtype=torch.float64).view(1, -1).to(device),
                    torch.tensor(action, dtype=torch.float64).view(1, -1).to(device),
                    torch.tensor(reward, dtype=torch.float64).view(1, -1).to(device),
                ],
                dim=1,
            )
            q_val_for_all_actions, hidden = q_network(q_obs, hidden)
            hidden_conditioner = hidden.clone().detach()
            q_val_t = q_val_for_all_actions.select(1, torch.tensor(action).long())

            # We only want gradients to update psi and not pass through to omega
            with th.no_grad():
                q_vals_off_path, _ = q_network(next_q_obs, hidden_conditioner)
                v_val_est, _ = q_vals_off_path.max(1)
                try:
                    b_val = reward + args.gamma * v_val_est
                except:
                    hh = 1
            b_val = b_val.detach().view(1, 1, 1, -1)
            q_val_t = q_val_t.detach().view(1, 1, 1, -1)
            loss_epistemic = model.ELBO(b_val, q_val_t, hidden_conditioner, i)
            loss_epistemic.backward()
            losses_epistemic.append(loss_epistemic.item())
            optimizer_psi.step()
            optimizer_psi.zero_grad()
            loss_epistemic = 0

            q_obs = th.cat(
                [
                    torch.tensor(next_state, dtype=torch.float64).view(1, -1).to(device),
                    torch.tensor(action, dtype=torch.float64).view(1, -1).to(device),
                    torch.tensor(reward, dtype=torch.float64).view(1, -1).to(device),
                ],
                dim=1,
            )

        # Generate b vals as in integration_msbbe
        # Replace model.sample_b with variant from code
        b1, b2, q_reshaped = sample_b(
            action,
            h_init,
            hidden,
            model,
            model.aleatoric_flows,
            model.epistemic_flow,
            q_network,
            reward,
            state,
            device=args.device,
        )
        loss_q = msbbe(b1, b2, q_reshaped)
        if update_period % 2 == 0:
            losses_q.append(loss_q.item())
        loss_q.backward()
        optimizer_q.step()
        optimizer_q.zero_grad()
    return (
        q_network,
        model,
        losses_q,
        losses_epistemic,
    )
