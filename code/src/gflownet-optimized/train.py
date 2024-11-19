from typing import Callable, List

import torch
import torch.nn.functional as F

from gridworld import Action
from .model import GFlowNet
from .env import GridWorldEnv, ReplayBuffer, Trajectory
from .config import GFlowNetTrainingConfig

def train(
    # env: GridWorldEnv, 
    env_factory: Callable[[], GridWorldEnv],
    gflownet: GFlowNet, 
    config: GFlowNetTrainingConfig,
):
    buffer = ReplayBuffer(config.replay_capacity, config.trajectory_length)
    rewards_history = []
    epsilon = config.epsilon_start
    temperature = config.temperature_start
    
    # Training loop
    for episode in range(config.num_episodes):
        # Collect multiple trajectories
        print("Collecting trajectories...")
        trajectories = collect_trajectories_parallel(
            env_factory,
            gflownet,
            epsilon,
            temperature,
            num_trajectories=config.batch_size,
        )

        # Add trajectories to buffer
        for trajectory in trajectories:
            buffer.add_trajectory(trajectory)
            rewards_history.append(trajectory.total_reward)

        # current_trajectory = collect_trajectory(
        #     env, gflownet, epsilon, temperature
        # )
        # buffer.add_trajectory(current_trajectory)
        
        # Train on batch
        if len(buffer) >= config.min_experiences:
            train_trajectories = buffer.sample(config.batch_size)
            if train_trajectories is not None:
                print("Updating gflownet...")
                metrics = gflownet.update(train_trajectories)
                
                if (episode + 1) % 10 == 0:
                    print(f"Episode {episode + 1}:")
                    for k, v in metrics.items():
                        print(f"  {k}: {v:.4f}")
        
        # Update exploration parameters
        epsilon = max(
            config.epsilon_end,
            epsilon * config.epsilon_decay
        )
        temperature = max(
            config.temperature_end,
            temperature * config.temperature_decay
        )
        
    return rewards_history

def collect_trajectory(
    env: GridWorldEnv,
    gflownet: GFlowNet,
    epsilon: float,
    temperature: float,
) -> Trajectory:
    """Collect a single trajectory by running the agent in the environment.
    
    Args:
        env: The environment to interact with
        gflownet: The GFlowNet model
        epsilon: Probability of random action
        temperature: Temperature for action sampling
        
    Returns:
        Trajectory: Collected trajectory with states, actions, and rewards
    """
    current_trajectory = Trajectory([], [], [], [], False, 0.0, 0)
    step_data = env.reset()
    
    while not step_data.done:
        state = step_data.state
        
        # Get action using epsilon-greedy with temperature
        if torch.rand(1) < epsilon:
            action = Action(torch.randint(gflownet.config.action_dim, (1,)).item())
        else:
            with torch.no_grad():
                output = gflownet.forward(state.unsqueeze(0))
                # Temperature scaled probabilities with numerical stability
                logits = output.logits / temperature
                probs = F.softmax(logits, dim=-1)
                action_idx = torch.multinomial(probs[0], 1).item()
                action = Action(action_idx)
        
        # Take step in environment
        next_step_data = env.step(action)
        
        # Add to current trajectory
        current_trajectory.states.append(state)
        current_trajectory.action_indices.append(
            torch.tensor(action.value, dtype=torch.long, device=state.device)
        )
        current_trajectory.actions.append(action)
        current_trajectory.rewards.append(next_step_data.reward)
        current_trajectory.total_reward += next_step_data.reward.item()
        current_trajectory.length += 1
        
        step_data = next_step_data
    
    # Add final state without action
    current_trajectory.states.append(step_data.state)
    current_trajectory.done = True
    
    return current_trajectory

def collect_trajectories_parallel(
    env_factory: Callable[[], GridWorldEnv],
    gflownet: GFlowNet,
    epsilon: float,
    temperature: float,
    num_trajectories: int,
    num_workers: int = 4
) -> List[Trajectory]:
    """Collect multiple trajectories in parallel using multiple workers.
    
    Args:
        env_factory: Function that creates new environment instances
        gflownet: The GFlowNet model
        epsilon: Probability of random action
        temperature: Temperature for action sampling
        num_trajectories: Number of trajectories to collect
        num_workers: Number of parallel workers
        
    Returns:
        List[Trajectory]: List of collected trajectories
    """
    from concurrent.futures import ThreadPoolExecutor
    
    def collect_single():
        env = env_factory()
        return collect_trajectory(env, gflownet, epsilon, temperature)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        trajectories = list(executor.map(
            lambda _: collect_single(),
            range(num_trajectories)
        ))
    
    return trajectories

def visualize_flows(
        # env: GridWorldEnv, 
        gflownet: GFlowNet, 
        state: torch.Tensor,
    ) -> None:
    """
    Visualize the flows and policy for a given state
    """
    with torch.no_grad():
        output = gflownet.forward(state.unsqueeze(0))
        flows = output.flows[0]
        policy = flows / output.state_flow[0]
        
        print("\nFlows F(s,a):")
        for action in Action:
            print(f"{action.name}: {flows[action.value]:.3f}")
            
        print("\nPolicy π(a|s):")
        for action in Action:
            print(f"{action.name}: {policy[action.value]:.3f}")
