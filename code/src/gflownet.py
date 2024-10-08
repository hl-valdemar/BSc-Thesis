import random

class GFlowNet:
    def __init__(self, env, learning_rate=0.01) -> None:
        self.env = env
        self.lr = learning_rate
        self.flows = {}  # (state, action) -> flow value

    def get_flow(self, state: tuple[int, int], action: str) -> float:
        return self.flows.get((state, action), 1.0)  # Default to 1.0

    def set_flow(self, state: tuple[int, int], action: str, value: float):
        self.flows[(state, action)] = value

    def sample_action(self, state: tuple[int, int]):
        actions = self.env.get_actions()
        flows = [self.get_flow(state, a) for a in actions]
        total_flow = sum(flows)
        probs = [f / total_flow for f in flows]
        return random.choices(actions, weights=probs)[0]

    def train_step(self, trajectory: list[tuple[tuple[int, int], str]]):
        for t in range(len(trajectory) - 1):
            state, action = trajectory[t]
            next_state = trajectory[t+1][0]

            # Calculate incoming and outgoing flows
            incoming_flow = self.get_flow(state, action)
            outgoing_flow = sum(self.get_flow(next_state, a) for a in self.env.get_actions())

            if t == len(trajectory) - 2:  # Last transition
                outgoing_flow = 1.0  # Terminal state has reward/outgoing flow of 1

            # Update flow to satisfy consistency equation
            flow_update = self.lr * (outgoing_flow - incoming_flow)
            self.set_flow(state, action, self.get_flow(state, action) + flow_update)
