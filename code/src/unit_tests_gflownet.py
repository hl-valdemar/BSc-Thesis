import unittest

import numpy as np
import torch

from gflownet_new.env import GridWorldGFlowNet


class TestGFlowNet(unittest.TestCase):
    def setUp(self):
        """Setup common test components."""
        self.grid_size = 3
        self.hidden_dim = 64
        self.model = GridWorldGFlowNet(self.grid_size, self.hidden_dim)
        self.batch_size = 8

    def _get_random_state(self) -> torch.Tensor:
        """Helper to get valid random state."""
        state = torch.zeros(self.batch_size, self.grid_size * self.grid_size)
        for i in range(self.batch_size):
            pos = np.random.randint(0, self.grid_size * self.grid_size)
            state[i, pos] = 1
        return state

    def test_flow_non_negativity(self):
        """Test that all flow values are non-negative."""
        state = self._get_random_state()
        policy_logits, state_flow, edge_flows = self.model(state)

        self.assertTrue(torch.all(state_flow >= 0), "State flow must be non-negative")
        self.assertTrue(torch.all(edge_flows >= 0), "Edge flows must be non-negative")

    def test_flow_matching_loss_computable(self):
        """Test that we can compute a flow matching loss (not that it's small)."""
        state = self._get_random_state()
        policy_logits, state_flow, edge_flows = self.model(state)

        # Just verify we can compute a loss and it's finite
        loss = self.model.compute_flow_matching_loss(
            state, policy_logits, state_flow, edge_flows
        )

        self.assertFalse(torch.isinf(loss), "Flow matching loss should not be infinite")
        self.assertFalse(torch.isnan(loss), "Flow matching loss should not be NaN")

    def test_policy_normalization(self):
        """Test that policy probabilities sum to 1."""
        state = self._get_random_state()
        policy_logits, _, _ = self.model(state)
        policy_probs = torch.softmax(policy_logits, dim=-1)

        sums = torch.sum(policy_probs, dim=-1)
        torch.testing.assert_close(
            sums,
            torch.ones_like(sums),
            rtol=1e-5,
            atol=1e-5,
            msg="Policy probabilities must sum to 1",
        )

    def test_state_transitions(self):
        """Test that state transitions are valid."""
        # Create a single state at (0,0)
        state = torch.zeros(1, self.grid_size * self.grid_size)
        state[0, 0] = 1

        # Test all actions
        actions = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

        for action, name in actions.items():
            next_state = self.model.get_next_state(state[0], action)

            # Check one-hot property
            self.assertEqual(
                torch.sum(next_state).item(),
                1.0,
                f"Next state for action {name} must be one-hot",
            )

            # Check valid position
            pos = torch.where(next_state == 1)[0].item()
            x, y = pos % self.grid_size, pos // self.grid_size
            self.assertTrue(
                0 <= x < self.grid_size and 0 <= y < self.grid_size,
                f"Invalid position ({x},{y}) after action {name}",
            )

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        state = self._get_random_state()

        # Get baseline outputs to determine shapes
        policy_logits, state_flow, edge_flows = self.model(state)

        # Test with very small flows
        tiny_state_flow = torch.full_like(state_flow, 1e-10)
        tiny_edge_flows = torch.full_like(edge_flows, 1e-10)
        loss_tiny = self.model.compute_flow_matching_loss(
            state, policy_logits, tiny_state_flow, tiny_edge_flows
        )
        self.assertFalse(
            torch.isnan(loss_tiny).any(), "Loss should handle very small values"
        )

        # Test with very large flows
        large_state_flow = torch.full_like(state_flow, 1e10)
        large_edge_flows = torch.full_like(edge_flows, 1e10)
        loss_large = self.model.compute_flow_matching_loss(
            state, policy_logits, large_state_flow, large_edge_flows
        )
        self.assertFalse(
            torch.isnan(loss_large).any(), "Loss should handle very large values"
        )

    def test_trajectory_termination(self):
        """Test that trajectories properly terminate."""
        # Start from (0,0)
        initial_state = torch.zeros(self.grid_size * self.grid_size)
        initial_state[0] = 1

        trajectory = self.model.sample_trajectory(initial_state, max_steps=100)

        # Check if trajectory terminates
        final_state = self.model.get_next_state(trajectory[-1][0], trajectory[-1][1])

        self.assertTrue(
            len(trajectory) <= 100, "Trajectory must terminate within max_steps"
        )

        # If reached goal, check it's valid
        if self.model.is_terminal(final_state):
            goal_pos = self.grid_size * self.grid_size - 1
            self.assertEqual(
                torch.where(final_state == 1)[0].item(),
                goal_pos,
                "Terminal state must be goal position",
            )


if __name__ == "__main__":
    unittest.main()
