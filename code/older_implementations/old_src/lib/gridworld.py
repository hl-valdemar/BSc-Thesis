import numpy as np
from enum import Enum
import threading
from raylib import rl, colors
import time
from enum import Enum
import torch

class Cell(Enum):
    EMPTY = 0
    AGENT_SPAWN = 1
    GOAL = 2
    WALL = 3
    OBSTACLE = 4

class Action(Enum):
    # INVALID = -1
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

State = tuple[int, int]

class GridWorld:
    def __init__(
        self,
        description: str,
        render=False,
        font_path: str = "../assets/fonts/monogram-extended.ttf",
        font_size: int = 30,
    ) -> None:
        # Clean up description
        description = "\n".join([line.strip() for line in description.strip().split("\n")])

        self.width = len(description.split("\n")[0])
        self.height = len(description.split("\n"))
        self.state_count = 0
        self.action_size = 4
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.grid_visited = np.zeros((self.height, self.width), dtype=bool)
        # self.grid_visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        self.actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        self.action_dict = {
            Action.UP: (0, -1),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
            Action.RIGHT: (1, 0),
        }

        state: State | None = None
        for y, row in enumerate(description.split("\n")):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.grid[y, x] = Cell.WALL.value
                elif cell == "s":
                    state = (x, y)
                    self.grid[y, x] = Cell.AGENT_SPAWN.value
                    self.grid_visited[y, x] = True
                    self.state_count += 1
                elif cell == "g":
                    self.grid[y, x] = Cell.GOAL.value
                    self.state_count += 1
                elif cell == "o":
                    self.grid[y, x] = Cell.OBSTACLE.value
                    self.state_count += 1

        if state is None:
            raise ValueError("No agent spawn point found in world description")

        self.state: State = state
        self.policy: np.ndarray | None = None

        self.should_render = render
        self.render_policy = False
        self.font_path = font_path
        self.font_size = font_size
        self.renderer_running = False
        self.finished_training = False
        self.lock = threading.Lock()
        self.paused = False

    def get_valid_actions(self) -> tuple[list[Action], torch.Tensor]:
        return self.get_valid_actions_for_state(self.state)

    def get_valid_actions_for_state(self, state: State) -> tuple[list[Action], torch.Tensor]:
        available = []
        mask = torch.zeros(len(self.actions))
        for i, action in enumerate(self.actions):
            delta = self.action_dict[action]
            new_row = state[0] + delta[0]
            new_col = state[1] + delta[1]
            if 0 <= new_row < self.width and 0 <= new_col < self.height:
                available.append(action)
                mask[i] = 1.0
        return available, mask

    def get_parents_for_state(self, state: State) -> tuple[list[State], list[int]]:
        parents: list[State] = []
        parent_actions_idx: list[int] = []
        for i, action in enumerate(self.actions):
            delta = self.action_dict[action]
            prev_row = state[0] - delta[0]
            prev_col = state[1] - delta[1]
            if 0 <= prev_row < self.width and 0 <= prev_col < self.height:
                parents.append((prev_row, prev_col))
                parent_actions_idx.append(i)
        return parents, parent_actions_idx

    def step(self, action: Action) -> tuple[State, Cell]:
        x, y = self.state
        match action:
            case Action.RIGHT if x < self.width - 1:
                x += 1
            case Action.DOWN if y < self.height - 1:
                y += 1
            case Action.LEFT if x > 0:
                x -= 1
            case Action.UP if y > 0:
                y -= 1

        if self.grid[y, x] != Cell.WALL.value:
            self.state = (x, y)

        return self.state, self.grid[y, x]

    def reset(self) -> State:
        """Reset the GridWorld to its initial state."""
        self.policy = None
        self.state = next((x, y) for y in range(self.height) for x in range(self.width) if self.grid[y, x] == Cell.AGENT_SPAWN.value)
        return self.state

    def set_policy(self, policy: np.ndarray, render_policy: bool = True):
        """Set the policy for the GridWorld and optionally enable policy rendering."""
        if policy.shape != (self.height, self.width):
            raise ValueError("Policy shape must match the GridWorld dimensions")
        self.policy = policy
        self.render_policy = render_policy

    def run_training(self, train_func, *args, **kwargs):
        """Run the training function and render if desired."""
        if self.should_render:
            self._start()

        try:
            result = train_func(self, *args, **kwargs)
        finally:
            print("Training finished. Press 'Q' or ESC to close the window.")
            self.finished_training = True

            # Wait for the user to exit the renderer
            while self.renderer_running and not rl.WindowShouldClose():
                pass

            self._stop()

        return result

    # +------------------------------------------- +
    # Rendering-related methods                    |
    # These methods are not part of the public API |
    # +------------------------------------------- +

    def _start(self):
        """Start the GridWorld simulation and rendering."""
        self.renderer_running = True
        self.render_thread = threading.Thread(target=self._render_loop)
        self.render_thread.start()

    def _stop(self):
        """Stop the GridWorld simulation and rendering."""
        self.renderer_running = False
        if hasattr(self, "render_thread"):
            # Wait for up to 5 seconds for the thread to join
            self.render_thread.join(timeout=5)
            if self.render_thread.is_alive():
                print("Render thread did not exit cleanly. Forcing window closure.")
                self._force_close_window()

    def _force_close_window(self):
        """Force close the window from the main thread."""
        rl.CloseWindow()

    def _render_loop(self):
        font_scale = 0.75

        rl.SetTraceLogLevel(rl.RL_LOG_NONE)
        rl.InitWindow(int(self.width * self.font_size * font_scale), int(self.height * self.font_size * font_scale), b"BSc :: GridWorld")
        rl.SetTargetFPS(30)

        font = rl.LoadFont(self.font_path.encode())

        while self.renderer_running and not rl.WindowShouldClose():
            if rl.IsKeyPressed(rl.KEY_Q) or rl.IsKeyPressed(rl.KEY_ESCAPE):
                self.renderer_running = False
                break

            rl.BeginDrawing()
            rl.ClearBackground(colors.BLACK)

            grid_copy = None
            with self.lock:
                grid_copy = self.grid.copy()

            # Draw grid (doesn't need to be locked since it's read-only)
            self._draw_grid(grid_copy, font, font_scale)

            rl.EndDrawing()

        rl.CloseWindow()

    def _draw_grid(self, grid, font, font_scale_factor: float):
        for y in range(self.height):
            for x in range(self.width):
                cell = grid[y, x]
                color = colors.DARKGRAY
                texture = b"."

                if cell == Cell.WALL.value:
                    color = colors.LIGHTGRAY
                    texture = b"#"
                elif cell == Cell.AGENT_SPAWN.value:
                    color = colors.VIOLET
                    texture = b"s"
                elif cell == Cell.GOAL.value:
                    color = colors.YELLOW
                    texture = b"g"
                elif cell == Cell.OBSTACLE.value:
                    color = colors.RED
                    texture = b"o"

                screen_pos = [
                    x * self.font_size * font_scale_factor,
                    y * self.font_size * font_scale_factor
                ];
                rl.DrawTextEx(font, texture, screen_pos, self.font_size, 0, color)

                if self.render_policy and self.policy is not None:
                    action = self.policy[y, x]
                    policy_texture = ""

                    if cell == Cell.EMPTY.value:
                        if action == Action.LEFT.value:
                            policy_texture = "<"
                        elif action == Action.RIGHT.value:
                            policy_texture = ">"
                        elif action == Action.UP.value:
                            policy_texture = "^"
                        elif action == Action.DOWN.value:
                            policy_texture = "v"
                        elif action == Action.INVALID.value:
                            policy_texture = "x"

                    rl.DrawTextEx(font, policy_texture.encode(), screen_pos, self.font_size, 0, colors.ORANGE)

        agent_x, agent_y = self.state
        screen_pos = [
            agent_x * self.font_size * font_scale_factor,
            agent_y * self.font_size * font_scale_factor
        ]
        rl.DrawTextEx(font, b"A", screen_pos, self.font_size, 0, colors.BLUE)
