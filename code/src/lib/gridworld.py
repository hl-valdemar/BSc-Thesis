import numpy as np
from enum import Enum
import threading
from raylib import rl, colors
import time

class Key(Enum):
    ESCAPE = rl.KEY_ESCAPE
    CAPSLOCK = rl.KEY_CAPS_LOCK
    Q = rl.KEY_Q

class Cell(Enum):
    EMPTY = 0
    AGENT_SPAWN = 1
    GOAL = 2
    WALL = 3
    OBSTACLE = 4

class GridWorld:
    def __init__(self, description: str, render=False) -> None:
        # Clean up description
        description = "\n".join([line.strip() for line in description.strip().split("\n")])

        self.width = len(description.split("\n")[0])
        self.height = len(description.split("\n"))
        self.state: tuple[int, int]
        self.grid = np.zeros((self.height, self.width), dtype=int)

        for y, row in enumerate(description.split("\n")):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.grid[y, x] = Cell.WALL.value
                elif cell == "s":
                    self.grid[y, x] = Cell.AGENT_SPAWN.value
                    self.state = (x, y)
                elif cell == "g":
                    self.grid[y, x] = Cell.GOAL.value
                elif cell == "o":
                    self.grid[y, x] = Cell.OBSTACLE.value

        if self.state is None:
            raise ValueError("No agent spawn point found in world description")

        self.should_render = render
        self.font_size = 30
        self.running = False
        self.lock = threading.Lock()

    def get_actions(self) -> list[str]:
        actions = []
        x, y = self.state
        if x < self.width - 1 and self.grid[y, x+1] != Cell.WALL.value:
            actions.append('right')
        if y < self.height - 1 and self.grid[y+1, x] != Cell.WALL.value:
            actions.append('down')
        if x > 0 and self.grid[y, x-1] != Cell.WALL.value:
            actions.append('left')
        if y > 0 and self.grid[y-1, x] != Cell.WALL.value:
            actions.append('up')
        return actions

    def step(self, action: str) -> tuple[tuple[int, int], bool]:
        with self.lock:
            x, y = self.state
            if action == 'right' and x < self.width - 1:
                x += 1
            elif action == 'down' and y < self.height - 1:
                y += 1
            elif action == 'left' and x > 0:
                x -= 1
            elif action == 'up' and y > 0:
                y -= 1
            else:
                raise ValueError(f"Invalid action: {action}")

            if self.grid[y, x] != Cell.WALL.value:
                self.state = (x, y)

            done = self.grid[y, x] == Cell.GOAL.value
            return self.state, done

    def start(self):
        """Start the GridWorld simulation and rendering."""
        self.running = True
        self.render_thread = threading.Thread(target=self._render_loop)
        self.render_thread.start()

    def stop(self):
        """Stop the GridWorld simulation and rendering."""
        self.running = False
        if hasattr(self, "render_thread"):
            self.render_thread.join()

    def _render_loop(self):
        rl.InitWindow(self.width * self.font_size, self.height * self.font_size, b"GridWorld")
        rl.SetTargetFPS(30)

        while self.running and not rl.WindowShouldClose():
            if rl.IsKeyPressed(rl.KEY_Q) or rl.IsKeyPressed(rl.KEY_ESCAPE):
                self.running = False

            rl.BeginDrawing()
            rl.ClearBackground(colors.BLACK)

            grid_copy = None
            with self.lock:
                grid_copy = self.grid.copy()

            # Draw grid (doesn't need to be locked since it's read-only)
            self._draw_grid()

            rl.EndDrawing()

        rl.CloseWindow()

    def run_training(self, train_func):
        """Run the training function and render if desired."""
        if self.should_render:
            self.start()

        try:
            train_func(self)
        finally:
            print("Simulation finished. Press 'Q' or ESC to close the window.")

    def _draw_grid(self):
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y, x]
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

                rl.DrawText(texture, x * self.font_size, y * self.font_size, self.font_size, color)

        agent_x, agent_y = self.state
        rl.DrawText(b"A", agent_x * self.font_size, agent_y * self.font_size, self.font_size, colors.BLUE)
