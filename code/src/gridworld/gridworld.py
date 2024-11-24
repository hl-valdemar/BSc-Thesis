from typing import Tuple, List, Dict, Any
from enum import Enum

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class GridWorld:
    def __init__(self, layout: str | None = None, width: int | None = None, height: int | None = None, 
                 start_pos: Tuple[int, int] | None = None, goal_pos: Tuple[int, int] | None = None, 
                 obstacles: List[Tuple[int, int]] | None = None) -> None:
        """Initialize GridWorld either from a text layout or explicit parameters.
        
        Args:
            layout: Text representation of the world using:
                   's' for start
                   'g' for goal
                   'o' for obstacles
                   '#' for walls/bounds
                   ' ' for empty spaces
            width: Explicit width of the grid
            height: Explicit height of the grid 
            start_pos: Explicit start position
            goal_pos: Explicit goal position
            obstacles: Explicit list of obstacle positions
        """
        if layout is not None:
            self._init_from_layout(layout.strip())
        elif all(x is not None for x in [width, height, start_pos, goal_pos, obstacles]):
            self._init_from_parameters(width, height, start_pos, goal_pos, obstacles)
        else:
            raise ValueError("Must provide either a layout string or all explicit parameters")
            
        # Initialize state
        self.current_pos = self.start_pos
        self.steps_taken = 0
        self.max_steps = self.width * self.height * 2  # Set maximum steps as twice the grid area
        
        # Define rewards
        self.rewards = {
            'goal': 100.0,
            'obstacle': -100.0,
            'step': -1.0
        }
    
    def _init_from_layout(self, layout: str) -> None:
        """Initialize the grid from a text layout."""
        # Split into rows and verify rectangular shape
        rows = layout.strip().split('\n')
        rows = [row.strip() for row in rows]
        
        if not rows:
            raise ValueError("Empty layout")
            
        # Verify rectangular shape
        width = len(rows[0])
        if not all(len(row) == width for row in rows):
            raise ValueError("Layout must be rectangular")
            
        self.width = width - 2  # Subtract boundary walls
        self.height = len(rows) - 2  # Subtract boundary walls
        
        # Initialize positions
        self.start_pos = None
        self.goal_pos = None
        self.obstacles = []
        
        # Parse layout
        for y, row in enumerate(rows[1:-1]):  # Skip boundary rows
            for x, cell in enumerate(row[1:-1]):  # Skip boundary columns
                pos = (x, self.height - 1 - y)  # Invert y to match coordinate system
                if cell.lower() == 's':
                    self.start_pos = pos
                elif cell.lower() == 'g':
                    self.goal_pos = pos
                elif cell.lower() == 'o':
                    self.obstacles.append(pos)
                    
        # Validate required positions
        if self.start_pos is None:
            raise ValueError("Layout must contain a start position 's'")
        if self.goal_pos is None:
            raise ValueError("Layout must contain a goal position 'g'")
    
    def _init_from_parameters(self, width: int, height: int, 
                            start_pos: Tuple[int, int], goal_pos: Tuple[int, int], 
                            obstacles: List[Tuple[int, int]]) -> None:
        """Initialize the grid from explicit parameters."""
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles
        self.current_pos: Tuple[int, int] = start_pos
    
    def get_layout_str(self) -> str:
        """Returns the string representation of the current grid."""
        # Create empty grid
        grid = [[' ' for _ in range(self.width + 2)] for _ in range(self.height + 2)]
        
        # Add boundaries
        for i in range(self.width + 2):
            grid[0][i] = '#'
            grid[-1][i] = '#'
        for i in range(self.height + 2):
            grid[i][0] = '#'
            grid[i][-1] = '#'
            
        # Add internal elements
        for y in range(self.height):
            for x in range(self.width):
                pos = (x, y)
                grid_y = self.height - y  # Invert y for display
                if pos == self.start_pos:
                    grid[grid_y][x + 1] = 's'
                elif pos == self.goal_pos:
                    grid[grid_y][x + 1] = 'g'
                elif pos in self.obstacles:
                    grid[grid_y][x + 1] = 'o'
                    
        return '\n'.join(''.join(row) for row in grid)
        
    def is_within_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if the position is within grid boundaries."""
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height
    
    def get_next_state(self, state: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """Get next state given current state and action.
        
        Only prevents moving outside boundaries. Allows moving into obstacles
        but those moves will be penalized in the step function.
        """
        x, y = state
        if action == Action.UP:
            next_pos = (x, y + 1)
        elif action == Action.RIGHT:
            next_pos = (x + 1, y)
        elif action == Action.DOWN:
            next_pos = (x, y - 1)
        else:  # LEFT
            next_pos = (x - 1, y)
            
        # Only prevent out-of-bounds moves
        return next_pos if self.is_within_bounds(next_pos) else state
    
    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """Execute action and return (next_state, reward, done, info).
        
        Returns:
            next_state: The new position after taking the action
            reward: The reward received for this step
            done: Whether the episode has ended
            info: Additional information about the step
        """
        self.steps_taken += 1
        next_pos = self.get_next_state(self.current_pos, action)
        self.current_pos = next_pos
        
        # Initialize info dict
        info = {
            'hit_obstacle': False,
            'reached_goal': False,
            'timeout': False
        }
        
        # Check termination conditions and calculate reward
        if next_pos == self.goal_pos:
            reward = self.rewards['goal']
            done = True
            info['reached_goal'] = True
        elif next_pos in self.obstacles:
            reward = self.rewards['obstacle']
            done = True
            info['hit_obstacle'] = True
        else:
            reward = self.rewards['step']
            done = False
            
        # Check for timeout
        if self.steps_taken >= self.max_steps:
            done = True
            info['timeout'] = True
            
        return next_pos, reward, done, info
    
    def reset(self) -> Tuple[int, int]:
        """Reset the environment to initial state.
        
        Returns:
            The initial state (start position)
        """
        self.current_pos = self.start_pos
        self.steps_taken = 0
        return self.start_pos
