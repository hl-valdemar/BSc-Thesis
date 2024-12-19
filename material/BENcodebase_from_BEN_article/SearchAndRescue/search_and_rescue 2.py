# Core modules
import math

# 3rd party modules
import gymnasium as gym
import torch
from gymnasium import spaces

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


class SearchRescueEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        reward_hazard=-100,
        reward_rescue=10,
        reward_listen=-1,
        max_steps_per_episode=200,
        number_of_hazards=2,
        number_of_victims=2,
        gridwidth=5,
        signal_noise=0.05,
    ):
        """
        Environment for the search and rescue task with Gridwidth x Gridwidth grid. Grid width should be an odd number,
         code corrects for this if not. Agent is always initialised at (0,0) which is the centre of the grid.

         ----------
        """
        self.reward_hazard = reward_hazard
        self.reward_rescue = reward_rescue
        self.reward_listen = reward_listen
        self.signal_noise = signal_noise
        self.max_steps_per_episode = max_steps_per_episode
        self.number_of_hazards = number_of_hazards
        self.number_of_victims = number_of_victims
        self.rescue_number = 0
        self.victims_to_be_rescued = self.number_of_victims - self.rescue_number
        self.hazard_number = 0
        self.out_of_bounds = math.ceil((gridwidth + 1) / 2)
        self.gridwidth = self.out_of_bounds + self.out_of_bounds - 1

        self.curr_episode = -1  # Set to -1 b/c reset() adds 1 to episode

        self.hazard_locations = []
        self.victim_locations = []
        self.locations_continuous = []

        self.curr_step = 0
        # Agent's states: [POSITION X, POSITION Y, DIST TO VICTIMS ...,DIST TO HAZARDS ...,]
        self.state = torch.zeros(2 + self.number_of_hazards + self.number_of_victims)
        self.next_location = torch.zeros(2)

        # Define what the agent can do: LISTEN, UP, DOWN, LEFT, RIGHT
        self.action_space = spaces.Discrete(5)

        self.initialise_env()

    def initialise_env(self):
        # Define the space of possible hazard and victim locations
        locations = torch.zeros(self.gridwidth * 4, 2)
        locations[0 : self.gridwidth, 1] = self.out_of_bounds
        locations[0 : self.gridwidth, 0] = torch.tensor([range(-(self.out_of_bounds - 1), self.out_of_bounds)])
        locations[self.gridwidth : 2 * self.gridwidth, 1] = -self.out_of_bounds
        locations[self.gridwidth : 2 * self.gridwidth, 0] = torch.tensor(
            [range(-(self.out_of_bounds - 1), self.out_of_bounds)]
        )
        locations[2 * self.gridwidth : 3 * self.gridwidth, 0] = self.out_of_bounds
        locations[2 * self.gridwidth : 3 * self.gridwidth, 1] = torch.tensor(
            [range(-(self.out_of_bounds - 1), self.out_of_bounds)]
        )
        locations[3 * self.gridwidth : 4 * self.gridwidth, 0] = -self.out_of_bounds
        locations[3 * self.gridwidth : 4 * self.gridwidth, 1] = torch.tensor(
            [range(-(self.out_of_bounds - 1), self.out_of_bounds)]
        )

        # Randomly assign victims and hazards to the possible locations
        permuted_indices = torch.randperm(self.gridwidth * 4)
        locations = locations[permuted_indices, :]
        self.victim_locations = locations[0 : self.number_of_victims, :]
        self.hazard_locations = locations[self.number_of_victims : self.number_of_hazards + self.number_of_victims, :]

        # Create a randomised permutation of locations for the state observation that the agent receives
        victim_locations_continuous = self.victim_locations + torch.rand_like(self.victim_locations) - 0.5
        hazard_locations_continuous = self.hazard_locations + torch.rand_like(self.hazard_locations) - 0.5
        self.locations_continuous = torch.cat((victim_locations_continuous, hazard_locations_continuous), 0)

    def step(self, action):
        done = self.curr_step >= self.max_steps_per_episode
        if done:
            print("Episode is done")
            self.reset()
        self.curr_step += 1
        done = self.curr_step >= self.max_steps_per_episode

        self.transition(action)
        reward = self.get_reward(action)

        return self.state, reward, done

    def translate_action(self, action):
        return ACTION_MAP[action]

    def reset(self):
        self.initialise_env()
        self.next_location = torch.zeros(2)
        self.state[:2] = 0
        self.curr_step = 0
        self.rescue_number = 0
        self.victims_to_be_rescued = self.number_of_victims - self.rescue_number
        self.hazard_number = 0
        self.curr_episode += 1

    def move(self, action):
        if action == "UP":
            return torch.tensor([0, 1])
        if action == "DOWN":
            return torch.tensor([0, -1])
        if action == "LEFT":
            return torch.tensor([-1, 0])
        if action == "RIGHT":
            return torch.tensor([1, 0])

    def transition(self, action):
        if action == "LISTEN":
            self.listen_to_hazard_and_victims(self.state[:2])
        else:
            self.next_location = self.state[:2] + self.move(action)
            if torch.max(self.next_location.abs()) < self.out_of_bounds:
                self.state[:2] = self.next_location

    def listen_to_hazard_and_victims(self, state):
        relative_distances = torch.norm(self.locations_continuous - state, dim=1)
        relative_distances = relative_distances / self.gridwidth
        relative_distances = relative_distances + torch.randn_like(relative_distances) * self.signal_noise
        self.state[2:] = torch.exp(relative_distances)

    def get_reward(self, action):
        reward = None
        if action == "LISTEN":
            reward = self.reward_listen
        elif torch.max(self.next_location.abs()) >= self.out_of_bounds:
            for j in range(self.number_of_victims):
                if (self.victim_locations[j, :] == self.next_location).all():
                    self.rescue_number = self.rescue_number + 1
                    # Victim found, make sure they can't be found again and move to far enough away not to give listening signal:
                    self.victim_locations[j, :] = torch.tensor([1000 * self.gridwidth, 1000 * self.gridwidth])
                    print("\n Victim found!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    reward = self.reward_rescue
            for i in range(self.number_of_hazards):
                if (self.hazard_locations[i, :] == self.next_location).all():
                    self.hazard_number += 1
                    print("\n Hazard found!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    reward = self.reward_hazard
        if reward is None:
            reward = 0.0
        return reward
