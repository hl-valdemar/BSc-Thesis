import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.lines as lines
import numpy as np

base_face = lambda: (plt.gca().add_patch(pat.Circle((0.5,0.5),0.5,fc=(.9,.9,0))),
                     plt.gca().add_patch(pat.Circle((0.25,0.6),0.1,fc=(0,0,0))),
                     plt.gca().add_patch(pat.Circle((0.75,0.6),0.1,fc=(0,0,0))))
patches = {
  'smile': lambda: plt.gca().add_patch(pat.Polygon(np.stack([np.linspace(0.2,0.8), 0.3-np.sin(np.linspace(0,3.14))*0.15]).T, closed=False, fill=False, lw=3)),
  'frown': lambda: plt.gca().add_patch(pat.Polygon(np.stack([np.linspace(0.2,0.8), 0.15+np.sin(np.linspace(0,3.14))*0.15]).T, closed=False, fill=False, lw=3)),
  'left_eb_down': lambda: plt.gca().add_line(lines.Line2D([0.15, 0.35], [0.75,0.7], color=(0,0,0))),
  'right_eb_down': lambda: plt.gca().add_line(lines.Line2D([0.65, 0.85], [0.7,0.75], color=(0,0,0))),
  'left_eb_up': lambda: plt.gca().add_line(lines.Line2D([0.15, 0.35], [0.7,0.75], color=(0,0,0))),
  'right_eb_up': lambda: plt.gca().add_line(lines.Line2D([0.65, 0.85], [0.75,0.7], color=(0,0,0))),
}
sorted_patches = sorted(patches.keys())

def draw_face(face: list[str]) -> None:
    base_face()
    for i in face:
        patches[i]()
    plt.axis("scaled")
    plt.axis("off")

# f, ax = plt.subplots(1, 2)
# plt.sca(ax[0])
# draw_face(['smile', 'left_eb_down', 'right_eb_down'])
# plt.sca(ax[1])
# draw_face(['frown', 'left_eb_up', 'right_eb_up'])
# plt.show()

def has_overlap(face: list[str]) -> bool:
  # Can't have two overlapping eyebrows!
  if 'left_eb_down' in face and 'left_eb_up' in face:
    return True
  if 'right_eb_down' in face and 'right_eb_up' in face:
    return True
  # Can't have two overlapping mouths!
  if 'smile' in face and 'frown' in face:
    return True
  return False

def face_reward(face: list[str]) -> int:
  if has_overlap(face):
    return 0
  eyebrows = ('left_eb_down', 'left_eb_up', 'right_eb_down', 'right_eb_up')
  # Must have exactly two eyebrows
  if sum(i in face for i in eyebrows) != 2:
    return 0
  # We want twice as many happy faces as sad faces so here we give a reward of 2 for smiles
  if 'smile' in face:
    return 2
  if 'frown' in face:
    return 1  # and a reward of 1 for frowns
  # If we reach this point, there's no mouth
  return 0

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import tqdm

STATE_SIZE = len(patches.keys())

def face_to_tensor(face: list[str]) -> torch.Tensor:
    return torch.tensor([i in face for i in sorted_patches]).float()

class FlowModel(nn.Module):
    def __init__(self, num_hidden: int = 512) -> None:
        super().__init__()
        # We encoded the current state as binary vector, for each patch the associated
        # dimension is either 0 or 1 depending on the absence or precense of that patch.
        # Therefore the input dimension is 6 for the 6 patches.
        self.mlp = nn.Sequential(
            nn.Linear(STATE_SIZE, num_hidden),
            nn.LeakyReLU(),
            # We also output 6 numbers, since there are up to
            # 6 possible actions (and thus child states), but we
            # will mask those outputs for patches that are # already there.
            nn.Linear(num_hidden, STATE_SIZE),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # We take the exponential to get positive numbers, since flows must be positive,
        # and multiply by (1 - x) to give 0 flow to actions we know we can't take
        # (in this case, x[i] is 1 if a feature is already there, so we know we
        # can't add it again).
        F = self.mlp(state).exp() * (1 - state)
        return F

def face_parents(state):
    parent_states = []  # states that are parents of state
    parent_actions = []  # actions that lead from those parents to state

    for face_part in state:
        # For each face part, there is a parent without that part
        parent_states.append([i for i in state if i != face_part])
        # The action to get there is the corresponding index of that face part
        parent_actions.append(sorted_patches.index(face_part))

    return parent_states, parent_actions

# Instantiate model and optimizer
F_sa = FlowModel(512)
opt = torch.optim.Adam(F_sa.parameters(), 3e-4)

# Let's keep track of the losses and the faces we sample
losses = []
sampled_faces = []
# To not complicate the code, I'll just accumulate losses here and take a
# gradient step every `update_freq` episode.
minibatch_loss = 0
update_freq = 4
for episode in tqdm.tqdm(range(50000), ncols=40):
  # Each episode starts with an "empty state"
  state = []
  # Predict F(s, a)
  edge_flow_prediction = F_sa(face_to_tensor(state))
  for t in range(3):
    # The policy is just normalizing, and gives us the probability of each action
    policy = edge_flow_prediction / edge_flow_prediction.sum()
    # Sample the action
    action = Categorical(probs=policy).sample()
    # "Go" to the next state
    new_state = state + [sorted_patches[action]]

    # Now we want to compute the loss, we'll first enumerate the parents
    parent_states, parent_actions = face_parents(new_state)
    # And compute the edge flows F(s, a) of each parent
    px = torch.stack([face_to_tensor(p) for p in parent_states])
    pa = torch.tensor(parent_actions).long()
    parent_edge_flow_preds = F_sa(px)[torch.arange(len(parent_states)), pa]
    # Now we need to compute the reward and F(s, a) of the current state,
    # which is currently `new_state`
    if t == 2:
      # If we've built a complete face, we're done, so the reward is > 0
      # (unless the face is invalid)
      reward = face_reward(new_state)
      # and since there are no children to this state F(s,a) = 0 \forall a
      edge_flow_prediction = torch.zeros(6)
    else:
      # Otherwise we keep going, and compute F(s, a)
      reward = 0
      edge_flow_prediction = F_sa(face_to_tensor(new_state))

    # The loss as per the equation above
    flow_mismatch = (parent_edge_flow_preds.sum() - edge_flow_prediction.sum() - reward).pow(2)
    minibatch_loss += flow_mismatch  # Accumulate
    # Continue iterating
    state = new_state

  # We're done with the episode, add the face to the list, and if we are at an
  # update episode, take a gradient step.
  sampled_faces.append(state)
  if episode % update_freq == 0:
    losses.append(minibatch_loss.item())
    minibatch_loss.backward()
    opt.step()
    opt.zero_grad()
    minibatch_loss = 0

# plt.figure(figsize=(10,3))
# plt.plot(losses)
# plt.yscale('log')
# plt.show()

f, ax = plt.subplots(8,8,figsize=(4,4))
print('Ratio of faces with a smile:', sum(['smile' in i for i in sampled_faces[-128:]]) / 128)
print('Ratio of valid faces:', sum([face_reward(i) > 0 for i in sampled_faces[-128:]]) / 128)
# for i, face in enumerate(sampled_faces[-64:]):
#     plt.sca(ax[i//8,i%8])
#     draw_face(face)
# plt.show()
