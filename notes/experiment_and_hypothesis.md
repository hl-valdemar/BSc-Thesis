# Idea

## Experimental design

1. Compare GFlowNets and BEN on both n-chain and GridWorld with varying degrees of reward sparcity.

2. Metrics to measure:
    1. Sample efficiency (steps needed to reach optimal policy) - note: measure partly by training loss?
    2. Final performance (average reward/success rate) - note: measure by difference between sample distribution and true distributions? Otherwise, this probably doesn't make sense for the semi-deterministic n-chain environment, as we know we'll succeed in n steps.
    3. Exploration behavior (state coverage over time)

## Possible hypothesis

"In environments with highly delayed rewards, BEN's Bayesian exploration strategy leads to more efficient learning compared to GFlowNets particularly in the early stages of training. However, this advantage diminishes as the delay between actions and rewards decreases."

Rationale:
* BEN's Bayesian approach might handle uncertainty better in highly delayed reward scenarios because it has a dedicated aleatoric network to model this, as opposed to GFlowNets which models it inherently/internally.
* GFlowNets focus on generating diverse trajectories, which might require more samples initialy.
* The comparison would be particularly interesting in n-chain with varying chain lengths (affecting reward delay).
