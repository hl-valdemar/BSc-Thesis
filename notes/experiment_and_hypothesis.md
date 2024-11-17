# Idea

## Experimental design

1. Compare GFlowNets and BEN on both n-chain and GridWorld with varying degrees of reward sparcity.

2. Key metrics to measure:
    * Sample efficiency (steps needed to reach optimal policy)
    * Final performance (average reward/success rate)
    * Exploration behavior (state coverage over time)

## Possible hypothesis

"In environments with highly delayed rewards, BEN's Bayesian exploration strategy leads to more efficient learning compared to GFlowNets particularly in the early stages of training. However, this advantage diminishes as the delay between actions and rewards decreases."

Rationale:
* BEN's Bayesian approach might handle uncertainty better in highly delayed reward scenarios.
* GFlowNets focus on generating diverse trajectories, which might require more samples initialy.
* The comparison would be particularly interesting in n-chain with varying chain lengths (affecting reward delay).
