= Bachelor's Proposal: Sequential Decision Making with Delayed and Sparse Rewards

Valdemar H. Lorenzen (valor21)

== Problem
In reinforcement learning, obtaining algorithms with good sample efficiency while maintaining high asymptotic performance is crucial. This is especially challenging in environments with delayed and sparse rewards, where traditional reward design can be expensive or infeasible. Examples include drug design or image synthesis, where intermediate steps cannot be easily evaluated. This project aims to explore methods for sequential decision-making in such environments, focusing on sparse reward techniques.

== Data
We will evaluate the success of our algorithms using illustrative simulation environments commonly used in research on discrete Markov Decision Processes, such as:

- n-chain
- GridWorld

//These environments will provide a controlled setting to test our algorithms' performance with delayed and sparse rewards.

== Methods
We will investigate sparse reward techniques, including:

- GFlowNets@bengio2021flownetworkbasedgenerative
- Deep exploration networks (BEN)@fellows2024bayesianexplorationnetworks

//Our approach will involve developing models that can effectively navigate environments with delayed and sparse rewards. We will explore how these methods can improve sample efficiency and performance in scenarios where reward signals are limited.

== Evaluation
To evaluate our algorithms, we will use the simulation environments mentioned in the Data section.
//We will measure performance using metrics such as:
//
//- Sample efficiency (learning speed)
//- Asymptotic performance
//- Ability to handle delayed rewards
//- Exploration-exploitation balance

== Distribution of work
The following is an estimate of the percentage-wise distribution of workload for this project:

- Literature Survey: 20%
- Implementation of proof of concept: 25%
- Iterative hypothesis refinement: 40%
- Report writing: 15%

#bibliography("works.bib", style: "ieee")