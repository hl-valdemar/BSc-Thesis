#import "@preview/showybox:2.0.3": showybox

#set page(
    header: context {
      if counter(page).get().first() > 1 {
        grid(
          columns: (1fr, 1fr, 1fr),
          box(width: 100%, align(left)[Valdemar H. Lorenzen]),
          box(width: 100%, align(center)[BSc Thesis]),
          box(width: 100%, align(right)[IMADA, SDU]),
        )
      }
    },
    footer: context {
        if here().page() != 1 {
            align(center)[#counter(page).get().first()]
        }
    },
    numbering: "1",
    margin: (
      top: 4cm,
      bottom: 4cm,
      x: 2.75cm,
    ),
)

#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
//#set text(size: 11pt)
#set par(
  //first-line-indent: 1em,
  //spacing: 1em,
  justify: true,
)

#show heading: set block(above: 1.75em, below: 1em)

#let show_notes = true

#let note(body, fill: yellow) = {
    if show_notes {
        set text(black)
        block(
          fill: fill,
          inset: 12pt,
          breakable: true,
          [#smallcaps[note] #v(0pt) #body],
        )
    }
}

#let todo(body, fill: orange.lighten(50%)) = {
    if show_notes {
        set text(black)
        block(
            fill: fill,
            inset: 12pt,
            breakable: true,
            [
                #set align(left)
                #smallcaps[todo] #v(0pt) #body
            ],
        )
    }
}

#let maybe(body, fill: red) = {
    highlight(fill: fill, extent: 1pt)[#body]
}

#let citation_needed(fill: red) = {
    highlight(fill: fill, extent: 1pt, "[CITATION NEEDED]")
}

#let attention(title, body) = {
    showybox(
        title-style: (
            weight: 900,
            color: black,
            sep-thickness: 0pt,
            align: left
        ),
        frame: (
            title-color: white,
            border-color: black.lighten(50%),
            thickness: (left: 1.5pt),
            radius: 0pt
        ),
    )[*#title:* #body]
}

#let make_title(title, authors, supervisors) = {
    align(center)[#text(size: 18pt, weight: 900)[#title]]
    v(0.5em)
    align(center)[
        #for (i, name) in authors.enumerate() {
            name

            // Spacing between names (if not last name in list)
            if i + 1 != authors.len() {
                h(1em)
            }
        }
    ]
    align(center)[
        #for (i, name) in supervisors.enumerate() {
            [\*#name]

            // Spacing between names (if not last name in list)
            if i + 1 != supervisors.len() {
                h(1em)
            }
        }
    ]
    v(1.5em)
    align(center)[IMADA, SDU]
}

#let make_abstract(content) = {
    align(center)[
        #text(size: 16pt, weight: 900)[Abstract]
        \ \
        #content
    ]
}

//#note[
//- Use consistent mathematical notation throughout
//- Include clear figures and diagrams
//- Provide code snippets for key implementations
//- Reference related work appropriately
//- Writing style: maintain an academic tone while ensuring readability. Use precise technical language but explain complex concepts clearly. Include examples and visualizations to aid understanding.
//]
//
//#note[
//    Maybe there is not enough natural stochasticiy in the environment's reward-state transition dynamics to make this interesting.
//    Maybe this could remidied by:
//    - Making the chain length random between episodes?
//    - Making the reward for reaching a terminal state random following a pre-determined distribution (by sampling the rewards from pre-determined distributions corresponding to each terminal state, we can still make assumptions about the desired behavior --- bigger mean rewards $=>$ higher desired sample rates for trained model).
//]

#let title = "BSc Thesis"
#let authors = (
    "Valdemar H. Lorenzen",
)
#let supervisors = (
    "Melih Kandemir",
)

#make_title(title, authors, supervisors)
#v(1cm)
#make_abstract[#todo[Write the abstract]]

//#set page(columns: 2)

#pagebreak()
#counter(page).update(1)

//#outline()

= Introduction <introduction>

Many real-world applications present an inherent challenge that current reinforcement learning (RL) methods struggle to address effectively: the problem of delayed and sparse rewards @sutton2018reinforcementlearninganintroduction @houthooft2017vimevariationalinformationmaximizing.

#attention([Delayed and Sparse Rewards])[
    Learning scenarios where meaningful feedback signals (rewards) are provided only far after a long sequence of actions, and where most actions yield no immediate feedback.

    _Example: In drug discovery, the effectiveness of a designed molecule can only be evaluated after its complete synthesis, with no intermediate feedback during the design process._
]

Consider, for instance, the process of drug design, where a reinforcement learning agent must make a series of molecular modifications to create an effective compound.
The value of these decisions --- the drug's efficacy --- can only be assessed once the entire molecule is complete.
Similarly, in robotics tasks like assembly or navigation, success often depends on precise sequences of actions where feedback is only available upon having completed the entire task.

Traditional reinforcement learning algorithms face two critical limitations in such environments:

+ *Credit Assignment:* When rewards are delayed, the algorithm struggles to correctly attribute success or failure to specific actions in a long sequence @harutyunyan2019hindsightcreditassignment.
    This is analogous to trying to improve a chess strategy when only knowing the game's outcome, without understanding which moves were actually decisive.

+ *Exploration Efficiency:* With sparse rewards, random exploration becomes highly inefficient @osband2016deepexplorationbootstrappeddqn @pathak2017curiositydrivenexplorationselfsupervisedprediction.
    An agent might need to execute precisely the right sequence of actions to receive any feedback at all, making random exploration about as effective as searching for a needle in a haystack.

This thesis investigates a novel approach to addressing these challenges through the comparison of two promising methodologies: *Generative Flow Networks* (GFlowNets) as proposed by @bengio2021flownetworkbasedgenerative, and *Bayesian Exploration Networks* (BEN) as proposed by @fellows2024bayesianexplorationnetworks.
These approaches represent different perspectives on handling uncertainty and exploration in reinforcement learning.

+ _GFlowNets_ frame the learning process as a flow network, potentially offering more robust learning in situations with multiple viable solutions.

+ _BENs_ leverages Bayesian uncertainty estimation to guide exploration more efficiently, potentially making better use of limited feedback.

By comparing these approaches, we aim to understand their relative strengths and limitations in environments with delayed and sparse rewards.
Our investigation focuses specifically on examining these methods in carefully designed environments that capture the essential characteristics of delayed and sparse reward scenarios while remaining tractable for systematic analysis.

== Research Objectives and Contributions

This thesis aims to advance our understanding of efficient learning in sparse reward environments through three primary objectives:

+ *Comparative Analysis:* Conduct a rigorous empirical comparison between GFlowNets and Bayesian Exploration Networks in standardized environments with delayed rewards.

+ *Hypothesis Testing:* Investigate whether BEN's Bayesian exploration strategy leads to more efficient learning compared to GFlowNets in highly delayed reward scenarios, particularly during early training stages. 

+ *Algorithmic Understanding:* Analyze the underlying mechanisms that drive performance differences between these approaches, focusing on their handling of uncertainty and exploration.

The contributions of this work include:

- A comprehensive empirical evaluation using the n-chain environment with varying degrees of reward delay.

- #maybe[Insights into the relative strengths and limitations of Bayesian and flow-based approaches to exploration.]

- Implementation and analysis of both algorithms with comparisons.

== Thesis and Structure

The remainder of this thesis is structured as follows:

*@preliminaries: Preliminaries* provides the theoretical foundations of reinforcement learning and explores existing approaches to handling sparse rewards.
This chapter establishes the mathematical framework and notation used throughout the thesis.

*@theoretical_framework: Theoretical Framework* presents our hypothesis and analytical approach.
We develop the mathematical foundations for comparing GFlowNets and BEN.

*@experimental_design: Experimental Design* details our testing methodology, including environment specifications, evaluation metrics, and implementation details.

*@results_and_analysis: Results and Analysis* presents our findings, including both quantitative performance metrics and qualitative analysis of learning behaviors.
We examine how each algorithm handles the exploration-exploitation trade-off and adapts to varying levels of reward sparsity.

*@conclusion: Conclusion* summarizes our findings, discusses their implications for the field, and suggests directions for future research.


= Preliminaries <preliminaries>

== Flow Networks

GFlowNets rely on the concept of flow networks. The flow network is represented as a directed acyclic graph $G = (cal(S), cal(A))$, where $cal(S)$ represents the state space and $cal(A)$ represents the action space.

#attention([Flow Network])[
    A directed acyclic graph with a single source node (initial state) and one or more sink nodes (terminal states), where flow is conserved at each intermediate node @bengio2021flownetworkbasedgenerative @malkin2023trajectorybalanceimprovedcredit.

    _Example: In molecular design, states represent partial molecules and actions represent adding molecular fragments._
]

=== States and Trajectories

We distinguish severel types of states:

- An initial state $s_0 in cal(S)$ (the source);
- Terminal states $x in cal(X) subset cal(S)$ (sinks);
- Intermediate states that form the pathways from source to sinks.

A trajectory $tau$ represents a complete path through the network, starting at $s_0$ and ending at some terminal state $x$.
Formally, we write a trajectory as an ordered sequence $tau = (s_0 -> s_1 -> ... -> s_n = x)$, where each transition $(s_t -> s_(t + 1))$ corresponds to an action in $cal(A)$.

=== Flow Function and Conservation

The _trajectory flow function_ $F: cal(T) -> RR_(>= 0)$ assigns a non-negative value to each possible trajectory @malkin2023trajectorybalanceimprovedcredit.
From this flow function, two important quantities are derived:

+ *State flow*: For any state $s$, its flow is the sum of flows through all trajectories passing through it: $ F(s) = sum_(s in tau) F(tau). $

+ *Edge flow*: For any action (edge) $s -> s'$, its flow is the sum of flows through all trajectories using that edge: $ F(s -> s') = sum_(tau = (... -> s -> s' -> ...)) F(tau). $

These flows must satisfy a conservation principle known as the _flow matching constraint_:

#attention([Flow Matching])[
    For any non-terminal state $s$, the total incoming flow must equal the total outgoing flow: $ F(s) & = sum_((s'' -> s) in cal(A)) F(s'' -> s) & = sum_((s -> s') in cal(A)) F(s -> s'). $
]

=== Markovian Flow

The flow function induces a probability distribution over trajectories.
Given a flow function $F$, we define $P(tau) = 1 / Z F(tau)$ @malkin2023trajectorybalanceimprovedcredit, where $Z = F(s_0) = sum_(tau in cal(T)) F(tau)$ is the _partition function_ --- i.e., the total flow through the network.

#attention([Markovian Flow])[
    A flow is _Markovian_ when it can be factored into local decisions at each state.
    This occurs when the following criteria are met @malkin2023trajectorybalanceimprovedcredit:

    + Forward policies $P_F (-|s)$ over children of each non-terminal state s.t. $ & P(tau = (s_0 -> ... -> s_n)) & = product_(t = 1)^n P_F (s_t|s_(t - 1)). $

    + Backward policies $P_B (-|s)$ over parents of each non-initial state s.t. $ & P(tau = (s_0 -> ... -> s_n)|s_n = x) & = product_(t = 1)^n P_B (s_(t - 1)|s_(t)). $
]

The Markovian property allows us to decompose complex trajectory distributions into simple local decisions, making learning tractable while maintaining the global flow constraints.

== GFlowNets

GFlowNets are an approach to learning policies that sample from desired probability distributions @bengio2021flownetworkbasedgenerative.
They frame the learning process as discovering a flow function that makes the probability of generating any particular object proportional to its reward.

Given a reward function $R: cal(X) -> RR_(>= 0)$ defined over the set of terminal states $cal(X)$, GFlowNets aim to approximate a Markovian flow $F$ on the graph $G$ s.t. $F(x) = R(x)$ for all $x in cal(X)$.
We will make use of the following definition of a GFlowNet.

#attention([GFlowNet])[
    @malkin2023trajectorybalanceimprovedcredit defines a GFlowNet as any learning algorithm that discovers flow functions matching terminal state rewards, consisting of: 

    + A model that outputs:
        - Initial state flow $Z = F(s_0)$;
        - Forward action distributions $P_F (-|s)$ for non-terminal states.

    + An objective function that, when globally minimized, guarantees $F(x) = R(x)$ for all terminal states.

    _Example: In molecular design, this ensures that high-reward molecules are genered more frequently, while maintaining diversity through exploration of multiple pathways._
]

The power of GFlowNets lies in their ability to handle situations where multiple action sequences can lead to the same terminal state --- a common scenario in real-world applications like molecular design or image synthesis.
Unlike traditional RL methods that focus on finding a single optimal path, GFlowNets learn a distribution over all possible paths directly proportional to their rewards.

=== Learning Process

The learning process of GFlowNets involves iteratively improving both flow estimates and the policies.
The forward policy of a GFlowNet can sample trajectories from the learned Markovian flow $F$ by sequentially selecting actions according to $P_F (-|s)$.
When the training converges to a global minimum of the objective function, this sampling process guarantees that $P(x) prop R(x)$.
That is, the probability of generating any terminal state $x$ is proportional to its reward $R(x)$.
This property makes GFlowNets particularly well-suited for:

+ *Diverse Candidate Generation:* Rather than converging to a single solution, GFlowNets maintain a distribution over solutions weighted by their rewards.

+ *Multi-Modal Exploration:* The flow-based approach naturally handles problems with multiple distinct solutions of similar quality.

+ *Compositional-Structure Learning:* By learning flows over sequences of actions, GFlowNets can capture and generalize compositional patterns in the solution space.

To achieve this, GFlowNets employ various training objectives, with _trajectory balance_ @malkin2023trajectorybalanceimprovedcredit being one such particularly effective objective.

=== Trajectory Balance

Trajectory balance focuses on ensuring consistency across entire trajectories, instead of matching flows at every state (which can be computationally expensive).

#attention([Trajectory Balance])[
    A principle that ensures the probability of generating a trajectory matches its reward by maintaining consistency between forward generation and backward reconstruction probabilities.
]

Consider a Markovian flow $F$ that induces a distribution $P$ over trajectories according to $P(tau) = 1/Z F(tau)$.
The forward policy $P_F$ and backward policy $P_B$ must satisfy the following _trajectory balance constraint_ @malkin2023trajectorybalanceimprovedcredit $ Z product_(t = 1)^n P_F (s_t|s_(t - 1)) = F(x) product_(t = 1)^n P_B (s_(t - 1)|s_t). $

That is to say, the probability of constructing a trajectory forward should match the probability of reconstructing it backward, scaled by the appropriate rewards.


=== Trajectory Balance as an Objective

To convert the trajectory balance function into a training objective, we introduce a parametrized model with parameters $theta$ that outputs:

+ A forward policy $P_F (-|s; theta)$;
+ A backward policy $P_B (-|s; theta)$;
+ A scalar estimate $Z_theta$ of the partition function.

For any complete trajectory $tau = (s_0 -> ... -> s_n = x)$, we define the _trajectory balance loss_ as $ cal(L)_"TB" (tau) = (log (Z_theta product_(t=1)^n P_F (s_t|s_(t-1); theta)) / (R(x) product_(t=1)^n P_B (s_(t-1)|s_t; theta)))^2. $ <trajectory_balance_loss>

This loss captures how well our model satisfies the trajectory balance constraint.
When the loss approaches zero, our model has learned to generate samples proportional to their rewards.
In practice, we compute this loss in the log domain to avoid numerical stability, as suggested by @malkin2023trajectorybalanceimprovedcredit: $ cal(L)_"TB" (tau) = (log Z_theta + log sum_(t = 1)^n P_F (s_t|s_(t - 1); theta) - log R(x) - log sum_(t = 1)^n P_B (s_(t - 1)|s_t; theta))^2. $

@malkin2023trajectorybalanceimprovedcredit also remarks that a simplificatoin of @trajectory_balance_loss occurs in tree-structured state spaces (when $G$ is a directed tree), where each state has exactly one parent.
In such cases, the backward policy becomes deterministic ($P_B = 1$), reducing the loss function to $ cal(L)_"TB" (tau) = (log (Z_theta product_(t = 1)^n P_F (s_t|s_(t - 1); theta)) / (R(x)))^2, $ which can be exploited for the n-chain environment.

The model is trained by sampling trajectories from a training policy $pi_theta$ --- typically a tempered version of $P_F (-|-; theta)$ to encourage exploration --- and updating parameters using stochastic gradient descent: $theta <- theta - alpha EE_(tau ~ pi_theta) nabla_theta cal(L)_"TB" (tau).$

== Markov Decision Processes

The concept of the Markov Decision Process (MDP) @martin1994markovdecisionprocesses is fundamental in reinforcement learning and provides a model for sequential decision-making under uncertainty.

#attention([Markov Decision Process @martin1994markovdecisionprocesses])[
A tuple $cal(M) := angle.l cal(S), cal(A), P_0, P_S, P_R, gamma angle.r$ where:

- $cal(S)$ is the set of states;
- $cal(A)$ is the set of actions;
- $P_0 in cal(P)(cal(S))$ is the initial state distribution;
- $P_S : cal(S) times cal(A) -> cal(P)(cal(S))$ is the state transition distribution;
- $P_R : cal(S) times cal(A) -> cal(P)(RR)$ is the reward distribution;
- $gamma in [0, 1]$ is the discount factor.
]

At each timestep $t$, an agent observes its current state $s_t in cal(S)$ and selects an action $a_t in cal(A)$ according to some policy $pi: cal(S) -> cal(P)(cal(A))$.
The environment then transitions to a new state $s_(t+1)$ according to the transition distribution $P_S (s_t, a_t)$ and provides a reward $r_t$ sampled from $P_R (s_t, a_t)$.
The agent's objective is to find a policy $pi$ that maximizes the expected sum of discounted future rewards $ J^pi := EE_(tau ~ P^pi)[sum_(t=0)^infinity gamma^t r_t], $ where $tau = (s_0, a_0, r_0, s_1, ...)$ represents a trajectory through the environment and $P^pi$ is the distribution over trajectories induced by following policy $pi$.
An optimal policy $pi^* in Pi^* := op("arg max", limits: #true)_(pi) J^pi$ can be found through the optimal value function $V^*: cal(S) -> RR$ or the optimal action-value function $Q^*: cal(S) times cal(A) -> RR$, which satisfy the Bellman optimality equations @bellman1957dynamicprogramming @martin1994markovdecisionprocesses
$ V^* (s) = max_(a in cal(A)) Q^* (s, a), $
$ Q^* (s, a) = EE_(r, s' ~ P_(R,S)(s,a))[r + gamma max_(a' in cal(A)) Q^* (s', a')]. $

This framework serves as the building block for more sophisticated models like Contextual MDPs and Bayesian approaches to reinforcement learning, as described in the following sections.

== Contextual Reinforcement Learning

In contextual RL, we use the concept of a Contextual MDP.

#attention([Contextual MDP])[
    A Markov Decision Process augmented with a context variable that determines the specific dynamics of the environment @hallak2015contextualmarkovdecisionprocesses.
    This allows us to model uncertainty about the true environment through uncertainty about the context.
]

In a Contextual Markov Decision Process (CMDP), we work in an infinite-horizon, discounted setting where a context variable $phi in Phi subset.eq RR^d$ indexes specific MDPs.
Formally, we describe this as $ cal(M)(phi) := angle.l cal(S), cal(A), P_0, P_S (s, a, phi), P_R (s, a, phi), gamma angle.r. $

where the context $phi$ parametrizes both:

- A transition distribution $P_S (s, a, phi)$ determining how states evolve;
- A reward distribution $P_R (s, a, phi)$ determining the rewards received.

The agent has complete knowledge of the following aspects of the environment:

- The state space $cal(S) subset RR^n$;
- The action space $cal(A)$;
- The initial state distribution $P_0$;
- The discount factor $gamma$.

However, the agent does not know the true context $phi^*$ that determines the actual dynamics and rewards.

=== Policies and Histories

In contextual RL, an agent follows a _context-conditioned policy_ $pi: cal(S) times Phi -> cal(P)(cal(A))$, selecting actions according to $a_t ~ pi(s_t, phi)$.
As the agent interacts with the environment, it accumulates a history of experiences $h_t := {s_0, a_0, r_0, s_1, a_1, r_1, ..., a_(t-1), r_(t-1), s_t}$.
This history belongs to a state-action-reward product space $cal(H)_t$ and follows a context-conditioned distribution $P^pi_t (phi)$ with density $ p^pi_t (h_t|phi) = p_0(s_0) product_(i=0)^t pi(a_i|s_i, phi) p(r_i, s_(i+1)|s_i, a_i, phi). $

=== Optimization Objective

The agent's goal in a CMDP is to find a policy that optimizes the expected discounted return $ J^pi (phi) = EE_(tau_infinity ~ P^pi_infinity (phi))[sum_(t=0)^infinity gamma^t r_t]. $

An optimal policy $pi^*(dot, phi)$ belongs to the set $Pi^*_Phi (phi) := op("arg max", limits: #true)_(pi in Pi_Phi) J^pi (phi)$.
With this, we define the optimal Q-function $Q^* (h_t, a_t, phi)$.

#attention("Optimal Q-Function")[
For an optimal policy $pi^*$, _the optimal Q-function_ $Q^* : cal(S) times cal(A) times Phi -> RR$ satisfies the Bellman equation $ cal(B)^* [Q^*] (s_t, a_t, phi) = Q^*(s_t, a_t, phi), $ where $cal(B)^*$ is the optimal Bellman operator defined as $ cal(B)^* [Q^*] (s_t, a_t, phi) := EE_(r_t, s_(t+1) ~ P_(R, S) (s_t, a_t, phi)) [r_t + max_(a' in cal(A)) Q^* (s_(t+1), a', phi)]. $
]

=== The Learning Challenge

When an agent has access to the true MDP $cal(M)(phi^*)$, finding an optimal policy becomes a _planning problem_.
However, in real-world scenarios, agents typically lack access to the true transition dynamics and reward functions.
This transforms the task into a _learning problem_, where the agent must balance:

+ _Exploration_: learning about the environment's dynamics through interaction;
+ _Exploitation_: using current knowledge to maximize rewards.

This tension --- known as the exploration/exploitation dilemma --- remains one of the core challenges in reinforcement learning.
As we'll see in the next section, Bayesian approaches offer a principled framework for addressing this challenge.


== Bayesian Reinforcement Learning

In the Bayesian approach to RL, rather than viewing uncertainty as a problem to be eliminated, it becomes an integral part of the decision-making process --- something to be reasoned about systematically @Ghavamzadeh_2015.

#attention("Bayesian Epistemology")[
    A framework that characterizes uncertainty through probability distributions over possible worlds.
    In reinforcement learning, this means maintaining distributions over possible MDPs, updated as new evidence arrives @Ghavamzadeh_2015.
]

=== From Prior to Posterior

The Bayesian learning process starts with a _prior distribution_ $P_Phi$ representing our initial beliefs about the true context $phi^*$ before any observations.
As the agent interacts with the environment, it accumulates a history of experiences $h_t$ and updates these beliefs through Bayesian inference, forming a _posterior distribution_ $P_Phi (h_t)$.

This history-dependent posterior in Bayesian RL differentiates it from traditional RL approaches.

#attention("History-Conditioned Policies")[
    Unlike traditional RL policies that map states to actions, Bayesian policies operate on entire histories, defining a set of history-conditioned policies $Pi_cal(H) := {pi: cal(H) -> cal(P)(cal(A))}$, where $cal(H) := { cal(H)_t|t >= 0 }$ denotes the set of all histories @Ghavamzadeh_2015.
]

Where the prior $P_Phi$ represents our initial uncertainty (the special case where $h_t = emptyset$), the posterior $P_Phi (h_t)$ captures our refined beliefs after observing interactions with the environment.
This allows us to reason about future outcomes by marginalizing across all possible MDPs according to our current uncertainty.

=== The Bayesian Perspective on Transitions

The power of the Bayesian approach stems from how it handles state transitions.
Instead of committing to a single model of the environment, it maintains a distribution over possible transitions through the _Bayesian state-reward transition distribution_ $ P_(R, S) (h_t, a_t) := EE_(phi ~ P_Phi (h_t)) [P_(R, S) (s_t, a_t, phi)]. $

This distribution lets us reason about future trajectories using the _prior predictive distribution_ $P^pi_t$ with density $ p^pi_t (h_t) = p_0 (s_0) product_(i = 0)^t pi (a_i|h_i) p (r_i, s_(i + 1)|h_i, a_i). $

The belief transition distribution $P_cal(H) (h_t, a_t)$ captures how our beliefs evolve with new observations, with density $ p_cal(H)(h_(t+1)|h_t, a_t) = p(s_(t+1), r_t|h_t, a_t). $

This formulation leads to the definition of the Bayes-adaptive MDP (BAMDP) @duff2002optimallearningcomputationalprocedures $ cal(M)_"BAMDP" := angle.l cal(H), cal(A), P_0, P_cal(H)(h,a), gamma angle.r. $

=== Natural Resolution of the Exploration Dilemma

An interesting aspects of the Bayesian framework is how it naturally resolves the exploration-exploitation dilemma @duff2002optimallearningcomputationalprocedures @Ghavamzadeh_2015.
Rather than treating exploration as a separate mechanism, it emerges naturally from the optimization of expected returns under uncertainty $ J^pi_"Bayes" := EE_(h_infinity ~ P^pi_infinity)[sum_(i=0)^infinity gamma^i r_i]. $

A Bayes-optimal policy achieves perfect balance between exploration and exploitation because:

+ It accounts for uncertainty through the posterior at each timestep;
+ It considers how this uncertainty will evolve in the future;
+ It weights future information gain by the discount factor $gamma$.

=== The Optimal Bayesian Q-Function

For a Bayes-optimal policy $pi^*$, we can define the optimal Bayesian Q-function as $Q^* (h_t, a_t) := Q^(pi^*_"Bayes") (h_t, a_t)$.
This Q-function satisfies the optimal Bayesian Bellman equation $ Q^* (h_t, a_t) = cal(B)^* [Q^*] (h_t, a_t), $ where $cal(B)^* [Q^*]$ is the optimal Bayesian Bellman operator $ cal(B^*) [Q^*] (h_t, a_t) := EE_(h_(t + 1) ~ P_cal(H) (h_t, a_t)) [r_t + gamma max_a' Q^* (h_(t + 1), a')]. $

== Bayesian Exploration Networks

Model-free reinforcement learning takes a different approach to learning optimal behaviors compared to model-based methods.
Rather than explicitly modeling the environment's dynamics, model-free approaches attempt to learn optimal policies from experience.
Bayesian Exploration Networks (BENs) extend this idea into the Bayesian realm by characterizing uncertainty in the Bellman operator itself, instead of in the environment's transition dynamics @fellows2024bayesianexplorationnetworks.

#attention("Model-Free vs Model-Based")[
    While model-based approaches maintain explicit probabilistic models of the environment's dynamics, model-free methods like BEN directly learn mappings from states to values or actions @DayanPeter2008RlTG.
    This can be more computationally efficient but requires careful handling of uncertainty.
]

=== The Bootstrapping Perspective

Instead of modeling the full complexity of state transitions, we can use bootstrapping to estimate the optimal Bayesian Bellman operator directly.
Given samples from the true reward-state distribution $r_t, s_(t+1) ~ P^*_(R,S) (s_t, a_t)$, we estimate $b_t = beta_omega (h_(t+1)) := r_t + gamma max_a' Q_omega (h_(t+1), a').$

This bootstrapping process can be viewed as a transformation of variables --- mapping from the space of rewards and next states to a single scalar value.
This significantly reduces the dimensionality of the problem while preserving the essential information needed for learning optimal policies @fellows2024bayesianexplorationnetworks.

#attention("Bootstrapped Distribution")[
    The samples $b_t$ follow what we call the Bellman distribution $P^*_B (h_t, a_t; omega)$, which captures the distribution of possible Q-value updates @fellows2024bayesianexplorationnetworks.
    This distribution encapsulates both the environment's inherent randomness and our uncertainty about its true nature.
]

=== Sources of Uncertainty

When predicting future Q-values, BEN distinguishes between two types of uncertainty.

+ *Aleatoric Uncertainty*: The inherent randomness in the environment's dynamics that persists even with perfect knowledge.

    _Example: Rolling a fair die --- this uncertainty cannot be reduced with more data._

+ *Epistemic Uncertainty*: Our uncertainty about the true Bellman distribution itself.
    This represents our lack of knowledge about the environment and can be reduced through exploration and learning.

    _Example: Determining whether a die is fair --- this uncertainty is can be reduced with more data._

This separation of uncertainties allows BEN to distinguish between what is fundamentally unpredictable (aleatoric) and what can be learned through exploration (epistemic), leading to more efficient learning strategies @fellows2024bayesianexplorationnetworks.

=== Network Architecture

BEN implements this uncertainty handling through three neural networks @fellows2024bayesianexplorationnetworks:
+ *Recurrent Q-Network*:

    At its core, BEN uses a recurrent neural network (RNN) to approximate the optimal Bayesian Q-function.
    The Q-network processes the entire history of interactions.
    We denote the output at timestep $t$ as $q_t = Q_omega (h_t, a_t) = Q_omega (hat(h)_(t-1), o_t)$, where $h_t$ represents the history up to time $t$, $a_t$ is the action, $hat(h)_(t-1)$ is the recurrent encoding of previous history, and $o_t$ contains the current observation tuple ${r_(t-1), s_t, a_t}$.
    By conditioning on history rather than just current state, BENs can capture how uncertainty evolves over time, making it capable of learning Bayes-optimal policies @fellows2024bayesianexplorationnetworks.

+ *Aleatoric Network*:

    The aleatoric network models the inherent randomness in the environment.
    It uses normalizing flows to transform a simple base distribution (such as a standard Gaussian) into a more complex distribution $P_B (h_t, a_t, phi; omega)$ over possible next-state Q-values, representing the aleatoric uncertainty in the Bellman operator, by applying the transformation $b_t = B(z_"al", q_t, phi)$ @fellows2024bayesianexplorationnetworks, where 

    - $z_"al" in RR ~ P_"al"$ is a base variable with a zero-mean, unit variance Gaussian $P_"al"$;
    - $q_t$ is the Q-value from the recurrent network;
    - and $phi$ and $omega$ represent the network parameters.

    We optimize the parameters $omega$ of the recurrent Q-network and the aleatoric network using the Mean Squared Bayesian Bellman Error (MSBBE), which satisfies the optimal Bayesian Bellman equation for our Q-function approximator.

+ *Epistemic Network*:

    The epistemic network captures our uncertainty about the environment itself.
    The network maintains a dataset of bootstrapped samples $cal(D)_omega (h_t) := {(b_i, h_i, a_i)}_(i=0)^(t-1)$ collected from interactions with the environment.
    Each tuple in this dataset consists of

    - a bootstrapped value estimate $b_i$;
    - the history at that timestep $h_i$;
    - the action taken $a_i$.

    Given this dataset, we would ideally compute the posterior distribution $P_Phi (cal(D)_omega (h_t))$ representing our refined beliefs about the environment after observing these samples.
    However, computing this posterior directly is typically intractable for complex environments @fellows2024bayesianexplorationnetworks.
    Instead, BEN employs normalizing flows for variational inference.

    The epistemic network learns a tractable approximation $P_psi$ parametrized by $psi in Psi$ that aims to capture the essential characteristics of the true posterior.
    We optimize this approximation by minimizing the KL-divergence between our approximation and the true posterior: $ "KL"(P_psi || P_Phi (cal(D)_omega (h_t))). $
    
    This optimization is performed indirectly by maximizing the Evidence Lower Bound (ELBO) $"ELBO"(psi; h, omega)$, which is equivalent as proved by @fellows2024bayesianexplorationnetworks.


=== Training Process

The network is trained through a dual optimization process:

+ *MSBBE Optimization:* The Mean Squared Bayesian Bellman Error (MSBBE) is computed as the difference between the predictive optimal Bellman operator $B^+ [Q_omega]$ and $Q_omega$ @fellows2024bayesianexplorationnetworks: $ "MSBBE"(omega; h_t, psi) := norm(B^+ [Q_omega] (h_t, a_t) - Q_omega (h_t, a_t))^2_rho, $ which is minimized to learn the parametrisation $omega^*$, satisfying the optimal Bayesian Bellman equation for our Q-function approximator, with $rho$ being an arbitrary sampling distribution with support over $cal(A)$.
    
    The predictive optimal Bellman operator can be obtained by taking expectations over variable $b_t$ using the predictive optimal Bellman distribution $P_B (h_t, a_t; omega)$: $ B^+ [Q_omega] (h_t, a_t) := EE_(b_t ~ P_B (h_t, a_t; omega)) [b_t], $ where $P_B (h_t, a_t; omega) = EE_(phi ~ P_Phi (cal(D)_omega (h_t))) [P_B (h_t, a_t, phi; omega)]$.

    This gives rise to a nested optimisation problem, as is common in model-free RL @fellows2024bayesianexplorationnetworks, which can be solved using two-timescale stochastic approximation @borkar2008stochasticapproximationadynamicalsystemsviewpoint.
    In the case of BEN, we update the epistemic network parameters $psi$ using gradient descent on an asymptotically faster timescale than the function approximator parameters $omega$ to ensure convergence to a fixed point, as propposed by @fellows2024bayesianexplorationnetworks.

+ *ELBO Optimization:* The Evidence Lower BOund (ELBO) serves as the optimization objective for training BEN's epistemic network.
    While minimizing the KL-divergence $"KL"(P_psi || P_Phi (cal(D)_omega (h_t)))$ directly would give us the most accurate approximation of the true posterior, computing this divergence is typically intractable.
    Instead, we can derive and optimize the ELBO, which provides a tractable lower bound on the model evidence @fellows2024bayesianexplorationnetworks.
    
    By applying Baye's rule on this KL-divergence, @fellows2024bayesianexplorationnetworks derives $ "ELBO"&(psi; h_t, omega) \ &:= EE_(z_"ep" ~ P_"ep") [ sum_(i=0)^(t-1) ( B^(-1)(b_i, q_i, phi)^2 - log bar.v partial_b B^(-1)(b_i, q_i, phi) bar.v ) - log p_Phi (phi) ], $ where $phi = t_psi (z_"ep")$ and:
    
    - $z_"ep"$ is drawn from the base distribution $P_"ep"$ (a standard Gaussian $cal(N)(0, I^d)$);
    - $B^(-1)$ is the inverse of the aleatoric network's transformation;
    - $partial_b B^(-1)$ is the Jacobian of this inverse transformation;
    - $t_psi$ represents the epistemic network's transformation.
    
    #attention("Jacobian Term")[
        The term $partial_b B^(-1)$ accounts for how the epistemic network's transformation changes the volume of probability space.
        This is important for maintaining proper probability distributions when using normalizing flows @pmlr-v37-rezende15.
    ]
    
    The ELBO objective breaks down into three key components:
    
    + A reconstruction term $B^(-1)(b_i, q_i, phi)^2$ that measures how well our model can explain the observed Q-values;
    + A volume correction term $log|partial_b B^(-1)(b_i, q_i, phi)|$ that accounts for the change in probability space;
    + A prior regularization term $log p_Phi (phi)$ that encourages the approximated posterior to stay close to our prior beliefs.
    
    By minimizing the ELBO, we obtain an approximate posterior that balances accuracy with computational tractability, allowing BEN to maintain and update its uncertainty estimates efficiently during learning @fellows2024bayesianexplorationnetworks.

#attention("Training Dynamics")[
    The two optimization processes occur at different timescales, with epistemic updates happening more frequently than the Q-network updates.
    This separation ensures stable convergence while maintaining the ability to adapt to new information @borkar2008stochasticapproximationadynamicalsystemsviewpoint.
]

With this architecture, BEN can learn truly Bayes-optimal policies while maintaining the computational efficiency of model-free methods @fellows2024bayesianexplorationnetworks.
This makes it particularly well-suited for environments with sparse, delayed rewards where efficient exploration is important.

= Theoretical Framework <theoretical_framework>

In this section, we develop a theoretical framework for comparing GFlowNets and Bayesian Exploration Networks (BENs) in environments with delayed and sparse rewards.
Our goal is to establish precise criteria for evaluating these different approaches to exploration and uncertainty handling.

== Problem Formulation

Consider an environment with delayed rewards characterized by

- *Reward Delay:* The temporal gap $T_"reward"$ between an action and its corresponding reward signal
    . We formally define this as $ T_"reward" := min{t | s_t in cal(X), r_t != 0}. $
    In our n-chain environment, $T_"reward"$ corresponds to the chain length.

- *Reward Sparsity:* The proportion $rho$ of state-action pairs that yield non-zero rewards: $ rho := abs({(s, a) in cal(S) times cal(A) : EE[R(s, a)] != 0}) / abs(cal(S) times cal(A)), $ where $R(dot)$ is some reward distribution.

These characteristics create distinct challenges, as discussed, for reinforcement learning algorithms.

+ The _temporal credit assignment problem_ becomes more severe with increasing $T_"reward"$.
+ The _exploration efficiency_ becomes critical as $rho$ decreases.
+ The _signal-to-noise ratio_ in value estimation deteriorates with both $T_"reward"$ and $rho$.

=== Value Propagation Mechanisms

The algorithms differ in how they handle value propagation. GFlowNet value propagation maintains consistency between forward and backward flows through the trajectory balance constraint $Z product_(t = 1)^n P_F (s_t|s_(t - 1)) = F(x) product_(t = 1)^n P_B (s_(t - 1)|s_t).$

BEN value propagation directly models the distribution of bootstrapped values through the estimated Bellman operator $b_t = r_t + gamma max_a' Q_omega (h_(t+1), a').$

=== Uncertainty Representation

Both approaches maintain uncertainty estimates but through different mechanisms:

- GFlowNets implicitly capture uncertainty through the learned flow distribution;
- BENs explicitly separate aleatoric and epistemic uncertainty.

This leads to our central hypothesis.

#attention("Hypothesis")[
In environments with highly delayed rewards (large $T_"reward"$), BEN's explicit uncertainty decomposition leads to more efficient learning compared to GFlowNets, particularly in early training stages.
]

However, this advantage diminishes as $T_"reward"$ decreases.
This hypothesis is supported by the following three observations.

+ BEN's direct modeling of the Bellman operator allows for faster value propagation.
+ The explicit separation of uncertainty types enables more targeted exploration.
+ GFlowNets must learn complete trajectories before gaining signal about reward structure.

=== Analytical Framework <analytical_framework>

To evaluate our hypothesis about the relative performance of GFlowNets and BENs in delayed reward environments, we establish three metrics that capture different aspects of learning and exploration efficiency.

+ *Sample Efficiency:* Measures how quickly each algorithm converges to optimal behavior through their respective loss functions.

    For GFlowNets, we track the trajectory balance loss $cal(L)_"TB" (tau) = (log Z_theta + log sum_(t = 1)^n P_F (s_t|s_(t - 1); theta) - log R(x) - log sum_(t = 1)^n P_B (s_(t - 1)|s_t; theta))^2$.

    While for BEN, we monitor the Mean Squared Bayesian Bellman Error $"MSBBE"(omega; h_t, psi) = norm(B^+ [Q_omega] (h_t, a_t) - Q_omega (h_t, a_t))^2_rho$.

    These metrics allow us to quantify learning progress and compare convergence rates between algorithms as a function of reward delay $T_"reward"$.

+ *Distribution Matching:* Evaluates how well the learned policy matches the true underlying reward structure.

    In our n-chain environment, where terminal states are guaranteed to be reached, we measure the KL-divergence between the true terminal state distribution $P$ (determined by rewards) and the empirical distribution $Q$ generated by each algorithm $"KL"(P || Q)$.
    This metric is particularly relevant for GFlowNets, as they explicitly aim to learn a sampling distribution proportional to the reward function.

+ *Exploration Efficiency:* Captures how effectively each algorithm explores the state space before converging to optimal behavior.

    We introduce two complementary metrics.

    - *State Coverage Ratio:* Measures the proportion of the state space explored over time as $ E(t) := abs(S_("visited")(t)) / abs(S), $ where $S_("visited")(t)$ represents the set of unique states visited up to time $t$.

    - *Time-to-First-Success:* Quantifies initial exploration effectiveness as $ T_("success") := min{t | s_t in cal(X), r_t > 0}. $

    This metric becomes increasingly important as reward delay $T_"reward"$ grows, as it indicates how quickly each algorithm can discover successful trajectories in sparse reward settings.

Combining these metrics, we construct a evaluation framework that addresses three important aspects of performance:

+ Learning efficiency through loss convergence analysis;
+ Policy quality through distribution matching;
+ Exploration effectiveness through coverage and discovery time.

This framework allows us to investigate how the advantage of BEN's explicit uncertainty decomposition versus GFlowNet's flow-based approach vary with reward delay $T_"reward"$ and sparsity $rho$.
In particular, we can test our hypothesis that BEN's advantages become more pronounced as $T_"reward"$ increases by examining the correlation between reward delay and relative performance across the mentioned metrics.

= Experimental Design <experimental_design>

We implement a modified n-chain environment that serves as a testbed for studying delayed rewards.
This environment presents properties that make it particularly suitable for our analysis.

#attention("N-Chain Environment")[
    A sequential decision-making environment with a branching structure, where rewards are only received at terminal states.
]

Parameters of the n-chain environment include a chain length $n$, controlling reward delay $T_"reward"$, a branching factor $b$, affecting exploration complexity, and terminal state rewards, determining optimal distributions.
Adjusting the chain length $n$ and branching factor $b$ also allow us to control the reward sparsity $rho$ by proxy, as a longer chain involves more states, increasing sparsity.
Similarly, increasing the branching factor $b$ introduces more branches, again inreasing the number of states and, thus, the sparsity $rho$ as well.

The environment consists of three main components:

+ *State Space:* A chain of length $n$ with a branching point at the middle ($floor(n/2)$), creating multiple possible trajectories.
    This results in a total of $floor(n / 2) + b (n - floor(n / 2))$ possible states, and exactly $b$ terminal states.

+ *Action Space:* At each state, an agent can move forward with the $"FORWARD"$ action, or stay in terminal states with the $"TERMINAL_STAY"$ action.
    At the split point, the agent must choose a branch using the $"BRANCH"_i$ action, where $i in {1, ..., b}$.
    This results in a total of $2 + b$ actions.

+ *Reward Structure:* Generally, a reward function $R: cal(X) -> RR$ is defined over terminal states $x in cal(X)$, creating a natural target distribution for sampling and clear optimal policies.
    For GFlowNets, the reward function is further constrained to the domain $RR_(>0)$, yielding a reward function $R: cal(X) -> RR_(>0)$.

This design creates a sparse reward landscape --- agents must execute sequences of $floor(n/2) - 1$ actions before reaching the branch point, where the chosen branch determines the final reward, followed by another $floor(n/2)$ actions to reach any terminal state.
This structure allows us to precisely control both reward delay $T_"reward"$ and sparsity $rho$.

== Evaluation Protocol

We evaluate each algorithm through the following experiment.

- *Base Configuration:*
    - Three terminal states with fixed rewards;
    - GFlowNet: 
        - Rewards: ${10, 20, 70}$;
        - Exploration factor $epsilon = 0.1$;
    - BEN: 
        - Rewards: ${-500, 10, 200}$;
        - Discount factor $gamma = 0.9$.

- *Delay Variation Studies:*
    - Chain lengths $n in {3, 5, 7, 9, 11}$;
    - Keeping terminal rewards fixed;
    - Measuring performance vs. delay $T_"reward"$.

For each configuration, we conduct 10 independent trials with different random seeds to minimize the impact of statistical variance.
We then apply the framework discussed in @analytical_framework on the results of these three configurations for analysis.


== Implementation Details

=== Environment Setup

Our implementation of the n-chain environment creates a decision space that enables precise control over reward delay and sparsity.
The environment is implemented as a deterministic MDP, where each state is encoded as a composite tensor of shape `[n*3 + 4]`, consisting of a one-hot position encoding of length 3n to account for all three branches, as well as a one-hot branch encoding of length 4 (pre-split + 3 possible branches).

The state space is managed through a `NChainState` class that tracks three attributes:

+ _Position:_ An integer in [0, n-1] indicating location in the chain;
+ _Branch:_ An integer flag (-1 for pre-split, {0,1,2} for branch selection);
+ _Chain Length:_ The parameter n that determines the delay between actions and rewards.

The action space is managed through a `NChainAction` enum and consists of the following five distinct actions:

- `TERMINAL_STAY`: Available only in terminal states;
- `FORWARD`: For progression along the chosen path;
- `BRANCH_0`, `BRANCH_1`, `BRANCH_2`: Available only at the split point.

The environment enforces strict action masking through a get_valid_actions method that returns only legitimate actions for each state.
This creates three distinct decision phases:

+ Pre-split: Only `FORWARD` actions are valid;
+ Split-point: Only `BRANCH_i` actions are valid, for $i in {0, 1, 2}$;
+ Post-split: `FORWARD` until terminal, then `TERMINAL_STAY`.

For interaction with the environment, the implementation provides methods like `step`, `reset`, as well as utility methods like state-to-tensor conversions.
This approach allows for systematic variation of both reward delay through the chain length $n$, and reward sparsity through the ratio of rewarding states to total states.

=== Network Architectures

The GFlowNet implementation consists of three primary components working in concert to learn flow-matching policies:

+ _State Encoder:_ A multi-layer perceptron that processes state tensors of shape `[batch_size, state_dim]` into a learned encoding of shape `[batch_size, hidden_dim]`.
    This encoding captures the essential features of each state necessary for policy decisions.

+ _Forward Policy Network:_ A single dense layer that transforms the state encoding into a forward policy distribution over actions, outputting tensors of shape `[batch_size, num_actions]`.
    This network determines the probabilities of taking each possible action from the current state.

+ _Backward Policy Network:_ For the n-chain environment, this component is simplified due to the tree structure of the state space.
    Since each non-initial state has exactly one parent, the backward policy becomes deterministic, requiring only a lightweight network layer to maintain architectural symmetry.

Additionally, we maintain a scalar parameter representing $log Z$ (the partition function), which is important for our flow matching approach and is learned alongside the network parameters.

The BEN implementation is heavily based on the implementation provided by @fellows2024bayesianexplorationnetworks, but is adapted for the n-chain environment.
It comprises three interacting networks:

+ _Q-Bayes Network:_ A recurrent neural network that processes observation tuples (state, action, reward) to generate Q-values and maintain a history encoding.
    The network accepts inputs of shape `[batch_size, state_dim + action_dim + reward_dim]` and outputs both Q-values `[batch_size, num_actions]` and a RNN hidden state `[batch_size, rnn_hidden_dim]` for further processing.

+ _Aleatoric Network:_ Implements a normalizing flow to model the inherent randomness in Bellman updates through:
   - A conditioning network (ConditionerMLP) @fellows2024bayesianexplorationnetworks that generates parameters for the flow based on Q-values and RNN state;
   - An inverse autoregressive flow that transforms a base distribution into the desired Bellman distribution.

+ _Epistemic Network:_ Another normalizing flow that captures uncertainty about the environment itself, transforming a base variable $z_"ep" in RR^d$ into the parameter space that defines our beliefs about the environment.

=== Training Procedures

Training of the GFlowNet follows a tempered exploration strategy where:

+ The forward policy is "softened" during training using an $epsilon$-greedy approach with $epsilon = 0.1$, allowing for off-policy exploration while maintaining flow-matching properties:
    #align(center)[
        ```python
        (1 - self.epsilon) * policy + self.epsilon * random_policy.
        ```
    ]

+ The trajectory balance loss is minimized using stochastic gradient descent with the Adam optimizer @kingma2017adammethodstochasticoptimization:
    #align(center)[
        ```python
        L_TB(tau) = (log_Z + sum_log_pf - log_R - sum_log_pb).pow(2),
        ```
    ]
    where `tau` represents a trajectory, `sum_log_pf` represents the sum of log forward probabilities over `tau`, and `sum_log_pb` similarly represents the sum of log backward probabilities over `tau`.

BEN employs a two-timescale optimization process:

+ _Fast Timescale:_ Updates to the epistemic network parameters $psi$ through ELBO minimization:
    #align(center)[
        ```python
        ELBO(psi; h, omega) = -log_p - torch.mean(log_q) - 1 / (time_period + 1) * prior,
        ```
    ]
    where `log_p` represents the base variable $z_"al"$ obtained from $B^(-1) (b_i, q_i, phi)^2$, `torch.mean(log_q)` represents the log Jacobian of this base variable $log |partial_b B^(-1) (b_i, q_i, phi)|$, and `1 / (time_period + 1) * prior` represents the log prior $log p_Phi (phi)$.

2. _Slow Timescale:_ Updates to the Q-network parameters $omega$ through MSBBE minimization:
    #align(center)[
        ```python
        MSBBE(omega; h_t, psi) = torch.abs((b1 - q) * (b2 - q)),
        ```
    ]
    where `b1` and `b2` represent two samples from the predictive Bellman operator $B^+ [Q_omega] (h_t, a_t)$, and `q` represents the Q-value obtained from $Q_omega (h_t, a_t)$.
    This operation is similar to a simple squared error, with the only difference being the use of two different samples, minimizing statistical variance.

This separation of timescales ensures stable convergence while maintaining the ability to adapt to new information, controlled by the discount factor $gamma$.
For details about hyperparameter selection, we refer to @hyperparameter_selection.

= Results and Analysis <results_and_analysis>

#note[
- Quantitative results
    - Performance comparisons
    - Statistical analysis
- Qualitative analysis
    - Exploration patterns
    - Learning behavior
- Discussion of findings
]

== GFlowNet

#todo[
    FACT CHECK!
        - Run the statistical significance tests and calculate difference in $mu$'s of terminal rewards
    - Find out what Cohen's d is (effect sizes)
]

#todo[
- Loss curves across different chain lengths
    - Show plot for n=3, n=7, and n=11
    - Rest in appendix
- Statistical significance tests
    - T-tests between early/mid/late stages
- Convergence time analysis
    - By what iteration does loss seem to converge?
]

===  Training Stability and Loss Dynamics

#figure(
    placement: top,
    image("figures/gflownet/mean_trajectory_balance_loss.png", width: 120%),
    caption: [Mean GFlowNet trajectory balance loss across chain lengths.],
) <mean_trajectory_balance_loss_plot>

The trajectory balance loss follows a consistent pattern of reduction across all chain lengths, with three distinct phases:

- _Early Stage:_ High loss values ($mu approx$ 6.8-7.4) with substantial variance;
- _Mid Stage:_ Significant reduction ($mu approx$ 1.2-1.7) with moderate variance;
- _Late Stage:_ Near-zero loss ($mu <$ 0.003) with minimal variance.

This progression is statistically significant across all transitions ($p < 0.001$), with particularly large effect sizes (Cohen's $d > 2.0$) between early and late stages, indicating robust convergence regardless of chain length, as seen in @mean_trajectory_balance_loss_plot.

=== Solution Quality Evolution

#figure(
    placement: top,
    image("figures/gflownet/mean_terminal_reward.png", width: 100%),
    caption: [Mean GFlowNet terminal rewards across chain lengths with a moving average (window size: 50).],
) <mean_terminal_rewards_plot>

Terminal rewards show a monotonic improvement pattern, as seen in @mean_terminal_rewards_plot.
The improvement in terminal rewards shows a consistent "two-step" enhancement, with moderate-to-large effect sizes (0.75-1.0) between consecutive stages and large effect sizes (1.4-1.9) between early and late stages.

Notably, longer chains ($n >= 9$) demonstrate larger improvements in terminal rewards, with $n = 9$ showing the most substantial gain ($Delta mu approx 16$ between early and late stages). This suggests that the algorithm becomes especially effective at optimizing longer sequences as training progresses.

The time-to-first-success is trivial for the n-chain environment: since the agent must always move forward, it is simply the chain length $n$.

==== Policy Quality Assessment

As GFlowNets aim to learn policies proportional to their reward distributions, we use the KL-divergence between the true reward distribution $R_i$ and the learned policy $P_i$ to assess the quality of the learned policy for $i in {3, 5, 7, 9, 11}$ corresponding to the different chain lengths.
The KL-divergence measurements look as follows:

- _Chain length 3:_ $"KL"(P_3 | R_3) = 0.00401$;
- _Chain length 5:_ $"KL"(P_5 | R_5) = 0.02062$;
- _Chain length 7:_ $"KL"(P_7 | R_7) = 0.01696$;
- _Chain length 9:_ $"KL"(P_9 | R_9) = 0.00385$;
- _Chain length 11:_ $"KL"(P_11 | R_11) = 0.00701$;
- _Mean chain length:_ $overline("KL"(P | R)) = 0.01049$.

Looking at these measurements, the relationship between chain length and distributional accuracy does not seem follow a simple monotonic pattern.
We conjecture that this is a consequence of the small sample size of 10 training runs per chain length, and as such, the KL values should converge given enough runs.

Looking at the average KL-divergence across all chain lengths, we conclude that the learned policy is very close to the true reward distribution.

=== Exploration-Exploitation Balance

#figure(
    placement: top,
    image("figures/gflownet/mean_exploration_ratio.png", width: 100%),
    caption: [Mean GFlowNet forward and backward entrody across chain lengths.],
) <mean_exploration_ratio_plot>

The state coverage ratio exhibits rapid convergence to optimal exploration ($1.0$) by mid-stage of only 15 iterations, as seen in @mean_exploration_ratio_plot, as well as:

- Statistically significant transition ($p < 0.005$) from early to mid stage;
- Perfect maintenance of exploration in late stage ($sigma = 0$).

This indicates that GFlowNet quickly learns to fully explore the solution space and maintains this behavior throughout training.
This also suggests that the state space is too small to pose a serious challenge for the model, which could be accomodated by increasing the number of critical decision points (by increasing the number branches in the environment).

=== Information Theoretic Measures

#note[Maybe move this to the appendix as we haven't calculated entropies for BEN.]

#figure(
    placement: top,
    image("figures/gflownet/mean_entropy.png", width: 100%),
    caption: [Mean GFlowNet forward and backward entropy across chain lengths.],
) <mean_entropy_plot>

Both forward and backward entropy demonstrate reduction across training, as seen in @mean_entropy_plot.

- _Forward Entropy:_ Shows consistent, statistically significant decreases ($p < 10^(-82)$) across all stages, with effect sizes growing with chain length;
- _Backward Entropy:_ Exhibits the most dramatic reductions among all metrics, with effect sizes ranging from 8.5 to 16.3 between early and late stages.

The relative magnitude of entropy reduction remains notably consistent across chain lengths, suggesting a scale-invariant learning process.

=== Chain Length Sensitivity

The analysis reveals some chain length-dependent effects:

- Longer chains ($n >= 9$) show higher terminal rewards in late stages;
- Convergence stability (measured by loss variance) remains consistent across chain lengths;
#maybe[- Information theoretic measures scale proportionally with chain length while maintaining similar convergence patterns.]

This suggests that the algorithm's learning dynamics remain robust across different problem scales.

In short, the statistical analysis reveals structured convergence patterns that combine rapid initial improvement with stable late-stage optimization.
The consistent statistical significance ($p < 0.001$) across multiple metrics and stages suggests that GFlowNet's learning process is both reliable and scalable across different chain lengths.

== BEN

=== Training Stability and Loss Dynamics

#figure(
    placement: top,
    image("figures/ben/q_loss.png", width: 100%),
    caption: [Mean BEN Q-learning (MSBBE) loss across chain lengths.],
) <mean_q_loss_plot_ben>

- *Q-Learning Loss*: The Q-Learning loss demonstrates a consistent and statistically significant reduction across all chain lengths, as seen in @mean_q_loss_plot_ben, with three notable characteristics.

    - _Early Stage Volatility:_ High variance in early training ($sigma$ ranging from 610 to 2948);
    - _Mid Stage Stabilization:_ Sharp reduction in both mean and variance;
    - _Late Stage Refinement:_ Convergence to stable, low values.

    Particularly noteworthy is the scale-dependent convergence rate --- longer chains ($n >= 7$) show more gradual convergence, while shorter chains achieve stability more quickly.

    The magnitude of improvement (effect size) between early and late stages increases with chain length, suggesting that longer chains require more substantial transformations in the learning process.

#figure(
    placement: top,
    image("figures/ben/epistemic_loss.png", width: 100%),
    caption: [Mean BEN epistemic (ELBO) loss across chain lengths.],
) <mean_epistemic_loss_plot_ben>

- *Epistemic Loss*: The epistemic uncertainty shows a more complex pattern than the Q-learning loss, as seen in @mean_epistemic_loss_plot_ben.

    - For $n=3$: Monotonic decrease (-2257.74 mean difference, $p < 0.001$);
    - For $n=5$: Sharp initial drop followed by gradual decline;
    - For $n in {7, 9, 11}$: Non-monotonic behavior with occasional increases.

This suggests that uncertainty management becomes more challenging with longer chains, possibly due to the expanded state space.

=== Reward Dynamics

#figure(
    placement: top,
    image("figures/ben/rewards.png", width: 100%),
    caption: [Mean BEN terminal rewards across chain lengths.],
) <mean_terminal_rewards_plot_ben>

The reward patterns exhibit rather unpromising behavior as seen in @mean_terminal_rewards_plot_ben. The statistical tests yield the following results:

- _Short Chains ($n in {3,5}$):_ No statistically significant changes between stages;
- _Medium Chains ($n in {7, 9}$):_ Temporary dip in mid-stage performance;
- _Long Chains ($n = 11$):_ Significant improvement in late stage ($p < 0.05$).

The statistical significance of stage-wise improvements increases with chain length, suggesting that longer chains benefit more from extended training.
These statistical results, however, do not seem to capture the apparent "trend"of the rewards (which is seemingly random) --- the analysis would benefit from further investigation on this part.

=== Cumulative Returns

#figure(
    placement: top,
    image("figures/ben/cumulative_returns.png", width: 100%),
    caption: [Mean BEN cumulative returns across chain lengths.],
) <mean_cumulative_returns_plot_ben>

The cumulative returns, probably the most definitive metric with regards to the quality of the learned policy, can be seen in @mean_cumulative_returns_plot_ben. They show:

- Monotonic decrease across all chain lengths;
- Larger effect sizes (-7.13 to -8.59) for longer chains;
- _Statistical Significance:_ $p < 10^(-10)$ for all chain lengths.

The relative magnitude of stage-wise change remains consistent across chain lengths.
_This is critical as it indicates that the model is not learning optimal policies._

=== Exploration-Exploitation Balance

#todo[]

BEN's exploration strategies across different chain lengths look as follows:

- _Short Chains_ ($n = {3,5}$): The exploration patterns here show remarkable consistency, with no statistically significant differences between stages ($p > 0.05$).
    This suggests that BEN quickly establishes a stable exploration strategy for smaller state spaces.
    The mean ratios hover around:
    - _Early:_ ~1.0
    - _Mid:_ ~0.95
    - _Late:_ ~1.0

Medium Chains (n=7,9)
We begin to see more nuanced behavior:

Less consistent exploration ratios
Wider variance in mid-stage exploration (σ ≈ 0.24)
No statistically significant stage-wise changes, but more volatile patterns

Long Chains (n=11)
Here's where things get particularly interesting:

Early-Mid Transition: Significant decrease (p < 0.05)
Mid-Late Transition: Significant increase (p < 0.02)
Effect Size: Moderate to large (0.74-0.90)


== Comparison Between GFlowNet and BEN

=== Learning Stability and Convergence

GFlowNet demonstrates stable convergence properties across all chain lengths, with its trajectory balance loss consistently decreasing through well defined phases.
In contrast, BEN's learning trajectory shows more volatile behavior, especially in its epistemic loss patterns.
This difference becomes more pronounced with increasing chain length, indicating that GFlowNet maintains consistent learning dynamics across problem scales, while BEN's stability deteriorates with increasing chain length.
The most striking difference appears in their convergence behaviors.

- GFlowNet achieves near-zero loss values ($< 0.003$) in its late stages across all chain lengths;
- BEN's Q-learning loss shows persistent fluctuations, particularly in longer chains;
- BEN's epistemic loss exhibits non-monotonic behavior for $n >= 7$, indicating challenges in uncertainty estimation at larger scales.

=== Reward Optimization

The contrast in reward optimization capabilities is particularly noteworthy:

- GFlowNet shows consistent improvement in terminal rewards, with larger gains in longer chains ($n >= 9$);
- BEN displays concerning patterns in both terminal rewards and cumulative returns, with monotonic decreases across all chain lengths.

The statistical significance of these differences ($p < 10^(-10)$ for BEN's declining returns) strongly suggests that GFlowNet's flow-based approach is better suited to this environment's reward structure.

= Conclusion <conclusion>

#note[
- Summary of contributions
- Key insights
- Future work directions
]

== Future Research

#note[
    Everything I think I could have done better essentially.

    - Stochastic Analysis:
        - Introducing random rewards drawn from distributions;
        - Terminal states $x in cal(X)$ rewards: $R_x ~ cal(N)(mu_x, sigma^2)$;
        - Testing robustness to uncertainty.

    - Experiment with impact of changing:
        - exploration factor $epsilon$;
        - discount factor $gamma$.
]

#bibliography("refs.bib")

#set heading(numbering: "A.1.1", supplement: [Appendix])
#counter(heading).update(0)
#text(size: 2em)[Appendix]

= Hyperparameter selection <hyperparameter_selection>

_GFlowNet Hyperparameters:_

- Hidden dimension: 64;
- Learning rate: 1e-4;
- Exploration $epsilon$: 0.1;
- Batch size: 32.
_BEN Hyperparameters:_

- RNN hidden dimension: 64;
- Learning rate (Q-network): 1e-4;
- Learning rate (Epistemic network): 1e-4;
- Base dimension ($z_"ep"$): 8;
- Discount factor $gamma$: 0.9;
- Batch size: 32.

= Policy Quality Assessment Results

== GFlowNet

#figure(
    ```
    Chain length 3, KL divergence: 0.00401
    Chain length 5, KL divergence: 0.02062
    Chain length 7, KL divergence: 0.01696
    Chain length 9, KL divergence: 0.00385
    Chain length 11, KL divergence: 0.00701
    Mean KL divergence across all files: 0.01049
    ```,
    caption: [Quality assessment by KL divergence between true and observed reward distributions.]
)

= Statistical Significance Tests

#show figure: set block(breakable: true)

== GFlowNet

#figure(
    include("listings/gflownet/significance_test_gflownet_chain-3.typ"),
    caption: [GFlowNet, Chain Length $n = 3$.]
)

#v(1em)

#figure(
    include("listings/gflownet/significance_test_gflownet_chain-5.typ"),
    caption: [GFlowNet, Chain Length $n = 5$.]
)

#v(1em)

#figure(
    include("listings/gflownet/significance_test_gflownet_chain-7.typ"),
    caption: [GFlowNet, Chain Length $n = 7$.]
)

#v(1em)

#figure(
    include("listings/gflownet/significance_test_gflownet_chain-9.typ"),
    caption: [GFlowNet, Chain Length $n = 9$.]
)

#v(1em)

#figure(
    include("listings/gflownet/significance_test_gflownet_chain-11.typ"),
    caption: [GFlowNet, Chain Length $n = 11$.]
)

== BEN

#figure(
    include("listings/ben/significance_test_ben_chain-3.typ"),
    caption: [BEN, Chain Length $n = 3$.]
)

#v(1em)

#figure(
    include("listings/ben/significance_test_ben_chain-5.typ"),
    caption: [BEN, Chain Length $n = 5$.]
)

#v(1em)

#figure(
    include("listings/ben/significance_test_ben_chain-7.typ"),
    caption: [BEN, Chain Length $n = 7$.]
)

#v(1em)

#figure(
    include("listings/ben/significance_test_ben_chain-9.typ"),
    caption: [BEN, Chain Length $n = 9$.]
)

#v(1em)

#figure(
    include("listings/ben/significance_test_ben_chain-11.typ"),
    caption: [BEN, Chain Length $n = 11$.]
)
