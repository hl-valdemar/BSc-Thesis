#import "@preview/showybox:2.0.3": showybox

#set page(
    header: context {
      if counter(page).get().first() > 1 [
        #grid(
          columns: (1fr, 1fr, 1fr),
          box(width: 100%, align(left)[Valdemar H. Lorenzen]),
          box(width: 100%, align(center)[BSc Thesis]),
          box(width: 100%, align(right)[IMADA, SDU]),
        )
      ]
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
#set text(size: 11pt)
#set par(
  //first-line-indent: 1em,
  //spacing: 1em,
  //justify: true,
)

#let show_notes = true

#let note(body, fill: yellow) = {
    if show_notes {
        set text(black)
        block(
          fill: fill,
          inset: 12pt,
          breakable: true,
          [_NOTE:_\ \ #body],
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
            [_TODO:_\ \ #body],
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
}

#let make_abstract(content) = {
    align(center)[
        #text(size: 16pt, weight: 900)[Abstract]
        \ \
        #content
    ]
}

#note[
- Use consistent mathematical notation throughout
- Include clear figures and diagrams
- Provide code snippets for key implementations
- Reference related work appropriately
- Writing style: maintain an academic tone while ensuring readability. Use precise technical language but explain complex concepts clearly. Include examples and visualizations to aid understanding.
]

#note[
    Maybe there is not enough natural stochasticiy in the environment's reward-state transition dynamics to make this interesting.
    Maybe this could remidied by:
    - Making the chain length random between episodes?
    - Making the reward for reaching a terminal state random following a pre-determined distribution (by sampling the rewards from pre-determined distributions corresponding to each terminal state, we can still make assumptions about the desired behavior --- bigger mean rewards $=>$ higher desired sample rates for trained model).
]

#todo[Add title page.]
#todo[Add an abstract.]

#let title = "BSc Thesis"
#let authors = (
    "Valdemar H. Lorenzen",
)
#let supervisors = (
    "Melih Kandemir",
)

#make_title(title, authors, supervisors)
#v(1cm)
#make_abstract[Abstract goes here.]

//#set page(columns: 2)

#pagebreak()
#counter(page).update(1)

#outline()

= Introduction <introduction>

Many real-world applications present a fundamental challenge that current reinforcement learning (RL) methods struggle to address effectively: the problem of delayed and sparse rewards.

#attention([Delayed and Sparse Rewards])[
    Learning scenarios where meaningful feedback signals (rewards) are provided only far after a long sequence of actions, and where most actions yeild no immediate feedback.

    _Example: In drug discovery, the effectiveness of a designed molecule can only be evaluated after its complete synthesis, with no intermediate feedback during the design process._
]

Consider, for instance, the process of drug design, where a reinforcement learning agent must make a series of molecular modifications to create an effective compound.
The value of these decisions --- the drug's efficacy --- can only be assessed once the entire molecule is complete.
Similarly, in robotics tasks like assembly or navigation, success often depends on precise sequences of actions where feedback is only available upon having completed the entire task.

Traditional reinforcement learning algorithms face two critical limitations in such environments:

+ *Credit Assignment:* When rewards are delayed, the algorithm struggles to correctly attribute success or failure to specific actions in a long sequence.
    This is analogous to trying to improve a chess strategy when only knowing the game's outcome, without understanding which moves were actually decisive.

+ *Exploration Efficiency:* With sparse rewards, random exploration becomes highly inefficient.
    An agent might need to execute precisely the right sequence of actions to receive any feedback at all, making random exploration about as effective as searching for a needle in a haystack.

This thesis investigates a novel approach to addressing these challenges through the comparison of two promising methodologies: *Generative Flow Networks* (GFlowNets) @bengio2021flownetworkbasedgenerative and *Bayesian Exploration Networks* (BEN) @fellows2024bayesianexplorationnetworks.
These approaches represent fundamentally different perspectives on handling uncertainty and exploration in reinforcement learning:

+ GFlowNets frame the learning process as a flow network, potentially offering more robust learning in situations with multiple viable solutions.

+ BENs leverages Bayesian uncertainty estimation to guide exploration more efficiently, potentially making better use of limited feedback.

By comparing these approaches, we aim to understand their relative strengths and limitations in environments with delayed and sparse rewards, #maybe[ultimately contributing to the development of more efficient and practical reinforcement learning algorithms].
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

#note[Check that these titles correctly correspond to their reference.]

*@preliminaries: Preliminaries* provides the theoretical foundations of reinforcement learning and explores existing approaches to handling sparse rewards.
This chapter establishes the mathematical framework and notation used throughout the thesis.

*@theoretical_framework: Theoretical Framework* presents our hypothesis and analytical approach.
#maybe[We develop the mathematical foundations for comparing GFlowNets and BEN, with particular attention to their theoretical guarantees and limitations.]

*@experimental_design: Experimental Design* details our testing methodology, including environment specifications, evaluation metrics, and implementation details.
#maybe[This chapter ensures reproducibility and clarity in our experimental approach.]

*@results_and_analysis: Results and Analysis* presents our findings, including both quantitative performance metrics #maybe[and qualitative analysis of learning behaviors].
We examine how each algorithm handles the exploration-exploitation trade-off and adapts to varying levels of reward sparsity.

*@future_research: Future Research* #maybe[...]

*@conclusion: Conclusion* summarizes our findings, discusses their implications for the field, and suggests directions for future research.


= Preliminaries <preliminaries>

#note[
- Fundamentals of reinforcement learning
    - Q-learning and temporal difference methods
- Sparse reward challenges
- Survey of existing approaches
    - GFlowNets
    - Deep exploration networks (BEN)
    - Comparison of methodologies
]

== Flow Networks

GFlowNets @bengio2021flownetworkbasedgenerative rely on the concept of flow networks. A flow network is represented as a directed acyclic graph $G = (cal(S), cal(A))$, where $cal(S)$ represents the state space and $cal(A)$ represents the action space.

#attention([Flow Network])[
    A directed acyclic graph with a single source node (initial state) and one or more sink nodes (terminal states), where flow is conserved at each intermediate node.

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

The _trajectory flow function_ $F: cal(T) -> RR_(>= 0)$ assigns a non-negative value to each possible trajectory.
From this flow function, two important quantities are derived:

+ *State flow*: For any state $s$, its flow is the sum of flows through all trajectories passing through it: $ F(s) = sum_(s in tau) F(tau). $

+ *Edge flow*: For any action (edge) $s -> s'$, its flow is the sum of flows through all trajectories using that edge: $ F(s -> s') = sum_(tau = (... -> s -> s' -> ...)) F(tau). $

These flows must satisfy a conservation principle known as the _flow matching constraint_:

#attention([Flow Matching])[
    For any non-terminal state $s$, the total incoming flow must equal the total outgoing flow: $ F(s) & = sum_((s'' -> s) in cal(A)) F(s'' -> s) & = sum_((s -> s') in cal(A)) F(s -> s'). $
]

=== Markovian Flow

The flow function induces a probability distribution over trajectories.
Given a flow function $F$, we define $P(tau) = 1 / Z F(tau)$, where $Z = F(s_0) = sum_(tau in cal(T)) F(tau)$ is the _partition function_ @malkin2023trajectorybalanceimprovedcredit --- i.e., the total flow through the network.

#attention([Markovian Flow])[
    A flow is _Markovian_ when it can be factored into local decisions at each state.
    This occurs when the following exist @malkin2023trajectorybalanceimprovedcredit:

    + Forward policies $P_F (-|s)$ over children of each non-terminal state s.t. $ & P(tau = (s_0 -> ... -> s_n)) & = product_(t = 1)^n P_F (s_t|s_(t - 1)). $

    + Backward policies $P_B (-|s)$ over parents of each non-initial state s.t. $ & P(tau = (s_0 -> ... -> s_n)|s_n = x) & = product_(t = 1)^n P_B (s_(t - 1)|s_(t)). $
]

The Markovian property allows us to decompose complex trajectory distributions into simple local decisions, making learning tractable while maintaining the global flow constraints #citation_needed().


== GFlowNets

GFlowNets @bengio2021flownetworkbasedgenerative are an approach to learning policies that sample from desired probability distributions.
They frame the learning process as discovering a flow function that makes the probability of generating any particular object proportional to its reward.

Given a reward function $R: cal(X) -> RR_(>= 0)$ defined over the set of terminal states $cal(X)$, GFlowNets aim to approximate a Markovian flow $F$ on the graph $G$ s.t. $F(x) = R(x)$ for all $x in cal(X)$.

#attention([GFlowNet])[
    @malkin2023trajectorybalanceimprovedcredit defines a GFlowNet as any learning algorithm that discovers flow functions matching terminal state rewards, consisting of: 

    + A model that outputs:
        - Initial state flow $Z = F(s_0)$;
        - Forward action distributions $P_F (-|s)$ for non-terminal states.
    + An objective function that, when globally minimized, guarantees $F(x) = R(x)$ for all terminal states.

    _Example: In molecular design, this ensures that high-reward molecules are genered more frequently, while maintaining diversity through exploration of multiple pathways._
]

The power of GFlowNets lies in their ability to handle situations where multiple action sequences can lead to the same terminal state --- a common scenario in real-world applications like molecular design #maybe[or image synthesis]. 
Unlike traditional RL methods that focus on finding a single optimal path, GFlowNets learn a distribution over all possible paths proportional to their rewards.

=== Learning Process

The learning process of GFlowNets involves iteratively improving both flow estimates and the policies.
The forward policy of a GFlowNet can sample trajectories from the learned Markovian flow $F$ by sequentially selecting actions according to $P_F (-|s)$.
When the training converges to a global minimum of the objective function, this sampling process guarantees that $P(x) prop R(x)$.
#maybe[That is, the probability of generating any terminal state $x$ is proportional to its reward $R(x)$.]
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

#attention("Markov Decision Process")[
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
An optimal policy $pi^* in Pi^* := op("arg max", limits: #true)_(pi) J^pi$ can be found through the optimal value function $V^*: cal(S) -> RR$ or the optimal action-value function $Q^*: cal(S) times cal(A) -> RR$, which satisfy the Bellman optimality equations
$ V^* (s) = max_(a in cal(A)) Q^* (s, a), $
$ Q^* (s, a) = EE_(r, s' ~ P_(R,S)(s,a))[r + gamma max_(a' in cal(A)) Q^* (s', a')]. $

This framework serves as the building block for more sophisticated models like Contextual MDPs and Bayesian approaches to reinforcement learning, as described in the following sections.

== Contextual Reinforcement Learning

In contextual RL, we use the concept of a Contextual MDP.

#attention("Contextual MDP")[
    A Markov Decision Process augmented with a context variable that determines the specific dynamics of the environment.
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

In the Bayesian approach to RL, rather than viewing uncertainty as a problem to be eliminated, it becomes an integral part of the decision-making process --- something to be reasoned about systematically.

#attention("Bayesian Epistemology")[
    A framework that characterizes uncertainty through probability distributions over possible worlds.
    In reinforcement learning, this means maintaining distributions over possible MDPs, updated as new evidence arrives.
]

=== From Prior to Posterior

The Bayesian learning process begins with a _prior distribution_ $P_Phi$ representing our initial beliefs about the true context $phi^*$ before any observations.
As the agent interacts with the environment, it accumulates a history of experiences $h_t$ and updates these beliefs through Bayesian inference, forming a _posterior distribution_ $P_Phi (h_t)$.

This history-dependent posterior in Bayesian RL differentiates it from traditional RL approaches.

#attention("History-Conditioned Policies")[
    Unlike traditional RL policies that map states to actions, Bayesian policies operate on entire histories, defining a set of history-conditioned policies $Pi_cal(H) := {pi: cal(H) -> cal(P)(cal(A))}$, where $cal(H) := { cal(H)_t|t >= 0 }$ denotes the set of all histories.
]

Where the prior $P_Phi$ represents our initial uncertainty (the special case where $h_t = emptyset$), the posterior $P_Phi (h_t)$ captures our refined beliefs after observing interactions with the environment.
This allows us to reason about future outcomes by _marginalizing_ across all possible MDPs according to our current uncertainty.

=== The Bayesian Perspective on Transitions

The power of the Bayesian approach stems from how it handles state transitions.
Instead of committing to a single model of the environment, it maintains a distribution over possible transitions through the _Bayesian state-reward transition distribution_ $ P_(R, S) (h_t, a_t) := EE_(phi ~ P_Phi (h_t)) [P_(R, S) (s_t, a_t, phi)]. $

This distribution lets us reason about future trajectories using the _prior predictive distribution_ $P^pi_t$ with density $ p^pi_t (h_t) = p_0 (s_0) product_(i = 0)^t pi (a_i|h_i) p (r_i, s_(i + 1)|h_i, a_i). $

#attention("Belief Transitions")[
    The evolution of beliefs about the environment can itself be viewed as a transition system.
    This leads to the concept of a _Bayes-adaptive MDP_ (BAMDP) @duff2002optimallearningcomputationalprocedures.
]

The belief transition distribution $P_cal(H) (h_t, a_t)$ captures how our beliefs evolve with new observations, with density $ p_cal(H)(h_(t+1)|h_t, a_t) = p(s_(t+1), r_t|h_t, a_t). $

This formulation leads to the definition of the Bayes-adaptive MDP: $ cal(M)_"BAMDP" := angle.l cal(H), cal(A), P_0, P_cal(H)(h,a), gamma angle.r. $

=== Natural Resolution of the Exploration Dilemma

An interesting aspects of the Bayesian framework is how it naturally resolves the exploration-exploitation dilemma.
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
Bayesian Exploration Networks (BENs) extend this idea into the Bayesian realm by characterizing uncertainty in the Bellman operator itself, instead of in the environment's transition dynamics.

#attention("Model-Free vs Model-Based")[
    While model-based approaches maintain explicit probabilistic models of the environment's dynamics, model-free methods like BEN directly learn mappings from states to values or actions.
    This can be more computationally efficient but requires careful handling of uncertainty.
]

=== The Bootstrapping Perspective

Instead of modeling the full complexity of state transitions, we can use bootstrapping to estimate the optimal Bayesian Bellman operator directly.
Given samples from the true reward-state distribution $r_t, s_(t+1) ~ P^*_(R,S) (s_t, a_t)$, we estimate $b_t = beta_omega (h_(t+1)) := r_t + gamma max_a' Q_omega (h_(t+1), a').$

This bootstrapping process can be viewed as a transformation of variables --- mapping from the space of rewards and next states to a single scalar value.
This significantly reduces the dimensionality of the problem while preserving the essential information needed for learning optimal policies @fellows2024bayesianexplorationnetworks.

#attention("Bootstrapped Distribution")[
    The samples $b_t$ follow what we call the Bellman distribution $P^*_B (h_t, a_t; omega)$, which captures the distribution of possible Q-value updates.
    This distribution encapsulates both the environment's inherent randomness and our uncertainty about its true nature.
]

=== Sources of Uncertainty

When predicting future Q-values, BEN distinguishes between two types of uncertainty:

+ *Aleatoric Uncertainty*: The inherent randomness in the environment's dynamics that persists even with perfect knowledge.

    _Example: Rolling a fair die --- this uncertainty cannot be reduced with more data._

+ *Epistemic Uncertainty*: Our uncertainty about the true Bellman distribution itself.
    This represents our lack of knowledge about the environment and can be reduced through exploration and learning.

    _Example: Determining whether a die is fair --- this uncertainty is can be reduced with more data._

This separation of uncertainties allows BEN to distinguish between what is fundamentally unpredictable (aleatoric) and what can be learned through exploration (epistemic), leading to more efficient learning strategies.

=== Network Architecture

BEN implements this uncertainty handling through three neural networks:
+ *Recurrent Q-Network*:

    At its core, BEN uses a recurrent neural network (RNN) to approximate the optimal Bayesian Q-function.
    The Q-network processes the entire history of interactions.
    We denote the output at timestep $t$ as $q_t = Q_omega (h_t, a_t) = Q_omega (hat(h)_(t-1), o_t)$, where $h_t$ represents the history up to time $t$, $a_t$ is the action, $hat(h)_(t-1)$ is the recurrent encoding of previous history, and $o_t$ contains the current observation tuple ${r_(t-1), s_t, a_t}$.
    By conditioning on history rather than just current state, BENs can capture how uncertainty evolves over time, making it capable of learning Bayes-optimal policies.

+ *Aleatoric Network*:

    The aleatoric network models the inherent randomness in the environment.
    It uses normalizing flows to transform a simple base distribution (such as a standard Gaussian) into a more complex distribution $P_B (h_t, a_t, phi; omega)$ over possible next-state Q-values, representing the aleatoric uncertainty in the Bellman operator, by applying the transformation $b_t = B(z_"al", q_t, phi)$, where 

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
    
    This optimization is performed indirectly by maximizing the Evidence Lower Bound (ELBO) $"ELBO"(psi; h, omega)$, which is equivalent @fellows2024bayesianexplorationnetworks.

#note[We use the concept of variational inference without defining it]

=== Training Process

The network is trained through a dual optimization process:

+ *MSBBE Optimization:* The Mean Squared Bayesian Bellman Error (MSBBE) is computed as the difference between the predictive optimal Bellman operator $B^+ [Q_omega]$ and $Q_omega$: $ "MSBBE"(omega; h_t, psi) := norm(B^+ [Q_omega] (h_t, a_t) - Q_omega (h_t, a_t))^2_rho, $ which is minimized to learn the parametrisation $omega^*$, satisfying the optimal Bayesian Bellman equation for our Q-function approximator, with $rho$ being an arbitrary sampling distribution with support over $cal(A)$.
    
    The predictive optimal Bellman operator can be obtained by taking expectations over variable $b_t$ using the predictive optimal Bellman distribution $P_B (h_t, a_t; omega)$: $ B^+ [Q_omega] (h_t, a_t) := EE_(b_t ~ P_B (h_t, a_t; omega)) [b_t], $ where $P_B (h_t, a_t; omega) = EE_(phi ~ P_Phi (cal(D)_omega (h_t))) [P_B (h_t, a_t, phi; omega)]$.

    This gives rise to a nested optimisation problem, as is common in model-free RL @fellows2024bayesianexplorationnetworks, which can be solved using two-timescale stochastic approximation.
    In this case, we update the epistemic network parameters $psi$ using gradient descent on an asymptotically faster timescale than the function approximator parameters $omega$ to ensure convergence to a fixed point @fellows2024bayesianexplorationnetworks.

+ *ELBO Optimization:* The Evidence Lower BOund (ELBO) serves as the optimization objective for training BEN's epistemic network.
    While minimizing the KL-divergence $"KL"(P_psi || P_Phi (cal(D)_omega (h_t)))$ directly would give us the most accurate approximation of the true posterior, computing this divergence is typically intractable.
    Instead, we can derive and optimize the ELBO, which provides a tractable lower bound on the model evidence.
    
    Starting with the definition of the KL-divergence and applying Bayes' rule, @fellows2024bayesianexplorationnetworks derives $ "ELBO"&(psi; h_t, omega) \ &:= EE_(z_"ep" ~ P_"ep") [ sum_(i=0)^(t-1) ( B^(-1)(b_i, q_i, phi)^2 - log bar.v partial_b B^(-1)(b_i, q_i, phi) bar.v ) - log p_Phi (phi) ], $ where $phi = t_psi (z_"ep")$ and:
    
    - $z_"ep"$ is drawn from the base distribution $P_"ep"$ (a standard Gaussian $cal(N)(0, I^d)$);
    - $B^(-1)$ is the inverse of the aleatoric network's transformation;
    - $partial_b B^(-1)$ is the Jacobian of this inverse transformation;
    - $t_psi$ represents the epistemic network's transformation.
    
    #attention("Jacobian Term")[
        The term $partial_b B^(-1)$ accounts for how the epistemic network's transformation changes the volume of probability space.
        This is important for maintaining proper probability distributions when using normalizing flows.
    ]
    
    The ELBO objective breaks down into three key components:
    
    + A reconstruction term $B^(-1)(b_i, q_i, phi)^2$ that measures how well our model can explain the observed Q-values;
    + A volume correction term $log|partial_b B^(-1)(b_i, q_i, phi)|$ that accounts for the change in probability space;
    + A prior regularization term $log p_Phi (phi)$ that encourages the approximated posterior to stay close to our prior beliefs.
    
    By minimizing the ELBO, we obtain an approximate posterior that balances accuracy with computational tractability, allowing BEN to maintain and update its uncertainty estimates efficiently during learning.

#attention("Training Dynamics")[
    The two optimization processes occur at different timescales, with epistemic updates happening more frequently than the Q-network updates.
    This separation ensures stable convergence while maintaining the ability to adapt to new information.
]

With this architecture, BEN can learn truly Bayes-optimal policies while maintaining the computational efficiency of model-free methods.
This makes it particularly well-suited for environments with sparse, delayed rewards where efficient exploration is important.

= Theoretical Framework <theoretical_framework>

#note[
- Hypothesis development
- Problem formulation
    - Mathematical notation and definitions
    - Assumptions and constraints
- Proposed solution approach
]

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

=== Analytical Framework

To evaluate our hypothesis about the relative performance of GFlowNets and BENs in delayed reward environments, we establish three metrics that capture different aspects of learning and exploration efficiency.

+ *Sample Efficiency:* Measures how quickly each algorithm converges to optimal behavior through their respective loss functions.

    For GFlowNets, we track the trajectory balance loss $cal(L)_"TB" (tau) = (log Z_theta + log sum_(t = 1)^n P_F (s_t|s_(t - 1); theta) - log R(x) - log sum_(t = 1)^n P_B (s_(t - 1)|s_t; theta))^2$.

    While for BEN, we monitor the Mean Squared Bayesian Bellman Error $"MSBBE"(omega; h_t, psi) = norm(B^+ [Q_omega] (h_t, a_t) - Q_omega (h_t, a_t))^2_rho$.

    These metrics allow us to quantify learning progress and compare convergence rates between algorithms as a function of reward delay $T_"reward"$.

+ *Distribution Matching:* Evaluates how well the learned policy matches the true underlying reward structure.

    In our n-chain environment, where terminal states are guaranteed to be reached, we measure the KL divergence between the true terminal state distribution $P$ (determined by rewards) and the empirical distribution $Q$ generated by each algorithm $"KL"(P || Q)$.
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



//== Hypothesis
//
//Our hypothesis is that BEN's Bayesian exploration strategy leads to more efficient learning compared to GFlowNets in environments with highly delayed rewards, particularly during the early stages of training.
//We formalize this hypothesis through two claims:
//
//+ In environments with reward delay $d$, BEN requires $cal(O)(f(d))$ samples to achieve near-optimal performance, while GFlowNets require $cal(O)(g(d))$ samples, where $f(d)$ grows asymptotically slower than $g(d)$.
//
//+ This efficiency advantage diminishes as the delay between actions and rewards decreases, which would suggests that the benefits of explicit uncertainty modelling becomes less critical in environments with more immediate feedback.
//
//== Delay and Sparsity Formalization
//
//We define a reward function $R$ as $(d, s)$-delayed-sparse, where
//
//- $d in NN$ represents the minimum number of actions required before any non-zero reward;
//- $s in [0, 1]$ represents the sparsity ratio: the proportion of trajectories that yield non-zero rewards.
//
//For our n-chain environment with length $n$ and branching factor $b$, we have $d = n$, and $s = b / (|cal(A)|^n)$.
//That is, the delay equals the chain length, and the sparsity ratio depends on branching factor and action space size.
//
//== Information Processing Rate
//
//To compare the learning efficiency of both approaches, we introduce the concept of Random Deviation Index (RDI): $ "RDI"(t) = "KL"(pi_t || pi_"random"), $ where $pi_t$ is the policy at time $t$, $pi_"random"$ is a uniform random policy. //@kullback1951oninformationandsufficiency.
//This metric captures how quickly each method moves away from random exploration toward informed decision-making.
//
//== Uncertainty Representation
//
//We analyze how each method represents uncertainty.
//
//+ GFlowNets implicitly handle uncertainty through flow matching: $F(x) = R(x)$ for terminal states $x in cal(X)$.
//
//+ BEN explicitly models both aleatoric uncertainty through the Bellman distribution $P_B (h_t, a_t, phi; omega)$, and epistemic uncertainty through the approximation $P_psi$ of the posterior $P_Phi (cal(D)_omega (h_t))$ by minimization of $"KL"(P_psi || P_Phi (cal(D)_omega (h_t)))$.
//
= Experimental Design <experimental_design>

#note[
- Test environments
    - N-Chain implementation
- Evaluation metrics
    - Sample efficiency (steps needed to reach optimal policy) - measure by objective loss over training?
    - Final performance (average success/reward rate) - measure by difference between sample distribution and true distributions? Otherwise, this probably doesn't make sense for the semi-deterministic n-chain environment, as we know we'll succeed in n steps.
    - Exploration behavior (state coverage over time) - maybe not so interesting for the simple n-chain environment, but not entirely uninteresting either.
- Implementation details
    - Network architectures
    - Training procedures
        - For GFlowNets, mention tempered exploration during training (off-policy training)
    - Hyperparameter selection
]

== Test Environment

#todo[
    - Describe the n-chain environment:
        - a chain of length $n$
        - the chain has a branch point from which two branches emerge
        - last state is always a terminal state
        - terminal states have a disignated reward

    - Describe the action space:
        - actions include a _forward_ action;
        - a _left branch_ or _right branch_ choice at branch point.

    - #note[The _terminal stay_ action is probably inconsequensial.]
]

== Evaluation Metrics

#todo[
    - Describe how the sample efficiency is measured.
        - Maybe the objective loss over training (how many steps to convergence)
    - Describe the final performance:
        - In my n-chain implementation, a terminal state will always be reached, so average success rate wouldn't make sense as a metric (as success will always be 100%).
        - Instead, for GFlowNets, this should probably be evaluated as the difference from the true distribution (the expected distribution).
        - I.e., with rewards 10 and 5 for two terminal states, the true distribution would be $1/3$ samples with reward 5 and $2/3$ samples with reward 10.
    - Describe exploration behavior:
        - State coverage over time.
        - This will probably require a higher number of branches for any interesting metrics.
]

== Implementation Details

#todo[
    - Network architectures
        - GFlowNets:
            - Multi-layer perceptron for state encoding
                - Input: State tensor of shape [batch_size, state_dim]
                - Output: State encoding of  of shape [batch_size, hidden_dim]
            - Single layer for forward policy
                - Input: State encoding of shape [batch_size, hidden_dim]
                - Output: Forward policy [batch_size, num_actions]
            - Single layer for backward policy
                - Input: State encoding of shape [batch_size, hidden_dim]
                - Output: Backward policy [batch_size, num_actions]
                    - Note: for the n-chain environment, the graph is a directed tree and so each state can have at most one parent, meaning that only one action is possible for each state in the backward policy, as only one action could have lead from the previous state to this state (as actions are represented by edges)
            - Parameter (scalar) for log Z function approximation
    - Training procedures
        - For GFlowNets, mention tempered exploration during training (i.e., off-policy training)
            - Mention epsilon parameter for temperature control in guided exploration
    - Hyperparameter selection
]

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

= Future Research <future_research>

#note[Everything I think I could have done better essentially.]

= Conclusion <conclusion>

#note[
- Summary of contributions
- Key insights
- Future work directions
]

#bibliography("refs.bib")
